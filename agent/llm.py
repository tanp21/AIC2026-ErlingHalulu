"""Thin clients for the two models: DeepSeek plans and drives tools, Gemini looks.

DeepSeek has no vision in this design and no web access; Gemini does both the
image judgement and Google-Search-grounded lookups.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import threading
import time

import httpx

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    """Running token totals for one agent run, shared across threads."""
    deepseek_input: int = 0
    deepseek_output: int = 0
    gemini_input: int = 0
    gemini_output: int = 0
    gemini_calls: int = 0
    deepseek_calls: int = 0

    def __post_init__(self):
        self._lock = threading.Lock()

    def add(self, **counts):
        with self._lock:
            for name, value in counts.items():
                setattr(self, name, getattr(self, name) + int(value or 0))

    def as_dict(self):
        return {name: getattr(self, name) for name in (
            "deepseek_calls", "deepseek_input", "deepseek_output",
            "gemini_calls", "gemini_input", "gemini_output",
        )}


def parse_json(text: str):
    """JSON from a model reply, tolerating a ```json fence or leading prose."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start:end + 1])


class DeepSeekClient:
    """OpenAI-compatible chat completions against api.deepseek.com."""

    def __init__(self, api_key, model="deepseek-flash", base_url="https://api.deepseek.com",
                 *, thinking=True, timeout=180.0, retries=3, usage=None, http=None,
                 sleep=time.sleep):
        self.model = model
        self.thinking = thinking
        self.usage = usage or Usage()
        self._http = http or httpx.Client(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._retries = retries
        self._sleep = sleep

    def chat(self, messages, *, tools=None, json_mode=False, max_tokens=8192, thinking=None):
        """Return the assistant message dict (content, tool_calls, reasoning_content).

        In thinking mode with tools, DeepSeek requires every earlier assistant message
        to be sent back with its `reasoning_content`, so callers append the returned
        dict to `messages` unchanged.
        """
        thinking = self.thinking if thinking is None else thinking
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        if not thinking:
            body["temperature"] = 0.3
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(self._retries + 1):
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if attempt == self._retries:
                    raise LLMError(f"deepseek: {exc}") from exc
                self._sleep(2 ** attempt)
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self._retries:
                self._sleep(2 ** attempt)
                continue
            if response.status_code >= 400:
                raise LLMError(f"deepseek: HTTP {response.status_code} {response.text[:300]}")
            data = response.json()
            usage = data.get("usage") or {}
            self.usage.add(
                deepseek_calls=1,
                deepseek_input=usage.get("prompt_tokens"),
                deepseek_output=usage.get("completion_tokens"),
            )
            message = data["choices"][0]["message"]
            # JSON mode can come back empty (documented DeepSeek caveat); retry it.
            if json_mode and not (message.get("content") or "").strip() and attempt < self._retries:
                continue
            return {key: value for key, value in message.items() if value is not None}
        raise LLMError("deepseek: no usable response")

    def chat_json(self, messages, **kwargs):
        message = self.chat(messages, json_mode=True, **kwargs)
        try:
            return parse_json(message.get("content", ""))
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"deepseek returned invalid JSON: {message.get('content', '')[:300]}") from exc


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    sources: tuple[tuple[str, str], ...]
    queries: tuple[str, ...]


class GeminiClient:
    def __init__(self, api_key, model="gemini-3.8-flash", *, thinking_level="low",
                 retries=4, usage=None, client=None, sleep=time.sleep):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self.model = model
        self.thinking_level = thinking_level
        self.usage = usage or Usage()
        self._client = client
        self._retries = retries
        self._sleep = sleep

    def _generate(self, contents, config):
        for attempt in range(self._retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
            except Exception as exc:
                if attempt == self._retries or not _gemini_retryable(exc):
                    raise LLMError(f"gemini: {exc}") from exc
                self._sleep(min(30.0, 2 ** (attempt + 1)))
                continue
            usage = getattr(response, "usage_metadata", None)
            self.usage.add(
                gemini_calls=1,
                gemini_input=getattr(usage, "prompt_token_count", 0),
                gemini_output=getattr(usage, "candidates_token_count", 0),
            )
            return response
        raise AssertionError("unreachable")

    def judge(self, images: list[tuple[str, bytes]], prompt: str, schema: dict,
              *, max_output_tokens=8192):
        """Labelled JPEGs + instructions -> JSON matching `schema`.

        Each label is emitted as text right before its image so the model can refer
        to candidates by id.
        """
        from google.genai import types

        contents = []
        for label, data in images:
            contents.append(label)
            contents.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
        contents.append(prompt)
        response = self._generate(contents, {
            "temperature": 0.0,
            "thinking_config": {"thinking_level": self.thinking_level},
            "response_mime_type": "application/json",
            "response_json_schema": schema,
            "max_output_tokens": max_output_tokens,
            "automatic_function_calling": {"disable": True},
        })
        try:
            return parse_json(response.text)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise LLMError(f"gemini returned invalid JSON: {(response.text or '')[:300]}") from exc

    def web_search(self, question: str) -> GroundedAnswer:
        """Answer from Google Search grounding, with the sources it cited."""
        from google.genai import types

        response = self._generate(question, {
            "temperature": 0.0,
            "tools": [types.Tool(google_search=types.GoogleSearch())],
            "automatic_function_calling": {"disable": True},
        })
        metadata = None
        if getattr(response, "candidates", None):
            metadata = getattr(response.candidates[0], "grounding_metadata", None)
        sources = []
        for chunk in getattr(metadata, "grounding_chunks", None) or []:
            web = getattr(chunk, "web", None)
            if web is not None:
                sources.append((getattr(web, "title", "") or "", getattr(web, "uri", "") or ""))
        queries = tuple(getattr(metadata, "web_search_queries", None) or ())
        return GroundedAnswer((response.text or "").strip(), tuple(sources[:5]), queries)


def _gemini_retryable(error):
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if isinstance(code, int):
        return code in RETRYABLE_STATUS
    return isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError))
