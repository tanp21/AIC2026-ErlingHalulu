"""Thin clients for the two models: DeepSeek plans and drives tools, Gemini looks.

DeepSeek has no vision in this design and no web access; Gemini does both the
image judgement and Google-Search-grounded lookups.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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
    """Running token totals for one agent run, shared across threads.

    Every counter a backend may `add()` has to be declared here: `add` resolves
    each name with `getattr`, so an undeclared one raises rather than being
    silently dropped — which is how `vlm_calls` was caught missing.
    """
    deepseek_input: int = 0
    deepseek_output: int = 0
    gemini_input: int = 0
    gemini_output: int = 0
    gemini_calls: int = 0
    deepseek_calls: int = 0
    #: Judgements made by a local or forwarded vision backend. Token counts are not
    #: reported: the local models do not bill, and the HTTP backend would have to
    #: round-trip its own usage for it to mean anything.
    vlm_calls: int = 0

    def __post_init__(self):
        self._lock = threading.Lock()

    def add(self, **counts):
        with self._lock:
            for name, value in counts.items():
                setattr(self, name, getattr(self, name) + int(value or 0))

    def as_dict(self):
        return {name: getattr(self, name) for name in (
            "deepseek_calls", "deepseek_input", "deepseek_output",
            "gemini_calls", "gemini_input", "gemini_output", "vlm_calls",
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

    def judge(self, images: list[tuple[str, bytes]], prompt: str, schema: dict,
              *, max_output_tokens=8192, job=None, query=None, scenes="", question=None):
        """Labelled JPEGs + instructions -> JSON matching `schema`.

        The same contract as `GeminiClient.judge`, so the five jobs in
        `agent.vision` run unchanged against DeepSeek (`AIC_AGENT_VISION=deepseek`).
        Vision is not documented for this API, but the OpenAI-compatible endpoint
        accepts `image_url` parts and answers correctly; whether it is accurate
        enough to rely on is measured, not assumed —
        `scripts/benchmarks/bench_agent_vision.py` scores it on the same pool as
        Gemini and the local models.

        Thinking is off: a 0-10 score is a bounded judgement, and this model
        otherwise spends its whole output budget on `reasoning_content` and returns
        an empty `content`.
        """
        import base64

        from .vlm import format_instruction

        content: list = []
        for label, data in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
            }})
        content.append({"type": "text", "text": prompt + "\n\n" + format_instruction(schema)})
        message = self.chat([{"role": "user", "content": content}], json_mode=True,
                            max_tokens=max_output_tokens, thinking=False)
        text = message.get("content") or ""
        if not text.strip():
            # `deepseek-v4-pro` does not read images: it answers a vision call with an
            # empty `content`, or with schema-shaped JSON whose every score is 0. Both
            # look like a judge that works and scores badly, so say what actually
            # happened instead.
            raise LLMError(
                f"{self.model} returned no content for a vision request; it may not "
                f"support image input"
            )
        try:
            return parse_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"deepseek returned invalid JSON: {text[:300]}") from exc


@dataclass(frozen=True)
class GroundedAnswer:
    text: str
    sources: tuple[tuple[str, str], ...]
    queries: tuple[str, ...]


class OfflineGemini:
    """Stands in for Gemini when the run has no API key.

    With ``AIC_AGENT_VISION=local`` nothing in the five vision jobs needs Gemini —
    but the web lookup still does, and refusing to start over a missing key would
    make the offline path unusable on the day it exists for. `web_search` raises
    the same `LLMError` a failing call would, which `_lookup` already turns into a
    note on the ranking rather than a failed query.
    """

    def web_search(self, question: str) -> GroundedAnswer:
        raise LLMError("GEMINI_API_KEY is not set; web lookup is unavailable")

    def judge(self, images, prompt, schema, **_):
        raise LLMError("GEMINI_API_KEY is not set and the vision backend is not Gemini")


class GeminiClient:
    def __init__(self, api_key, model="gemini-3.8-flash", *, thinking_level="low",
                 retries=4, max_backoff=30.0, timeout=None, hedge_after=None, usage=None,
                 client=None, sleep=time.sleep):
        if client is None:
            from google import genai
            from google.genai import types

            options = types.HttpOptions(timeout=int(timeout * 1000)) if timeout else None
            client = genai.Client(api_key=api_key, http_options=options)
        self.model = model
        self.thinking_level = thinking_level
        self.usage = usage or Usage()
        self._client = client
        self._retries = retries
        self._max_backoff = max_backoff
        self._sleep = sleep
        self._hedge_after = hedge_after
        self._hedges = ThreadPoolExecutor(max_workers=16) if hedge_after else None

    def _generate(self, contents, config, hedge=True):
        """One generation; with `hedge_after`, a duplicate request races a slow one.

        A judge call normally takes ~4 s, but on live runs one in ~20 took 15-35 s,
        which alone blew a 20 s query budget. The loser keeps running and is ignored.
        """
        if not (hedge and self._hedges):
            return self._generate_once(contents, config)
        first = self._hedges.submit(self._generate_once, contents, config)
        done, _ = wait([first], timeout=self._hedge_after)
        if done:
            return first.result()
        racing = {first, self._hedges.submit(self._generate_once, contents, config)}
        error = None
        while racing:
            done, racing = wait(racing, return_when=FIRST_COMPLETED)
            for future in done:
                if future.exception() is None:
                    return future.result()
                error = future.exception()
        raise error

    def _generate_once(self, contents, config):
        for attempt in range(self._retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
            except Exception as exc:
                if attempt == self._retries or not _gemini_retryable(exc):
                    raise LLMError(f"gemini: {exc}") from exc
                self._sleep(min(self._max_backoff, 2 ** (attempt + 1)))
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
              *, max_output_tokens=8192, job=None, query=None, scenes="", question=None):
        """Labelled JPEGs + instructions -> JSON matching `schema`.

        Each label is emitted as text right before its image so the model can refer
        to candidates by id. `job`, `query`, `scenes` and `question` name the vision
        job and the original Vietnamese inputs; Gemini needs none of them because the
        prompt it is given is already complete — they are what the local backends
        rebuild a prompt they can follow from.
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
        }, hedge=False)  # runs in the background; a duplicate would only cost quota
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
