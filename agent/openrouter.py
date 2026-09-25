"""OpenRouter as a vision judge, behind the same `judge()` contract as the other backends.

Gemini is the measured baseline for looking at a keyframe, but it is a paid quota that can
simply run out mid-competition, and the local VLMs need a card that may already be holding
PE-Core and Qwen3-Embedding. OpenRouter is the third shape: a remote API like Gemini, but
one key across many models, so the judge can be swapped without touching any of the jobs.

The default model is `qwen/qwen3.7-flash` -- text+image+video in, 1M context, structured
output, and cheap enough ($0.03/M input) that a checker call costs well under a cent.

This is deliberately a near-clone of `DeepSeekClient`: both are OpenAI-compatible chat
completions with `image_url` content parts, and keeping the two readable side by side is
worth more than the handful of lines a shared base class would save. Two things differ:

- **The schema goes in `response_format`, not only in the prompt.** OpenRouter supports
  `json_schema`, so the shape is enforced by the provider where it can be.
- **`strict` is off, on purpose.** OpenAI-style strict mode requires `additionalProperties:
  false` on every object and every property listed in `required`; the agent's schemas do not
  satisfy that (`VERIFY_SCHEMA`'s `answer` and `scene` are optional, `agent/vision.py`), and
  a provider that enforces strictness would reject them. So the schema is sent unstrict AND
  rendered into words with `format_instruction`, which is what the local backends already
  rely on -- belt and braces, for the price of a few hundred input tokens.
"""
from __future__ import annotations

import base64
import json
import time

import httpx

from .llm import LLMError, RETRYABLE_STATUS, Usage, parse_json, repair_truncated_json

DEFAULT_MODEL = "qwen/qwen3.7-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
#: Sent for OpenRouter's own app attribution; neither is required and neither carries
#: anything about the person running the query.
APP_URL = "https://github.com/aic-hcmc"
APP_TITLE = "AIC HCMC retrieval"


class OpenRouterClient:
    """OpenAI-compatible chat completions against openrouter.ai."""

    def __init__(self, api_key, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
                 *, timeout=120.0, retries=3, usage=None, http=None, sleep=time.sleep,
                 referer=APP_URL, title=APP_TITLE, temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.usage = usage or Usage()
        self._http = http or httpx.Client(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}",
                     "HTTP-Referer": referer, "X-Title": title},
        )
        self._retries = retries
        self._sleep = sleep

    def close(self):
        self._http.close()

    def chat(self, messages, *, schema=None, schema_name="reply", json_mode=False,
             max_tokens=4096, temperature=None, reasoning=False):
        """Return the assistant message dict (`content`, and `reasoning` when the model emits it).

        `reasoning=False` (the default) sends `{"enabled": false}`, which OpenRouter documents
        as stopping the model spending time or tokens on thinking at all -- distinct from
        `exclude: true`, which still reasons internally (billed, and just as slow) and only
        hides it from the reply. Measured live: `qwen/qwen3.7-flash` with `enabled` unset spent
        its answer on internal reasoning and returned an EMPTY `content` after ~95 s, the same
        failure `DeepSeekClient.chat` disables `thinking` for. A judgement or a checklist is a
        bounded, low-temperature answer; nothing here needs a model to think out loud first.
        `reasoning=True` sends `{"enabled": true}` for a caller that wants it.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "reasoning": {"enabled": bool(reasoning)},
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema_name, "strict": False, "schema": schema}}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(self._retries + 1):
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as exc:
                if attempt == self._retries:
                    raise LLMError(f"openrouter: {exc}") from exc
                self._sleep(2 ** attempt)
                continue
            if response.status_code in RETRYABLE_STATUS and attempt < self._retries:
                self._sleep(2 ** attempt)
                continue
            if response.status_code >= 400:
                raise LLMError(f"openrouter: HTTP {response.status_code} {response.text[:300]}")
            data = response.json()
            # OpenRouter reports an upstream failure as a 200 with an `error` member, so a
            # missing `choices` is a real outcome rather than a malformed reply.
            if not data.get("choices"):
                detail = (data.get("error") or {}).get("message") or str(data)[:200]
                if attempt < self._retries:
                    self._sleep(2 ** attempt)
                    continue
                raise LLMError(f"openrouter: no choices ({detail})")
            usage = data.get("usage") or {}
            self.usage.add(
                openrouter_calls=1,
                openrouter_input=usage.get("prompt_tokens"),
                openrouter_output=usage.get("completion_tokens"),
            )
            message = data["choices"][0]["message"]
            if (schema is not None or json_mode) and not (message.get("content") or "").strip() \
                    and attempt < self._retries:
                continue
            return {key: value for key, value in message.items() if value is not None}
        raise LLMError("openrouter: no usable response")

    def chat_json(self, messages, *, salvage=False, schema=None, **kwargs):
        """The reply as JSON. With `salvage`, a reply cut off by the token limit keeps its complete part.

        `json_mode=True` is accepted and ignored when a `schema` is given: the schema is the
        stronger of the two and both would set the same `response_format`. This is the shape
        `agent.evidence.query_details` calls its client with, so the checker can hand it an
        OpenRouter client where the agent hands it DeepSeek.
        """
        kwargs.pop("json_mode", None)
        kwargs.pop("thinking", None)          # accepted for DeepSeekClient parity; no analogue here
        message = self.chat(messages, schema=schema, json_mode=schema is None, **kwargs)
        return _as_json(message.get("content", ""), salvage=salvage)

    def judge(self, images: list[tuple[str, bytes]], prompt: str, schema: dict,
              *, max_output_tokens=8192, job=None, query=None, scenes="", question=None,
              salvage=False):
        """Labelled JPEGs + instructions -> JSON matching `schema`.

        The same contract as `GeminiClient.judge`, so the five jobs in `agent.vision` and the
        checklist's `vision_summarize` run unchanged against OpenRouter
        (`AIC_AGENT_VISION=openrouter`).

        `job`, `query`, `scenes` and `question` are accepted and unused, exactly as Gemini
        accepts them: the prompt handed in is already complete, and they exist for the local
        backends that rebuild a short prompt of their own.

        `salvage` keeps the complete part of a reply the token limit cut off, for callers
        whose answer is a list of separate judgements (the checklist's is).
        """
        from .vlm import format_instruction

        content: list = []
        for label, data in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(data).decode(),
            }})
        content.append({"type": "text", "text": prompt + "\n\n" + format_instruction(schema)})
        message = self.chat([{"role": "user", "content": content}], schema=schema,
                            schema_name=job or "reply", max_tokens=max_output_tokens)
        text = message.get("content") or ""
        if not text.strip():
            # Say what actually happened rather than letting an empty reply read as a judge
            # that works and scores nothing -- the failure mode `deepseek-v4-pro` showed.
            raise LLMError(
                f"{self.model} returned no content for a vision request; it may not "
                f"support image input"
            )
        return _as_json(text, salvage=salvage)


def _as_json(text: str, *, salvage: bool):
    try:
        return parse_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        if salvage:
            try:
                return repair_truncated_json(text)
            except ValueError:
                pass
        raise LLMError(f"openrouter returned invalid JSON: {(text or '')[:300]}") from exc
