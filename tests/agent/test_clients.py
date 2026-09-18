import json
import unittest
from types import SimpleNamespace

import httpx

from agent.llm import DeepSeekClient, GeminiClient, LLMError, parse_json
from agent.search_client import SearchClient, SearchError


def search_client(handler, **kwargs):
    http = httpx.Client(base_url="http://server", transport=httpx.MockTransport(handler))
    return SearchClient("http://server", http=http, sleep=lambda _: None, **kwargs)


class SearchClientTests(unittest.TestCase):
    def test_retries_a_superseded_batch_then_returns_results(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            if len(calls) == 1:
                return httpx.Response(409, json={"detail": "superseded"})
            return httpx.Response(200, json={"results": {"v": {"kind": "visual", "hits": []}}})

        results = search_client(handler).search([{"kind": "visual", "id": "v", "text": "boat"}])
        self.assertEqual(results, {"v": {"kind": "visual", "hits": []}})
        self.assertEqual(len(calls), 2)

    def test_non_retryable_errors_surface_the_server_detail(self):
        client = search_client(lambda request: httpx.Response(422, json={"detail": "bad query"}))
        with self.assertRaisesRegex(SearchError, "bad query"):
            client.search([])

    def test_image_width_snaps_to_an_allowed_thumbnail_size(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, content=b"jpeg")

        client = search_client(handler)
        self.assertEqual(client.image("L01_V001", 5, 600), b"jpeg")
        client.image("L01_V001", 5, 0)
        self.assertEqual(seen, ["http://server/thumb/L01_V001/5?w=640", "http://server/frame/L01_V001/5"])


def deepseek(handler):
    http = httpx.Client(base_url="http://deepseek", transport=httpx.MockTransport(handler))
    return DeepSeekClient("key", http=http, sleep=lambda _: None)


def completion(content="", **message):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content, **message}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


class DeepSeekTests(unittest.TestCase):
    def test_request_carries_thinking_tools_and_json_mode(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return completion('{"ok": true}', reasoning_content="think")

        client = deepseek(handler)
        self.assertEqual(client.chat_json([{"role": "user", "content": "json"}]), {"ok": True})
        message = client.chat([], tools=[{"type": "function"}], thinking=False)
        self.assertEqual(message["reasoning_content"], "think")
        self.assertEqual(bodies[0]["thinking"], {"type": "enabled"})
        self.assertEqual(bodies[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", bodies[0])
        self.assertEqual((bodies[1]["thinking"], bodies[1]["temperature"]), ({"type": "disabled"}, 0.3))
        self.assertEqual(client.usage.deepseek_calls, 2)

    def test_retries_empty_json_and_server_errors(self):
        replies = iter([httpx.Response(503), completion(""), completion('```json\n{"a": 1}\n```')])
        self.assertEqual(deepseek(lambda request: next(replies)).chat_json([]), {"a": 1})

    def test_parse_json_tolerates_prose_around_the_object(self):
        self.assertEqual(parse_json('Here: {"a": [1]} done'), {"a": [1]})
        with self.assertRaises(ValueError):
            parse_json("no json here")


class FakeGenAI:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class GeminiTests(unittest.TestCase):
    def test_judge_labels_images_and_retries_rate_limits(self):
        limited = RuntimeError("429")
        limited.code = 429
        response = SimpleNamespace(text='{"candidates": []}', usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=7))
        fake = FakeGenAI([limited, response])
        gemini = GeminiClient("key", client=fake, sleep=lambda _: None)

        reply = gemini.judge([("[C1] label", b"jpeg")], "prompt", {"type": "object"})

        self.assertEqual(reply, {"candidates": []})
        _, contents, config = fake.calls[-1]
        self.assertEqual(contents[0], "[C1] label")
        self.assertEqual(contents[-1], "prompt")
        self.assertEqual(config["response_json_schema"], {"type": "object"})
        self.assertEqual(gemini.usage.gemini_input, 100)

    def test_non_retryable_errors_raise_llm_error(self):
        denied = RuntimeError("bad key")
        denied.code = 403
        gemini = GeminiClient("key", client=FakeGenAI([denied]), sleep=lambda _: None)
        with self.assertRaises(LLMError):
            gemini.judge([], "prompt", {})

    def test_web_search_returns_grounding_sources(self):
        web = SimpleNamespace(web=SimpleNamespace(title="Wiki", uri="https://example.org"))
        metadata = SimpleNamespace(grounding_chunks=[web], web_search_queries=["jaws 1975"])
        response = SimpleNamespace(text=" Cá mập (shark) ", usage_metadata=None,
                                   candidates=[SimpleNamespace(grounding_metadata=metadata)])
        answer = GeminiClient("key", client=FakeGenAI([response])).web_search("q")
        self.assertEqual(answer.text, "Cá mập (shark)")
        self.assertEqual(answer.sources, (("Wiki", "https://example.org"),))
        self.assertEqual(answer.queries, ("jaws 1975",))


if __name__ == "__main__":
    unittest.main()
