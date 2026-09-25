import json
import unittest

import httpx

from agent.llm import LLMError, Usage
from agent.openrouter import DEFAULT_MODEL, OpenRouterClient

SCHEMA = {
    "type": "object",
    "properties": {"details": {"type": "array", "items": {"type": "object", "properties": {
        "detail": {"type": "integer"}, "verdict": {"type": "string"},
    }, "required": ["detail", "verdict"]}}},
    "required": ["details"],
}


def client(handler, **kwargs):
    http = httpx.Client(base_url="http://router", transport=httpx.MockTransport(handler))
    return OpenRouterClient("sk-or-test", http=http, sleep=lambda _: None, **kwargs)


def reply(content, *, usage=None):
    return httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": usage or {"prompt_tokens": 17, "completion_tokens": 5},
    })


class JudgeRequestTests(unittest.TestCase):
    def test_each_label_is_a_text_part_right_before_its_own_image(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"details": []}')

        client(handler).judge(
            [("[M] this moment itself", b"one"), ("[M] shot 3s later in the same story", b"two")],
            "judge it", SCHEMA)

        parts = seen["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "[M] this moment itself"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(parts[2]["text"], "[M] shot 3s later in the same story")
        self.assertEqual(parts[3]["type"], "image_url")
        self.assertEqual(parts[4]["type"], "text")   # the prompt comes last

    def test_the_schema_travels_unstrict_and_is_also_rendered_into_the_prompt(self):
        # Strict mode would need `additionalProperties: false` and every property in `required`,
        # which the agent's schemas do not satisfy -- a provider enforcing it would reject them.
        # So the schema is sent unstrict AND spelled out in words, the way the local backends
        # are already told what to emit.
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"details": []}')

        client(handler).judge([("[M] a frame", b"x")], "judge it", SCHEMA, job="check")

        response_format = seen["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertFalse(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["name"], "check")
        self.assertEqual(response_format["json_schema"]["schema"], SCHEMA)
        prompt = seen["messages"][0]["content"][-1]["text"]
        self.assertIn("judge it", prompt)
        self.assertIn("details", prompt)

    def test_reasoning_is_off_by_default(self):
        # Measured live: qwen/qwen3.7-flash with reasoning left unset spent ~95s thinking and
        # returned an empty content -- the same failure DeepSeekClient disables thinking for.
        # `enabled: false` (not `exclude: true`) is what actually stops the spend, per OpenRouter's
        # own docs: exclude still reasons internally, billed and slow, and only hides the text.
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"details": []}')

        client(handler).judge([("[M] a frame", b"x")], "p", SCHEMA)
        self.assertEqual(seen["reasoning"], {"enabled": False})

    def test_reasoning_can_be_turned_on_for_a_caller_that_wants_it(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"details": []}')

        client(handler).chat([{"role": "user", "content": "think about it"}], reasoning=True)
        self.assertEqual(seen["reasoning"], {"enabled": True})

    def test_the_default_model_is_the_vision_capable_one(self):
        self.assertEqual(DEFAULT_MODEL, "qwen/qwen3.7-flash")
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"details": []}')

        client(handler).judge([("[M] a frame", b"x")], "p", SCHEMA)
        self.assertEqual(seen["model"], "qwen/qwen3.7-flash")
        self.assertEqual(seen["temperature"], 0.0)

    def test_the_named_vision_jobs_arguments_are_accepted_and_ignored(self):
        # The five jobs in agent/vision.py pass these; Gemini ignores them too, because the
        # prompt they hand over is already complete.
        parsed = client(lambda request: reply('{"details": []}')).judge(
            [("[M] a frame", b"x")], "p", SCHEMA,
            job="verify", query="con thuyền đỏ", scenes="s", question="mấy con?")
        self.assertEqual(parsed, {"details": []})


class JudgeReplyTests(unittest.TestCase):
    def test_a_fenced_reply_is_parsed(self):
        parsed = client(lambda r: reply('```json\n{"details": [{"detail": 0, "verdict": "exact"}]}\n```')) \
            .judge([("[M] a frame", b"x")], "p", SCHEMA)
        self.assertEqual(parsed["details"], [{"detail": 0, "verdict": "exact"}])

    def test_salvage_keeps_the_complete_part_of_a_reply_the_cap_cut_off(self):
        cut = '{"details": [{"detail": 0, "verdict": "exact"}, {"detail": 1, "verd'
        judge = client(lambda r: reply(cut)).judge
        parsed = judge([("[M] a frame", b"x")], "p", SCHEMA, salvage=True)
        self.assertEqual(parsed["details"], [{"detail": 0, "verdict": "exact"}])
        with self.assertRaisesRegex(LLMError, "invalid JSON"):
            judge([("[M] a frame", b"x")], "p", SCHEMA)

    def test_an_empty_reply_says_the_model_may_not_read_images(self):
        # The failure `deepseek-v4-pro` showed: schema-shaped nothing reads as a judge that
        # works and scores badly, so say what actually happened.
        with self.assertRaisesRegex(LLMError, "may not support image input"):
            client(lambda r: reply("   "), retries=0).judge([("[M] a frame", b"x")], "p", SCHEMA)


class TransportTests(unittest.TestCase):
    def test_a_retryable_status_is_retried_then_succeeds(self):
        replies = iter([httpx.Response(429, json={"error": "slow down"}),
                        reply('{"details": []}')])
        parsed = client(lambda r: next(replies)).judge([("[M] a frame", b"x")], "p", SCHEMA)
        self.assertEqual(parsed, {"details": []})

    def test_a_non_retryable_status_surfaces_the_body(self):
        with self.assertRaisesRegex(LLMError, "HTTP 400"):
            client(lambda r: httpx.Response(400, json={"error": {"message": "bad model"}})) \
                .judge([("[M] a frame", b"x")], "p", SCHEMA)

    def test_an_upstream_failure_reported_as_a_200_is_not_read_as_an_answer(self):
        # OpenRouter answers 200 with an `error` member when the provider behind it failed.
        body = httpx.Response(200, json={"error": {"message": "upstream is down"}})
        with self.assertRaisesRegex(LLMError, "upstream is down"):
            client(lambda r: body, retries=1).judge([("[M] a frame", b"x")], "p", SCHEMA)


class UsageTests(unittest.TestCase):
    def test_tokens_land_on_the_declared_counters(self):
        # `Usage.add` resolves each name with getattr and raises on an undeclared counter, so a
        # backend whose counters were never declared fails here rather than silently dropping.
        usage = Usage()
        client(lambda r: reply('{"details": []}', usage={"prompt_tokens": 1200, "completion_tokens": 40}),
               usage=usage).judge([("[M] a frame", b"x")], "p", SCHEMA)

        self.assertEqual(usage.openrouter_calls, 1)
        self.assertEqual(usage.openrouter_input, 1200)
        self.assertEqual(usage.openrouter_output, 40)
        self.assertIn("openrouter_calls", usage.as_dict())


class ChatJsonTests(unittest.TestCase):
    def test_it_takes_the_arguments_query_details_calls_it_with(self):
        # `evidence.query_details` is written against DeepSeek's client and passes `thinking`;
        # the checker hands it this client instead, so the two have to be interchangeable there.
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return reply('{"details": [{"vi": "một con vật màu vàng"}]}')

        data = client(handler).chat_json(
            [{"role": "user", "content": "split it"}], thinking=False, max_tokens=1500, temperature=0)

        self.assertEqual(data["details"], [{"vi": "một con vật màu vàng"}])
        self.assertEqual(seen["max_tokens"], 1500)
        self.assertEqual(seen["temperature"], 0)
        self.assertNotIn("thinking", seen)
        self.assertEqual(seen["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
