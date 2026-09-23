import json
import unittest
from types import SimpleNamespace

import httpx

from agent.llm import DeepSeekClient, GeminiClient, LLMError, parse_json, repair_truncated_json
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

    def test_image_retries_transient_ffmpeg_failures_but_search_does_not(self):
        replies = iter([httpx.Response(500, json={"detail": "ffmpeg failed"}),
                        httpx.Response(200, content=b"jpeg")])
        self.assertEqual(search_client(lambda request: next(replies)).image("L01_V001", 5), b"jpeg")

        calls = []

        def failing(request):
            calls.append(request.url.path)
            return httpx.Response(500, json={"detail": "boom"})

        with self.assertRaises(SearchError):
            search_client(failing, api="batch").search([])
        self.assertEqual(calls, ["/agent/search"])

    def test_image_width_snaps_to_an_allowed_thumbnail_size(self):
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(200, content=b"jpeg")

        client = search_client(handler)
        self.assertEqual(client.image("L01_V001", 5, 600), b"jpeg")
        client.image("L01_V001", 5, 0)
        self.assertEqual(seen, ["http://server/thumb/L01_V001/5?w=640", "http://server/frame/L01_V001/5"])


def item(video, frame, **extra):
    return {"video_id": video, "frame_idx": frame, "time": frame / 25, "score": 0.5, **extra}


class CompatServer:
    """The public routes of a server that predates /agent/search."""

    def __init__(self):
        self.paths = []

    def __call__(self, request):
        self.paths.append(request.url.path)
        path = request.url.path
        if path == "/agent/search":
            return httpx.Response(404, json={"detail": "Not Found"})
        if path == "/search/next":
            return httpx.Response(200, json={"results": [item("L01_V002", 7), item("L01_V003", 1)],
                                             "next_cursor": None})
        if path == "/search/temporal":
            event = {"frame_idx": 10, "time": 0.4, "score": 0.9, "rank": 0}
            return httpx.Response(200, json={"videos": [{"video_id": "L01_V001", "best_score": 0.9,
                "sequences": [{"rank": 1, "score": 0.5, "events": [event, event]},
                              {"rank": 0, "score": 0.9, "events": [event, event]}]}]})
        if path == "/search/image":
            payload = json.loads(dict(httpx.QueryParams(request.content.decode()))["payload"])
            self.image_payload = payload
            return httpx.Response(200, json={"results": [item("L01_V004", 3)], "next_cursor": None})
        body = json.loads(request.content)
        if body["mode"] == "audio":
            return httpx.Response(200, json={"results": [item(
                "L01_V001", 50, transcript_snippet="cá nhám", transcript_start=1.0,
                transcript_end=9.0, matched_keywords=["cá"])], "next_cursor": None})
        if body["mode"] == "ocr":
            return httpx.Response(200, json={"results": [
                item("L01_V001", 60, ocr_snippet="Mèo NÀ NÍ", ocr_start=2.0, ocr_end=3.0),
                item("L01_V002", 60, ocr_snippet="Na Nhi", ocr_start=2.0, ocr_end=3.0)],
                "next_cursor": None})
        return httpx.Response(200, json={"results": [item("L01_V001", 5), item("L01_V002", 6)],
                                         "next_cursor": "page2"})


class CompatTests(unittest.TestCase):
    def test_falls_back_once_and_maps_every_kind(self):
        server = CompatServer()
        client = search_client(server)
        with self.assertLogs("agent", "WARNING"):
            results = client.search([
                {"kind": "visual", "id": "v", "text": "boat", "limit": 150},
                {"kind": "visual", "id": "scoped", "text": "boat", "video_ids": ["L01_V002"]},
                {"kind": "visual", "id": "short", "text": "boat", "limit": 1},
                {"kind": "audio", "id": "a", "text": "cá nhám", "method": "semantic"},
                {"kind": "ocr", "id": "o", "text": "Nà Ní"},
                {"kind": "temporal", "id": "t", "events": ["a", "b"]},
                {"kind": "similar", "id": "s", "video_id": "L01_V009", "frame_idx": 4},
            ])

        # limit 150 needs a second page of 100; limit 1 stops inside the first
        self.assertEqual([(h["video_id"], h["frame_idx"], h["rank"]) for h in results["v"]["hits"]],
                         [("L01_V001", 5, 1), ("L01_V002", 6, 2), ("L01_V002", 7, 3), ("L01_V003", 1, 4)])
        self.assertEqual(len(results["short"]["hits"]), 1)
        self.assertEqual([h["frame_idx"] for h in results["scoped"]["hits"]], [6, 7])
        self.assertEqual(results["a"]["hits"][0]["text"], "cá nhám")
        self.assertEqual((results["a"]["hits"][0]["start"], results["a"]["hits"][0]["end"]), (1.0, 9.0))
        self.assertEqual([h["exact"] for h in results["o"]["hits"]], [True, False])
        self.assertEqual(results["t"]["hits"][0]["sequences"][0]["score"], 0.9)
        self.assertEqual(server.image_payload["reference"], {"video_id": "L01_V009", "frame_idx": 4})
        self.assertEqual(client.api, "compat")

        server.paths.clear()
        client.search([{"kind": "ocr", "id": "o", "text": "x"}])
        self.assertEqual(server.paths, ["/search"])

    def test_batch_mode_does_not_fall_back(self):
        client = search_client(CompatServer(), api="batch")
        with self.assertRaises(SearchError):
            client.search([{"kind": "visual", "id": "v", "text": "boat"}])


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

    def test_a_caller_can_ask_a_judgement_at_temperature_zero(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return completion('{"ok": true}')

        client = deepseek(handler)
        client.chat_json([{"role": "user", "content": "json"}], thinking=False, temperature=0)
        client.chat_json([{"role": "user", "content": "json"}], thinking=False)
        client.chat_json([{"role": "user", "content": "json"}], thinking=True, temperature=0)
        self.assertEqual([b.get("temperature") for b in bodies], [0, 0.3, None])  # thinking mode takes none

    def test_a_reply_cut_off_by_the_token_limit_keeps_what_is_complete_only_when_asked(self):
        cut = '{"details": [{"detail": 0, "judgements": [{"id": "e1", "why": "a \\"b\\" }", "verdict": "exact"}, {"id": "e2", "wh'
        self.assertEqual(repair_truncated_json(cut),
                         {"details": [{"detail": 0, "judgements": [{"id": "e1", "why": 'a "b" }', "verdict": "exact"}]}]})
        self.assertEqual(repair_truncated_json('```json\n{"a": [1, {"b": 2}, {"c"'), {"a": [1, {"b": 2}]})
        with self.assertRaises(ValueError):
            repair_truncated_json('{"a": "unfinished')   # nothing complete to keep
        client = deepseek(lambda request: completion(cut))
        with self.assertRaises(LLMError):
            client.chat_json([{"role": "user", "content": "json"}])     # a plan must never be a prefix
        self.assertEqual(client.chat_json([{"role": "user", "content": "json"}], salvage=True)["details"][0]["detail"], 0)

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


class CaptionWindowFetchTests(unittest.TestCase):
    def test_every_window_of_a_video_comes_back_from_the_windows_route(self):
        paths = []

        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(200, json={"videoId": "L26_V254", "available": True, "captions": [
                {"shot_idx": 6, "window_idx": 0, "frame_idx": None, "start": 12.0, "end": 19.3, "time": 15.6, "text": "A woman."}]})

        self.assertEqual(search_client(handler).caption_windows("L26_V254")[0]["text"], "A woman.")
        self.assertEqual(paths, ["/captions/L26_V254/windows"])

    def test_a_server_without_the_route_or_the_captions_is_an_empty_list(self):
        self.assertEqual(search_client(lambda request: httpx.Response(404, json={"detail": "nf"})).caption_windows("L26_V254"), [])
        self.assertEqual(search_client(lambda request: httpx.Response(200, json={"oops": 1})).caption_windows("L26_V254"), [])


class CaptionFetchTests(unittest.TestCase):
    def test_captions_come_back_keyed_by_frame_index(self):
        client = search_client(lambda request: httpx.Response(200, json={
            "video_id": "L21_V001", "available": True,
            "captions": {"100": {"caption": "A red boat.", "shot_index": 3}},
        }))
        self.assertEqual(client.captions_for_frames("L21_V001", [100]),
                         {100: {"caption": "A red boat.", "shot_index": 3}})

    def test_a_video_without_captions_returns_empty_rather_than_raising(self):
        # Every 2025 K* video. The route answers available:false so this must not
        # look like the route is missing.
        client = search_client(lambda request: httpx.Response(200, json={
            "video_id": "K01_V001", "available": False, "captions": {}}))
        self.assertEqual(client.captions_for_frames("K01_V001", [10, 20]), {})

    def test_a_server_without_the_route_is_asked_exactly_once_per_run(self):
        paths = []

        def handler(request):
            paths.append(request.url.path)
            return httpx.Response(404, json={"detail": "not found"})

        client = search_client(handler)
        self.assertEqual(client.captions_for_frames("L21_V001", [100]), {})
        self.assertEqual(client.captions_for_frames("L21_V002", [200]), {})
        self.assertEqual(len(paths), 1, "the latch should stop the second request")

    def test_no_frames_means_no_request_at_all(self):
        def handler(request):
            raise AssertionError("a caption request should not be made")

        self.assertEqual(search_client(handler).captions_for_frames("L21_V001", []), {})

    def test_captions_can_be_turned_off_entirely(self):
        # AIC_AGENT_CAPTIONS=0 has to stop the request, not just discard the answer.
        def handler(request):
            raise AssertionError("a disabled caption lookup should not reach the server")

        self.assertEqual(search_client(handler, captions=False).captions_for_frames("L21_V001", [1]), {})

    def test_the_setting_reaches_the_client(self):
        from types import SimpleNamespace

        settings = SimpleNamespace(server_url="http://server", server_user=None,
                                   server_password=None, trusted_user=None,
                                   search_api="batch", captions=False)
        self.assertEqual(SearchClient.from_settings(settings).captions_for_frames("L21_V001", [1]), {})

    def test_a_malformed_frame_key_is_skipped(self):
        client = search_client(lambda request: httpx.Response(200, json={
            "video_id": "L21_V001", "available": True,
            "captions": {"100": {"caption": "ok"}, "not-a-frame": {"caption": "x"}},
        }))
        self.assertEqual(list(client.captions_for_frames("L21_V001", [100])), [100])


class DeepSeekVisionTests(unittest.TestCase):
    """`DeepSeekClient.judge` — DeepSeek used as a vision backend.

    Only some of its models read images. `deepseek-v4-pro` answers a vision call
    with an empty `content`, and the schema-shaped zeros it can return instead look
    like a working judge scoring badly, so the empty reply has to be an error.
    """

    def client(self, content, *, status=200, model="deepseek-flash"):
        def handler(request):
            return httpx.Response(status, json={
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            })

        http = httpx.Client(base_url="https://api.deepseek.com",
                            transport=httpx.MockTransport(handler))
        return DeepSeekClient("k", model, http=http, sleep=lambda _: None)

    def test_a_vision_call_sends_the_image_and_parses_the_reply(self):
        seen = {}

        def handler(request):
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": '{"candidates": [{"id": "C1", "score": 7}]}'}}],
                "usage": {},
            })

        http = httpx.Client(base_url="https://api.deepseek.com",
                            transport=httpx.MockTransport(handler))
        reply = DeepSeekClient("k", "deepseek-flash", http=http, sleep=lambda _: None).judge(
            [("[C1] L21_V001 t=1.0s", b"jpeg")], "prompt",
            {"type": "object", "properties": {"candidates": {}}})
        self.assertEqual(reply["candidates"][0]["score"], 7)
        parts = seen["messages"][0]["content"]
        self.assertEqual(parts[0]["text"], "[C1] L21_V001 t=1.0s")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_an_empty_reply_is_an_error_not_a_zero_score(self):
        with self.assertRaisesRegex(LLMError, "may not support image input"):
            self.client("").judge([("x", b"jpeg")], "p", {})

    def test_a_whitespace_reply_is_also_an_error(self):
        with self.assertRaisesRegex(LLMError, "may not support image input"):
            self.client("   \n  ").judge([("x", b"jpeg")], "p", {})

    def test_invalid_json_surfaces_as_an_llm_error(self):
        with self.assertRaisesRegex(LLMError, "invalid JSON"):
            self.client("not json at all").judge([("x", b"jpeg")], "p", {})

    def test_salvage_keeps_the_complete_part_of_a_reply_the_cap_cut_off(self):
        """Measured on the checklist: a call shown ONE candidate's images answered with a dozen
        invented ones, ran past the token limit and was dropped whole -- the candidate lost every
        check it had just earned. Its first entries were complete, and those are worth keeping.
        """
        cut = ('{"candidates": [{"id": "C1", "details": [{"detail": 0, "verdict": "exact"}]}, '
               '{"id": "C2", "details": [{"detail": 1, "verdict": "unrel')
        reply = self.client(cut).judge([("x", b"jpeg")], "p", {}, salvage=True)
        self.assertEqual(reply, {"candidates": [{"id": "C1", "details": [{"detail": 0, "verdict": "exact"}]}]})
        with self.assertRaisesRegex(LLMError, "invalid JSON"):     # off unless a caller asks
            self.client(cut).judge([("x", b"jpeg")], "p", {})


if __name__ == "__main__":
    unittest.main()
