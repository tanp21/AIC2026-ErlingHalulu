import io
import json
import os
import unittest
from unittest.mock import patch

import httpx

from agent.llm import LLMError
from agent.vlm import (
    CANDIDATES, HttpVLM, LocalVLM, VLMError, align_ids, array_field, build_vision,
    compact_prompt, format_instruction, label_ids, spec,
)
from agent.vision import (
    ANSWER_SCHEMA, PICK_SCHEMA, SEQUENCE_SCHEMA, VERIFY_SCHEMA, MAX_CAPTION_CHARS,
    Inspector, candidate_label,
)
from backend.shots import strip_caption_lead_in


def jpeg(colour="red"):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), colour).save(buffer, "JPEG")
    return buffer.getvalue()


class FakeModel:
    """Stands in for a loaded checkpoint: records prompts, replays scripted replies."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []
        self.calls = 0

    def generate(self, pictures, prompt, max_new_tokens):
        self.calls += 1
        self.prompts.append({"prompt": prompt, "images": len(pictures),
                             "tokens": max_new_tokens})
        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def labelled(*ids):
    return [(f"[{cid}] L21_V001 t={index}.0s", jpeg()) for index, cid in enumerate(ids)]


class FormatInstructionTests(unittest.TestCase):
    def test_it_names_the_keys_required_inside_the_array(self):
        # The prompts in agent.prompts leave the output shape to response_json_schema,
        # so a local model has nothing telling it what to emit unless this says so.
        text = format_instruction(VERIFY_SCHEMA)
        self.assertIn("candidates", text)
        self.assertIn("id, score, reason", text)
        self.assertIn("0-10", text)

    def test_enums_and_nested_objects_are_rendered(self):
        text = format_instruction(PICK_SCHEMA)
        self.assertIn("confidence", text)
        self.assertNotIn("<value>", text)

    def test_a_schema_without_arrays_still_renders(self):
        text = format_instruction(ANSWER_SCHEMA)
        self.assertIn("answer, confidence", text)


class AlignIdsTests(unittest.TestCase):
    def test_it_recovers_ids_the_model_echoed_as_label_bodies(self):
        # Measured on Vintern-1B-v3_5: asked about "[C1]", it answers
        # "L21_V001 t=0.0s". `Inspector.verify` drops any id it does not know, so
        # without this the model scores nothing at all, silently.
        reply = {"candidates": [{"id": "L21_V001 t=0.0s", "score": 7},
                                {"id": "L21_V001 t=1.0s", "score": 3}]}
        cleaned = align_ids(reply, VERIFY_SCHEMA, labelled("C1", "C2"))
        self.assertEqual([row["id"] for row in cleaned["candidates"]], ["C1", "C2"])

    def test_it_accepts_plain_ids_and_one_based_numbers(self):
        snapshots = {"candidates": [{"id": "C1"}, {"id": "2"}, {"id": "[C3]"}]}
        cleaned = align_ids(snapshots, VERIFY_SCHEMA, labelled("C1", "C2", "C3"))
        self.assertEqual([row["id"] for row in cleaned["candidates"]], ["C1", "C2", "C3"])

    def test_duplicates_and_unknowns_are_dropped(self):
        reply = {"candidates": [{"id": "C1", "score": 9}, {"id": "C1", "score": 2},
                                {"id": "nonsense", "score": 5}]}
        cleaned = align_ids(reply, VERIFY_SCHEMA, labelled("C1", "C2"))
        self.assertEqual(cleaned["candidates"], [{"id": "C1", "score": 9}])

    def test_an_ambiguous_match_is_dropped_rather_than_guessed(self):
        # Two labels share a prefix; picking one at random is worse than leaving the
        # candidate unverified, which every caller already handles.
        reply = {"candidates": [{"id": "L21_V001", "score": 8}]}
        labels = [("[C1] L21_V001 t=0.0s", jpeg()), ("[C2] L21_V001 t=1.0s", jpeg())]
        self.assertEqual(align_ids(reply, VERIFY_SCHEMA, labels)["candidates"], [])

    def test_it_leaves_schemas_without_an_id_array_alone(self):
        reply = {"answer": "12", "confidence": 8}
        self.assertEqual(align_ids(reply, ANSWER_SCHEMA, labelled("C1")), reply)

    def test_missing_ids_are_ignored(self):
        snapshots = {"candidates": [{"id": "C1"}]}
        self.assertEqual(align_ids(snapshots, VERIFY_SCHEMA, [])["candidates"], [{"id": "C1"}])

    def test_label_ids_reads_the_bracketed_prefix(self):
        self.assertEqual(label_ids(labelled("C1", "C2")), ["C1", "C2"])
        self.assertEqual(label_ids([("no brackets", b"")]), [])

    def test_array_field_finds_the_scored_list(self):
        self.assertEqual(array_field(VERIFY_SCHEMA)[0], "candidates")
        self.assertEqual(array_field(SEQUENCE_SCHEMA)[0], "events")
        self.assertIsNone(array_field(ANSWER_SCHEMA))


class LocalVLMTests(unittest.TestCase):
    def model(self, *replies):
        return LocalVLM(CANDIDATES[0], backend=FakeModel(*replies))

    def test_per_image_jobs_fan_out_and_reassemble_the_array(self):
        # A small model handed five frames and a five-element schema invents the
        # ones it cannot see; asked one frame at a time it answers.
        reply = json.dumps({"id": "C1", "score": 8})
        model = self.model(reply)
        out = model.judge(labelled("C1", "C2", "C3"), "long prompt", VERIFY_SCHEMA,
                          job="verify", query="con thuyền đỏ")
        self.assertEqual([row["id"] for row in out["candidates"]], ["C1", "C2", "C3"])
        self.assertEqual(model._backend.calls, 3)
        self.assertEqual([p["images"] for p in model._backend.prompts], [1, 1, 1])

    def test_the_local_prompt_is_short_and_carries_the_vietnamese_query(self):
        model = self.model(json.dumps({"id": "C1", "score": 8}))
        model.judge(labelled("C1"), "LONG ENGLISH PROMPT", VERIFY_SCHEMA,
                    job="verify", query="con thuyền đỏ")
        sent = model._backend.prompts[0]["prompt"]
        self.assertIn("con thuyền đỏ", sent)
        self.assertNotIn("LONG ENGLISH PROMPT", sent)

    def test_without_a_query_it_falls_back_to_the_callers_prompt(self):
        model = self.model(json.dumps({"candidates": [{"id": "C1", "score": 8}]}))
        model.judge(labelled("C1"), "CALLER PROMPT", VERIFY_SCHEMA, job="verify")
        self.assertIn("CALLER PROMPT", model._backend.prompts[0]["prompt"])

    def test_multi_image_jobs_stay_in_one_call(self):
        # falsify and pick compare frames against each other; splitting them would
        # change the question.
        model = self.model(json.dumps({"id": "F1", "confidence": 9}))
        model.judge(labelled("F0", "F1"), "prompt", PICK_SCHEMA, job="pick")
        self.assertEqual(model._backend.calls, 1)
        self.assertEqual(model._backend.prompts[0]["images"], 2)

    def test_an_unparsable_reply_gets_exactly_one_repair_attempt(self):
        model = self.model("not json at all", json.dumps({"id": "C1", "score": 5}))
        out = model.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="q")
        self.assertEqual(out["candidates"][0]["score"], 5)
        self.assertEqual(model._backend.calls, 2)

    def test_two_failures_raise_and_the_error_is_an_llm_error(self):
        model = self.model("nope", "still nope")
        with self.assertRaises(VLMError):
            model.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="q")
        self.assertTrue(issubclass(VLMError, LLMError), "callers catch LLMError")

    def test_one_unanswerable_frame_does_not_cost_the_others_their_verdicts(self):
        # A candidate with no verdict is a state the pool already handles; a whole
        # chunk lost to one bad frame is the failure worth avoiding.
        class Flaky(FakeModel):
            def generate(self, pictures, prompt, max_new_tokens):
                self.calls += 1
                if "C2" in prompt:
                    return "not json"  # burns the repair retry too
                return json.dumps({"id": "C1", "score": 9})

        model = LocalVLM(CANDIDATES[0], backend=Flaky())
        out = model.judge(labelled("C1", "C2", "C3"), "p", VERIFY_SCHEMA,
                          job="verify", query="q")
        self.assertEqual([row["id"] for row in out["candidates"]], ["C1", "C3"])

    def test_a_chunk_where_nothing_answered_still_raises(self):
        model = self.model("nope", "still nope")
        with self.assertRaises(VLMError):
            model.judge(labelled("C1", "C2"), "p", VERIFY_SCHEMA, job="verify", query="q")

    def test_the_id_is_stamped_from_the_label_not_read_from_the_model(self):
        # Measured on Vintern: asked about "[C2]" it answers "L21_V001 t=1.0s".
        model = self.model(json.dumps({"id": "L21_V001 t=9.9s", "score": 4}))
        out = model.judge(labelled("C1", "C2"), "p", VERIFY_SCHEMA, job="verify", query="q")
        self.assertEqual([row["id"] for row in out["candidates"]], ["C1", "C2"])

    def test_a_second_out_of_memory_error_disables_the_backend(self):
        import torch

        class Oom(FakeModel):
            def generate(self, pictures, prompt, max_new_tokens):
                self.calls += 1
                raise torch.cuda.OutOfMemoryError("out of memory")

        model = LocalVLM(CANDIDATES[0], backend=Oom())
        with self.assertRaises(VLMError):
            model.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="q")
        self.assertTrue(model._disabled)
        with self.assertRaisesRegex(VLMError, "disabled"):
            model.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="q")

    def test_an_empty_image_list_is_refused(self):
        with self.assertRaises(VLMError):
            self.model("{}").judge([], "p", VERIFY_SCHEMA, job="verify")

    def test_images_are_resized_before_they_reach_the_model(self):
        seen = {}

        class Recorder(FakeModel):
            def generate(self, pictures, prompt, max_new_tokens):
                seen["sizes"] = [image.size for _, image in pictures]
                return json.dumps({"id": "C1", "score": 1})

        LocalVLM(CANDIDATES[0], backend=Recorder(), image_side=32).judge(
            labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="q")
        self.assertEqual(max(seen["sizes"][0]), 32)


class HttpVLMTests(unittest.TestCase):
    def client(self, handler, **kwargs):
        http = httpx.Client(base_url="http://vlm", transport=httpx.MockTransport(handler))
        return HttpVLM("http://vlm", http=http, sleep=lambda _: None, **kwargs)

    def test_a_round_trip_returns_the_reply_and_sends_the_images(self):
        seen = {}

        def handler(request):
            payload = json.loads(request.content)
            seen.update(payload)
            return httpx.Response(200, json={"reply": {"candidates": [{"id": "C1", "score": 7}]}})

        reply = self.client(handler).judge(labelled("C1"), "prompt", VERIFY_SCHEMA,
                                           job="verify", query="con thuyền đỏ")
        self.assertEqual(reply["candidates"][0]["score"], 7)
        self.assertEqual(seen["job"], "verify")
        self.assertEqual(seen["query"], "con thuyền đỏ",
                         "the remote backend needs the query to build its own prompt")
        self.assertEqual(seen["images"][0]["label"].split()[0], "[C1]")
        self.assertTrue(seen["images"][0]["b64"])

    def test_a_transient_failure_is_retried(self):
        replies = iter([httpx.Response(503, json={}),
                        httpx.Response(200, json={"reply": {"ok": True}})])
        reply = self.client(lambda request: next(replies)).judge(
            labelled("C1"), "p", VERIFY_SCHEMA, job="verify")
        self.assertEqual(reply, {"ok": True})

    def test_a_client_error_surfaces_as_a_vlm_error(self):
        client = self.client(lambda request: httpx.Response(422, json={"detail": "bad schema"}))
        with self.assertRaisesRegex(VLMError, "bad schema"):
            client.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify")

    def test_an_oversized_body_is_refused_rather_than_posted(self):
        # The `answer` job carries a full-resolution frame; pushing megabytes
        # through the ingress is worse than returning an unverified job.
        client = self.client(lambda request: httpx.Response(200, json={"reply": {}}),
                             max_body_mb=0)
        with self.assertRaisesRegex(VLMError, "too large"):
            client.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="answer")


class BuildVisionTests(unittest.TestCase):
    def settings(self, **kwargs):
        from types import SimpleNamespace

        base = dict(vision="gemini", gemini_api_key="k", gemini_model="gemini-3.8-flash",
                    gemini_thinking_level="low", vlm_url=None, vlm_model=None,
                    vlm_device=None, vlm_image_side=448, vlm_answer_side=1280,
                    vlm_max_new_tokens=128, vlm_max_tiles=6,
                    deepseek_api_key="k", deepseek_base_url="https://api.deepseek.com")
        return SimpleNamespace(**{**base, **kwargs})

    def test_gemini_is_the_default_and_needs_a_key(self):
        from agent.llm import GeminiClient

        self.assertIsInstance(build_vision(self.settings()), GeminiClient)
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
            build_vision(self.settings(gemini_api_key=""))

    def test_local_builds_the_in_process_backend_and_carries_the_knobs(self):
        vision = build_vision(self.settings(vision="local", vlm_model="Qwen/Qwen3-VL-2B-Instruct",
                                            vlm_image_side=320))
        self.assertIsInstance(vision, LocalVLM)
        self.assertEqual(vision.spec.name, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(vision.image_side, 320)

    def test_a_vlm_url_switches_to_the_http_backend(self):
        vision = build_vision(self.settings(vision="local", vlm_url="http://other:8000"))
        self.assertIsInstance(vision, HttpVLM)

    def test_deepseek_is_a_remote_vision_backend_like_gemini(self):
        from agent.llm import DeepSeekClient

        settings = self.settings(vision="deepseek", deepseek_api_key="k",
                                 deepseek_base_url="https://api.deepseek.com")
        vision = build_vision(settings)
        self.assertIsInstance(vision, DeepSeekClient)
        # Judging is a bounded 0-10 score; thinking would spend the answer budget on
        # reasoning_content and return an empty content.
        self.assertFalse(vision.thinking)

    def test_an_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be 'local', 'gemini' or 'deepseek'"):
            build_vision(self.settings(vision="magic"))


class CompactPromptTests(unittest.TestCase):
    def test_it_carries_the_query_and_the_candidate_id(self):
        text = compact_prompt("verify", "con thuyền đỏ", "C3")
        self.assertIn("con thuyền đỏ", text)
        self.assertIn('"C3"', text)

    def test_it_asks_for_no_free_text_reason(self):
        # `reason` is only logged -- `CandidatePool.ordered` ranks on score and scene
        # -- and it is where a small model spends its whole token budget and gets
        # truncated mid-JSON.
        self.assertNotIn("reason", compact_prompt("verify", "q", "C1"))
        self.assertNotIn("reason", compact_prompt("sequence", "q", "C1"))

    def test_jobs_that_reason_across_frames_get_no_compact_prompt(self):
        for job in ("pick", "falsify", "answer"):
            self.assertIsNone(compact_prompt(job, "q", "C1"))

    def test_a_missing_query_falls_back_to_the_callers_prompt(self):
        self.assertIsNone(compact_prompt("verify", "", "C1"))


class ModelRegistryTests(unittest.TestCase):
    def test_every_candidate_has_a_known_family_and_a_unique_slug(self):
        self.assertEqual(len({c.slug for c in CANDIDATES}), len(CANDIDATES))
        for candidate in CANDIDATES:
            self.assertIn(candidate.family, ("internvl", "qwen3vl", "qwen2_5vl"))
            self.assertTrue(candidate.snapshot_dir.name.startswith(candidate.name.split("/")[0]))

    def test_lookup_accepts_the_hub_id_and_the_slug(self):
        first = CANDIDATES[0]
        self.assertIs(spec(first.name), first)
        self.assertIs(spec(first.slug), first)
        with self.assertRaises(KeyError):
            spec("nope/nope")


class StripLeadInsTests(unittest.TestCase):
    def test_the_observed_preamble_is_removed(self):
        self.assertEqual(
            strip_caption_lead_in("Based on the provided video frames, here is the detailed "
                                  "description in chronological order: A red boat."),
            "A red boat.")

    def test_a_caption_without_a_preamble_is_untouched(self):
        text = "A red boat drifts downstream."
        self.assertEqual(strip_caption_lead_in(text), text)

    def test_an_empty_string_survives(self):
        self.assertEqual(strip_caption_lead_in(""), "")


class CandidateCaptionTests(unittest.TestCase):
    """The shot caption the agent reads into a candidate's label."""

    def candidate(self):
        from agent.pool import Candidate, Evidence

        return Candidate(
            cid="C1", video_id="L21_V001", frame_idx=100, time=3.5,
            evidence={"audio": Evidence("a", "audio", 1, 0.5, text="xin chào"),
                      "ocr": Evidence("o", "ocr", 1, 0.5, text="Nà Ní")},
            fused=0.4,
        )

    def test_without_a_caption_the_label_is_unchanged(self):
        # Every caller that has no caption -- and every 2025 video -- must keep the
        # label it had before this existed.
        label = candidate_label(self.candidate())
        self.assertNotIn("caption", label)
        self.assertIn("speech nearby", label)
        self.assertIn("on-screen text", label)

    def test_the_caption_is_quoted_and_marked_untrusted(self):
        # It comes from another model looking at the same frames, so it can be
        # confidently wrong; the judge has to weight what it sees over what it reads.
        label = candidate_label(self.candidate(), "A red boat drifts downstream.")
        self.assertIn("A red boat drifts downstream.", label)
        self.assertIn("may be wrong", label)
        self.assertIn('"', label)

    def test_a_long_caption_is_truncated(self):
        label = candidate_label(self.candidate(), "x" * (MAX_CAPTION_CHARS * 3))
        self.assertLessEqual(len(label), MAX_CAPTION_CHARS + 200)

    def test_an_empty_caption_is_left_out_entirely(self):
        self.assertNotIn("caption", candidate_label(self.candidate(), ""))


class InspectorCaptionTests(unittest.TestCase):
    def inspector(self, search):
        return Inspector(object(), search, chunk_size=2)

    def test_frames_are_grouped_into_one_request_per_video(self):
        calls = []

        class Search:
            def captions_for_frames(self, video_id, frames):
                calls.append((video_id, sorted(frames)))
                return {frame: {"caption": f"shot at {frame}"} for frame in frames}

        found = self.inspector(Search()).captions(
            [("L21_V001", 10), ("L21_V001", 20), ("L21_V002", 30)])
        self.assertEqual(sorted(calls, key=lambda c: c[0]),
                         [("L21_V001", [10, 20]), ("L21_V002", [30])])
        self.assertEqual(found[("L21_V001", 20)], "shot at 20")

    def test_a_second_look_at_the_same_frame_does_not_hit_the_server(self):
        calls = []

        class Search:
            def captions_for_frames(self, video_id, frames):
                calls.append(video_id)
                return {frame: {"caption": "once"} for frame in frames}

        inspector = self.inspector(Search())
        inspector.captions([("L21_V001", 10)])
        inspector.captions([("L21_V001", 10)])
        self.assertEqual(len(calls), 1)

    def test_a_server_without_the_route_yields_no_captions(self):
        class Search:
            pass

        self.assertEqual(self.inspector(Search()).captions([("L21_V001", 10)]), {})

    def test_a_search_failure_does_not_cost_the_round(self):
        class Search:
            def captions_for_frames(self, video_id, frames):
                raise RuntimeError("boom")

        self.assertEqual(self.inspector(Search()).captions([("L21_V001", 10)]), {})

    def test_a_caption_that_is_empty_is_not_recorded(self):
        class Search:
            def captions_for_frames(self, video_id, frames):
                return {frame: {"caption": ""} for frame in frames}

        self.assertEqual(self.inspector(Search()).captions([("L21_V001", 10)]), {})


class BackendContractTests(unittest.TestCase):
    """`Inspector._judge` calls whichever backend is configured the same way.

    A backend that silently drops a keyword raises only in the mode nobody
    exercised — which is how `HttpVLM` shipped without `query` and would have
    failed every remote judgement while the in-process tests stayed green.
    """

    def test_every_backend_accepts_what_the_inspector_sends(self):
        import inspect

        from agent.llm import DeepSeekClient, GeminiClient
        from agent.vlm import HttpVLM, LocalVLM

        expected = ("images", "prompt", "schema", "max_output_tokens", "job",
                    "query", "scenes", "question")
        for backend in (GeminiClient.judge, DeepSeekClient.judge,
                        LocalVLM.judge, HttpVLM.judge):
            params = inspect.signature(backend).parameters
            for name in expected:
                self.assertIn(name, params,
                              f"{backend.__qualname__} is missing {name!r}")

    def test_the_usage_counters_a_backend_writes_are_declared(self):
        # `Usage.add` resolves every name with getattr, so an undeclared counter
        # raises on the first vision call — in production only, since every test and
        # the benchmark pass usage=None.
        from agent.llm import Usage

        usage = Usage()
        usage.add(vlm_calls=1)
        self.assertEqual(usage.as_dict()["vlm_calls"], 1)
        self.assertIn("vlm_calls", usage.as_dict())

    def test_a_local_call_reports_its_usage(self):
        from agent.llm import Usage
        from agent.vlm import LocalVLM

        usage = Usage()
        model = LocalVLM(CANDIDATES[0], backend=FakeModel(json.dumps({"id": "C1", "score": 4})),
                         usage=usage)
        model.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="q")
        self.assertEqual(usage.as_dict()["vlm_calls"], 1)

    def test_the_inspector_hands_the_vietnamese_query_to_the_backend(self):
        from agent.vision import Inspector

        seen = {}

        class Backend:
            def judge(self, images, prompt, schema, **kwargs):
                seen.update(kwargs)
                return {"candidates": [{"id": "C1", "score": 5, "reason": "ok"}]}

        class Search:
            def image(self, video_id, frame_idx, width):
                return jpeg()

            def captions_for_frames(self, video_id, frames):
                return {}

        inspector = Inspector(Backend(), Search(), chunk_size=1)
        from agent.pool import Candidate

        inspector.verify(_task(), _plan(), [Candidate(cid="C1", video_id="L21_V001",
                                                      frame_idx=1, time=1.0)])
        self.assertEqual(seen["job"], "verify")
        self.assertEqual(seen["query"], "con thuyền đỏ")


def _task():
    from agent.tasks import task_from_text

    return task_from_text("q", "con thuyền đỏ", "kis")


def _plan():
    from agent.plan import Plan

    return Plan(summary_vi="một con thuyền", scenes=[])


class PromptLanguageTests(unittest.TestCase):
    """Each family is prompted in the language it follows best."""

    def local(self, slug, **kwargs):
        from agent.vlm import LocalVLM

        return LocalVLM(spec(slug), backend=FakeModel("{}"), **kwargs)

    def test_the_vietnamese_tuned_model_gets_vietnamese(self):
        self.assertEqual(self.local("5CD-AI/Vintern-1B-v3_5").prompt_language, "vi")

    def test_the_qwen_models_get_english(self):
        for slug in ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct"):
            self.assertEqual(self.local(slug).prompt_language, "en", slug)

    def test_the_environment_overrides_it_for_an_ablation(self):
        # The point of the knob is to be able to score both sides rather than assume.
        with patch.dict(os.environ, {"AIC_VLM_PROMPT": "vi"}):
            self.assertEqual(self.local("Qwen/Qwen3-VL-2B-Instruct").prompt_language, "vi")

    def test_an_explicit_argument_beats_the_environment(self):
        with patch.dict(os.environ, {"AIC_VLM_PROMPT": "vi"}):
            self.assertEqual(
                self.local("Qwen/Qwen3-VL-2B-Instruct", prompt_language="en").prompt_language,
                "en")

    def test_the_language_reaches_the_prompt_it_sends(self):
        model = LocalVLM(spec("Qwen/Qwen3-VL-2B-Instruct"),
                         backend=FakeModel(json.dumps({"id": "C1", "score": 5})))
        model.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="con thuyền đỏ")
        sent = model._backend.prompts[0]["prompt"]
        self.assertIn("Rate this frame", sent)
        self.assertIn("con thuyền đỏ", sent, "the query itself stays as the user wrote it")

    def test_vietnamese_is_still_what_internvl_sends(self):
        model = LocalVLM(spec("5CD-AI/Vintern-1B-v3_5"),
                         backend=FakeModel(json.dumps({"id": "C1", "score": 5})))
        model.judge(labelled("C1"), "p", VERIFY_SCHEMA, job="verify", query="con thuyền đỏ")
        self.assertIn("Đánh giá", model._backend.prompts[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
