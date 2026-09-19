import unittest

from agent.llm import GroundedAnswer
from agent.orchestrator import AgentOptions, reconcile_answer
from agent.plan import initial_queries, parse_plan
from agent.pool import CandidatePool
from agent.prompts import REWRITE_SYSTEM
from agent.tasks import task_from_text
from agent.web import is_cooking, tagged
from tests.agent.test_orchestrator import (
    HITS, PLAN, FakeDeepSeek, FakeGemini, FakeSearch, make_agent, tool_call,
)


class RewritingDeepSeek(FakeDeepSeek):
    """Returns the plan, then `rewrite` for the one rewrite call."""

    def __init__(self, plan=PLAN, rewrite=None):
        super().__init__(plan)
        self.rewrite = rewrite or {"queries": []}
        self.rewrite_prompts = []

    def chat_json(self, messages, **kwargs):
        if messages[0]["content"] == REWRITE_SYSTEM:
            self.rewrite_prompts.append(messages[1]["content"])
            return self.rewrite
        return super().chat_json(messages, **kwargs)


class WebGemini(FakeGemini):
    def __init__(self, *args, web_text="cá mập", **kwargs):
        super().__init__(*args, **kwargs)
        self.web_text = web_text
        self.web_prompts = []

    def web_search(self, question):
        self.web_prompts.append(question)
        return GroundedAnswer(self.web_text, (), ())


class FastModeTests(unittest.TestCase):
    def test_options_for_fast_mode(self):
        options = AgentOptions.for_mode("fast", time_budget=30)
        self.assertEqual((options.mode, options.time_budget, options.verify_chunk),
                         ("fast", 30, 5))
        self.assertEqual(AgentOptions.for_mode("deep").time_budget, 300.0)

    def test_confident_round_one_stops_without_deepseek_tools(self):
        deepseek = RewritingDeepSeek()
        result = make_agent(deepseek, FakeGemini({("L01_V002", 200): 9}), FakeSearch(HITS),
                            mode="fast").run(task_from_text("q-kis", "thuyền đỏ"))

        self.assertIsNone(result.error)
        self.assertEqual([entry["tool"] for entry in result.trace], ["plan", "search", "verify", "falsify"])
        self.assertEqual((deepseek.chats, deepseek.rewrite_prompts), ([], []))
        self.assertEqual(result.ranking[0].video_id, "L01_V002")

    def test_weak_round_one_gets_one_rewrite_and_a_scoped_search(self):
        deepseek = RewritingDeepSeek(rewrite={"queries": [
            {"kind": "audio", "text": "chiếc thuyền màu đỏ"}, {"kind": "nonsense"}]})
        search = FakeSearch(HITS)
        result = make_agent(deepseek, FakeGemini({("L01_V003", 300): 6}), search,
                            mode="fast").run(task_from_text("q-kis", "thuyền đỏ"))

        self.assertEqual(len(deepseek.rewrite_prompts), 1)
        self.assertIn("verdict=6", deepseek.rewrite_prompts[0])  # it sees round 1's verdicts
        self.assertEqual(deepseek.chats, [])  # no tool loop
        rewrite = search.batches[1]
        self.assertEqual([q["kind"] for q in rewrite], ["audio", "visual"])
        self.assertEqual(rewrite[1]["video_ids"][0], "L01_V003")  # the favoured video first
        self.assertIn("rewrite", [entry["tool"] for entry in result.trace])

    def test_cooking_qa_reads_the_video_and_the_recipe_side_by_side(self):
        task = task_from_text("q-qa", "Đầu bếp nhồi sả vào 4 con cá. Đây là loài cá gì?")
        plan = {**PLAN, "question_vi": "Loài cá gì?", "answer_source": "speech"}
        gemini = WebGemini({("L01_V001", 100): 9}, reading={100: ("cá", 3), 90: ("", 0), 110: ("", 0)},
                           web_text="Món cá sòng nướng.\nTỪ KHÓA: cá sòng nướng; sả\nĐÁP ÁN: Cá sòng")
        result = make_agent(RewritingDeepSeek(plan), gemini, FakeSearch(HITS), mode="fast",
                            answer_moments=1).run(task)

        self.assertEqual(result.answer, "Cá sòng")  # unsure reading -> the recipe decides
        self.assertIn("Món ngon mỗi ngày", gemini.web_prompts[0])  # searched before planning
        self.assertIn("xin chào", gemini.web_prompts[-1])  # the recipe search reads the opening
        self.assertIn("video đọc được: cá", result.answer_evidence)

    def test_a_falsified_confident_match_gets_the_rewrite_round(self):
        deepseek = RewritingDeepSeek()
        gemini = FakeGemini({("L01_V002", 200): 10}, reject={"L01_V002"})
        result = make_agent(deepseek, gemini, FakeSearch(HITS), mode="fast").run(
            task_from_text("q-kis", "thuyền đỏ"))
        self.assertEqual(len(deepseek.rewrite_prompts), 1)
        self.assertIn("rewrite", [entry["tool"] for entry in result.trace])

    def test_auto_mode_escalates_an_unsure_query_to_the_tool_loop(self):
        deepseek = RewritingDeepSeek()
        deepseek.turns = [tool_call("finish", ranking=["C3"])]
        result = make_agent(deepseek, FakeGemini({("L01_V003", 300): 5}), FakeSearch(HITS),
                            mode="auto").run(task_from_text("q-kis", "thuyền đỏ"))
        tools = [entry["tool"] for entry in result.trace]
        self.assertLess(tools.index("rewrite"), tools.index("escalate"))
        self.assertEqual(len(deepseek.chats), 1)  # the loop's one turn called finish
        self.assertEqual(result.ranking[0].cid, "C3")

        deepseek = RewritingDeepSeek()
        make_agent(deepseek, FakeGemini({("L01_V003", 300): 8}), FakeSearch(HITS), mode="auto").run(
            task_from_text("q-kis", "thuyền đỏ"))
        self.assertEqual(deepseek.chats, [])  # sure enough: no escalation

    def test_non_cooking_queries_get_no_automatic_web_search(self):
        gemini = WebGemini({("L01_V002", 200): 9})
        make_agent(RewritingDeepSeek(), gemini, FakeSearch(HITS), mode="fast").run(
            task_from_text("q-kis", "thuyền đỏ trên sông"))
        self.assertEqual(gemini.web_prompts, [])

    def test_trake_skips_point_verification_and_keeps_the_unrefined_path(self):
        task = task_from_text("q-trake", "E1: a\nE2: b")
        plan = {**PLAN, "events_en": ["start", "end"]}
        temporal = [{"rank": 1, "video_id": "L01_V005", "score": 0.8, "sequences": [
            {"score": 0.8, "events": [{"frame_idx": 100, "time": 10.0, "score": 0.5},
                                      {"frame_idx": 400, "time": 40.0, "score": 0.5}]}]}]
        search = FakeSearch(HITS, temporal)
        result = make_agent(RewritingDeepSeek(plan), FakeGemini(pick="F3"), search,
                            mode="fast").run(task)

        tools = [entry["tool"] for entry in result.trace]
        self.assertNotIn("verify", tools)
        self.assertEqual({q["max_gap_seconds"] for q in search.batches[0] if q["kind"] == "temporal"},
                         {45, 300})
        self.assertEqual([s.frames for s in result.sequences], [[110, 410], [100, 400]])
        self.assertEqual([s.refined for s in result.sequences], [True, False])


class HedgeTests(unittest.TestCase):
    def test_a_slow_call_is_raced_by_a_duplicate(self):
        import threading
        from agent.llm import GeminiClient

        release = threading.Event()
        calls = []

        class Models:
            def generate_content(self, model, contents, config):
                calls.append(1)
                if len(calls) == 1:
                    release.wait(5)  # the first request hangs
                return type("R", (), {"text": '{"ok": true}', "usage_metadata": None})()

        client = type("C", (), {"models": Models()})()
        gemini = GeminiClient("k", client=client, hedge_after=0.05)
        self.assertEqual(gemini.judge([], "p", {}), {"ok": True})
        self.assertEqual(len(calls), 2)
        release.set()


class ReconcileTests(unittest.TestCase):
    def test_agreement_and_confidence_decide(self):
        self.assertEqual(reconcile_answer("300g", "đọc", 8, "300 g", "recipe")[0], "300g")
        self.assertEqual(reconcile_answer("5", "đọc", 8, "7", "web")[0], "5")
        self.assertEqual(reconcile_answer("5", "đọc", 4, "7", "web")[0], "7")
        self.assertEqual(reconcile_answer("", "no reading", 0, "Cá sòng", "recipe")[0], "Cá sòng")
        self.assertEqual(reconcile_answer("5", "đọc", 3, None, None), ("5", "đọc"))

    def test_the_web_fixes_the_spelling_of_a_spoken_name(self):
        answer, evidence = reconcile_answer("Đèo Bứa", "nghe", 9, "Đèo Tà Pứa", "web", spoken=True)
        self.assertEqual(answer, "Đèo Tà Pứa")
        self.assertIn("Đèo Bứa", evidence)
        # a different answer altogether is not a spelling fix
        self.assertEqual(reconcile_answer("Lý Thường Kiệt", "nghe", 9, "Nguyễn Trãi", "web", spoken=True)[0],
                         "Lý Thường Kiệt")


class WebHelperTests(unittest.TestCase):
    def test_tagged_lines(self):
        text = "Món: ...\n**TỪ KHÓA:** cá sòng; sả\nĐÁP ÁN: Cá sòng"
        self.assertEqual(tagged(text, "ĐÁP ÁN"), "Cá sòng")
        self.assertEqual(tagged(text, "tu khoa"), "cá sòng; sả")
        self.assertIsNone(tagged("ĐÁP ÁN: không rõ", "ĐÁP ÁN"))
        self.assertIsNone(tagged("no tags", "ĐÁP ÁN"))

    def test_cooking_detection(self):
        self.assertTrue(is_cooking("Người đầu bếp đảo đều hỗn hợp trong chảo"))
        self.assertFalse(is_cooking("Hai con rồng vàng đang xoay vòng"))


class VerificationQuotaTests(unittest.TestCase):
    def test_each_query_gets_its_best_hit_verified(self):
        pool = CandidatePool()
        # a spoken passage spanning moments 1-3 lifts them over the caption's lone rank-1 hit
        pool.add_results({
            "cap": {"kind": "visual", "hits": [
                {"rank": 1, "video_id": "L01_V009", "frame_idx": 90, "time": 9.0}]},
            "other": {"kind": "visual", "hits": [
                {"rank": r, "video_id": "L01_V001", "frame_idx": r * 100, "time": r * 10.0}
                for r in (1, 2, 3)]},
            "speech": {"kind": "audio", "hits": [
                {"rank": 1, "video_id": "L01_V001", "frame_idx": 100, "time": 10.0,
                 "start": 5.0, "end": 35.0, "text": "x"}]},
        })
        self.assertNotEqual(pool.unverified(2)[0].video_id, "L01_V009")
        self.assertIn("L01_V009", [c.video_id for c in pool.to_verify(2)])


class SceneCoherenceTests(unittest.TestCase):
    def test_a_video_matching_several_scenes_outranks_one_perfect_shot(self):
        from agent.pool import Verdict

        pool = CandidatePool()
        pool.add_results({"v": {"kind": "visual", "hits": [
            {"rank": 1, "video_id": "L01_V001", "frame_idx": 10, "time": 1.0},
            {"rank": 2, "video_id": "L01_V002", "frame_idx": 10, "time": 1.0},
            {"rank": 3, "video_id": "L01_V002", "frame_idx": 900, "time": 90.0},
            {"rank": 4, "video_id": "L01_V002", "frame_idx": 1800, "time": 180.0},
        ]}})
        by_frame = {(c.video_id, c.frame_idx): c for c in pool.candidates.values()}
        one, gold_a, gold_b = (by_frame[("L01_V001", 10)], by_frame[("L01_V002", 10)],
                               by_frame[("L01_V002", 900)])
        one.verdict = Verdict(10, "", scene=2)
        gold_a.verdict = Verdict(9, "", scene=0)
        gold_b.verdict = Verdict(8, "", scene=1)
        by_frame[("L01_V002", 1800)].verdict = Verdict(7, "", scene=2)
        self.assertEqual(pool.ordered()[0].video_id, "L01_V002")


class LocalPathTests(unittest.TestCase):
    def test_events_in_order_and_close_together_beat_isolated_high_scores(self):
        from agent.pool import local_path

        def hit(time, score):
            return {"time": time, "frame_idx": int(time * 25), "score": score}

        events = [
            [hit(10.0, 0.30), hit(400.0, 0.31)],  # the stronger E1 is far from the others
            [hit(20.0, 0.30), hit(5.0, 0.35)],    # the stronger E2 is before every E1
            [hit(30.0, 0.30)],
        ]
        value, path = local_path(events, max_gap=45)
        self.assertEqual([h["time"] for h in path], [10.0, 20.0, 30.0])
        self.assertIsNone(local_path([[hit(10.0, 0.3)], [hit(100.0, 0.3)]], max_gap=45))
        self.assertIsNone(local_path([[hit(10.0, 0.3)], []], max_gap=45))


class TrakePathRankingTests(unittest.TestCase):
    def test_other_channels_lift_a_lower_temporal_path(self):
        def path(rank, video):
            return {"rank": rank, "video_id": video, "score": 1.0, "sequences": [
                {"score": 1.0, "events": [{"frame_idx": 10, "time": 1.0}, {"frame_idx": 20, "time": 2.0}]}]}

        pool = CandidatePool()
        pool.add_results({
            "t": {"kind": "temporal", "hits": [path(1, "L01_V001"), path(2, "L01_V002")]},
            "tw": {"kind": "temporal", "hits": [path(1, "L01_V002")]},
            "v": {"kind": "visual", "hits": [
                {"rank": 1, "video_id": "L01_V002", "frame_idx": 900, "time": 90.0}]},
        })
        self.assertEqual([s.video_id for s in pool.ranked_sequences(2)], ["L01_V002", "L01_V001"])

    def test_plan_builds_a_tight_and_a_loose_temporal_query(self):
        task = task_from_text("q-trake", "E1: a\nE2: b")
        plan = parse_plan({**PLAN, "events_en": ["a", "b"]}, task)
        temporal = [q for q in initial_queries(plan, task) if q["kind"] == "temporal"]
        self.assertEqual([(q["id"], q["max_gap_seconds"]) for q in temporal], [("r1.t", 45), ("r1.tw", 300)])
        self.assertEqual({q["audio_text"] for q in temporal}, {"thuyền đỏ"})


if __name__ == "__main__":
    unittest.main()
