"""The fully automatic agent: Qwen judges moments, Gemini researches, Q&A waits for the scores."""
import threading
import time
import unittest

from agent import evidence
from agent.config import AgentSettings, default_vision, openrouter_key
from agent.evidence import QueryDetails
from agent.judge import MomentJudge, judge_name
from agent.orchestrator import MOMENT_DEFAULTS, AgentOptions
from agent.plan import parse_plan
from agent.pool import Verdict
from agent.tasks import task_from_text
from agent.web import detect_topic, keywords
from tests.agent.test_fast_mode import RewritingDeepSeek, WebGemini
from tests.agent.test_orchestrator import HITS, PLAN, FakeSearch, make_agent


class FakeQwen:
    """A checker-shaped judge: every moment gets `score`, and a reading when there is a question."""

    model = "qwen/qwen3.7-flash"

    def __init__(self, score=8, answer="5"):
        self.score, self.answer = score, answer
        self.prompts, self.images = [], []

    def judge(self, images, prompt, schema, **_):
        self.prompts.append(prompt)
        self.images.append([label for label, _ in images])
        return {"details": [{"detail": 0, "why": "a red boat is visible", "verdict": "exact", "note_vi": "thấy thuyền đỏ"}],
                "overall": {"score": self.score, "reason": "khớp"},
                "answer": {"text": self.answer, "confidence": 8, "source": "frame"}}

    def chat_json(self, messages, **_):
        return {}


DETAILS = QueryDetails([{"vi": "thuyền màu đỏ", "terms_en": ["red boat"], "terms_vi": ["thuyền đỏ"],
                         "meaning_en": "a red boat", "same_en": [], "different_en": [],
                         "distinctiveness": 0.8}])


class ScriptedJudge:
    """Stands in for MomentJudge inside a run: scores by (video, frame), answers "5"."""

    name = "qwen"
    window = 60.0

    def __init__(self, scores, answers=None):
        self.scores, self.answers = scores, answers or {}
        self.calls, self.details = [], []

    def verify(self, task, plan, candidates, question=None, details=None):
        self.calls.append([c.cid for c in candidates])
        self.details.append(details)
        verdicts = {}
        for c in candidates:
            score = self.scores.get((c.video_id, c.frame_idx), 2)
            answer = self.answers.get((c.video_id, c.frame_idx))
            verdicts[c.cid] = Verdict(score, "khớp" if score >= 6 else "không khớp", answer=answer, by="qwen",
                                      answer_confidence=8.0 if answer else None)
        return verdicts, []


class MomentJudgeTests(unittest.TestCase):
    def test_one_call_per_candidate_over_the_moment_with_the_query_and_question(self):
        qwen = FakeQwen()
        judge = MomentJudge(qwen, FakeSearch(HITS), workers=2)
        task = task_from_text("q-qa", "Chiếc thuyền màu đỏ chở mấy người?", "qa")
        plan = parse_plan({**PLAN, "question_vi": "Thuyền chở mấy người?"}, task)
        from agent.pool import CandidatePool
        pool = CandidatePool()
        candidates = [pool.add_frame("L01_V001", 100, 10.0), pool.add_frame("L01_V002", 200, 20.0)]

        verdicts, errors = judge.verify(task, plan, candidates, details=DETAILS)

        self.assertEqual(errors, [])
        self.assertEqual(len(qwen.prompts), 2)  # one call per candidate
        self.assertIn("Chiếc thuyền màu đỏ chở mấy người?", qwen.prompts[0])  # the organisers' query
        self.assertIn("Thuyền chở mấy người?", qwen.prompts[0])               # the question
        self.assertIn("a red boat", qwen.prompts[0])                          # the frame to find
        self.assertGreater(len(qwen.images[0]), 1)  # its keyframe AND the ones around it
        verdict = verdicts[candidates[0].cid]
        self.assertEqual((verdict.score, verdict.reason, verdict.answer, verdict.by), (8.0, "khớp", "5", "qwen"))
        self.assertEqual(verdict.answer_confidence, 8.0)
        self.assertEqual(verdict.checks[0]["text"], "thuyền màu đỏ")
        self.assertEqual(verdict.checks[0]["status"], "supported")

    def test_a_kis_query_asks_for_no_answer(self):
        qwen = FakeQwen()
        task = task_from_text("q-kis", "thuyền đỏ", "kis")
        from agent.pool import CandidatePool
        candidate = CandidatePool().add_frame("L01_V001", 100, 10.0)
        verdicts, _ = MomentJudge(qwen, FakeSearch(HITS)).verify(task, parse_plan(PLAN, task), [candidate])
        self.assertIsNone(verdicts[candidate.cid].answer)
        self.assertNotIn("The question:", qwen.prompts[0])

    def test_a_failed_call_leaves_that_candidate_unjudged(self):
        class Broken(FakeQwen):
            def judge(self, *args, **kwargs):
                raise RuntimeError("provider down")

        task = task_from_text("q-kis", "thuyền đỏ", "kis")
        from agent.pool import CandidatePool
        candidate = CandidatePool().add_frame("L01_V001", 100, 10.0)
        verdicts, errors = MomentJudge(Broken(), FakeSearch(HITS)).verify(task, parse_plan(PLAN, task), [candidate])
        self.assertEqual(verdicts, {})
        self.assertEqual(len(errors), 1)

    def test_a_round_that_overruns_its_deadline_keeps_what_came_back(self):
        release = threading.Event()

        class Search(FakeSearch):
            def image(self, video_id, frame_idx, width=640):
                if video_id == "L01_V002":
                    release.wait(5)
                return b"jpeg"

        task = task_from_text("q-kis", "thuyền đỏ", "kis")
        from agent.pool import CandidatePool
        pool = CandidatePool()
        fast, slow = pool.add_frame("L01_V001", 100, 10.0), pool.add_frame("L01_V002", 200, 20.0)
        started = time.monotonic()
        verdicts, errors = MomentJudge(FakeQwen(), Search(HITS), deadline=0.3).verify(
            task, parse_plan(PLAN, task), [fast, slow])
        release.set()

        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(list(verdicts), [fast.cid])
        self.assertEqual(errors, [f"{slow.cid}: no verdict within 0 s"])

    def test_judge_names(self):
        self.assertEqual(judge_name(FakeQwen()), "qwen")
        self.assertEqual(judge_name(type("G", (), {"model": "gemini-3.8-flash"})()), "gemini")


class AutonomousRunTests(unittest.TestCase):
    def test_the_moment_judge_scores_the_run_with_the_querys_details(self):
        judge = ScriptedJudge({("L01_V002", 200): 9})
        agent = make_agent(RewritingDeepSeek(), WebGemini(), FakeSearch(HITS), mode="fast",
                           judge_style="moment")
        agent.judge = judge
        result = agent.run(task_from_text("q-kis", "thuyền đỏ"))

        self.assertIsNone(result.error)
        self.assertEqual((result.ranking[0].video_id, result.ranking[0].verdict.by), ("L01_V002", "qwen"))
        self.assertTrue(judge.calls)
        self.assertIsInstance(judge.details[0], QueryDetails)
        self.assertNotIn("falsify", [entry["tool"] for entry in result.trace])  # the judge already saw the shots around it
        data = result.to_dict()
        self.assertEqual(data["judge"], "qwen")
        self.assertTrue(any("Qwen" in line for line in data["strategy"]))

    def test_qa_reads_its_answer_at_the_best_judged_moments_after_judging(self):
        judge = ScriptedJudge({("L01_V002", 200): 9, ("L01_V002", 500): 8},
                              answers={("L01_V002", 200): "5", ("L01_V002", 500): "5"})
        search = FakeSearch(HITS + [("L01_V002", 500, 50.0)])
        gemini = WebGemini(reading="7", web_text="không rõ\nĐÁP ÁN: không rõ")
        agent = make_agent(RewritingDeepSeek({**PLAN, "question_vi": "mấy người?", "answer_source": "visual_count"}),
                           gemini, search, mode="fast", judge_style="moment", answer_moments=1)
        agent.judge = judge
        result = agent.run(task_from_text("q-qa", "Thuyền đỏ chở mấy người?", "qa"))

        reading = next(entry for entry in result.trace if entry["tool"] == "answer")
        tools = [entry["tool"] for entry in result.trace]
        self.assertLess(max(i for i, tool in enumerate(tools) if tool == "verify"), tools.index("answer"))
        self.assertIn(("5", 8.0), [(text, confidence) for text, confidence, _ in reading["args"]["readings"]])
        self.assertEqual(result.answer, "5")  # two judged readings outvote one dedicated reading

    def test_a_non_answer_is_no_vote_and_a_count_is_its_number(self):
        judge = ScriptedJudge({("L01_V002", 200): 9, ("L01_V002", 500): 8},
                              answers={("L01_V002", 200): "Không xác định được", ("L01_V002", 500): "12 chấm vàng"})
        search = FakeSearch(HITS + [("L01_V002", 500, 50.0)])
        gemini = WebGemini(reading="không rõ", web_text="ĐÁP ÁN: không rõ")
        agent = make_agent(RewritingDeepSeek({**PLAN, "question_vi": "mấy chấm?", "answer_source": "visual_count"}),
                           gemini, search, mode="fast", judge_style="moment", answer_moments=1)
        agent.judge = judge
        result = agent.run(task_from_text("q-qa", "Bản đồ có mấy chấm vàng?", "qa"))

        reading = next(entry for entry in result.trace if entry["tool"] == "answer")
        self.assertEqual([text for text, _, _ in reading["args"]["readings"]], ["12"])
        self.assertEqual(result.answer, "12")

    def test_a_side_questions_web_answer_never_becomes_the_answer(self):
        from agent.llm import GroundedAnswer

        class SideGemini(WebGemini):
            def web_search(self, question):
                self.web_prompts.append(question)
                if question.startswith("Answer briefly for a video search engine"):  # the plan's side question
                    return GroundedAnswer("Biển W.216.\nTỪ KHÓA: biển W.216\nĐÁP ÁN: Biển số W.216", (), ())
                return GroundedAnswer("ĐÁP ÁN: không rõ", (), ())

        judge = ScriptedJudge({("L01_V002", 200): 9})
        plan = {**PLAN, "question_vi": "Số trên biển báo?", "answer_source": "on_screen_text",
                "lookups": ["Biển báo xe lội nước là biển gì?"]}
        agent = make_agent(RewritingDeepSeek(plan), SideGemini(reading="không rõ"), FakeSearch(HITS),
                           mode="fast", judge_style="moment")
        agent.judge = judge
        result = agent.run(task_from_text("q-qa", "Số trên biển báo bên trái cầu là bao nhiêu?", "qa"))

        self.assertNotEqual(result.answer, "Biển số W.216")

    def test_moment_budgets_are_long_enough_for_two_rounds_of_judging(self):
        options = AgentOptions.for_mode("fast", judge_style="moment")
        self.assertEqual(options.time_budget, MOMENT_DEFAULTS["fast"]["time_budget"])
        self.assertFalse(options.falsify)
        self.assertEqual(AgentOptions.for_mode("fast").time_budget, 20.0)  # Gemini's keyframe judge unchanged
        self.assertEqual(AgentOptions.for_mode("fast", judge_style="moment", time_budget=40).time_budget, 40)


class WebStrategyTests(unittest.TestCase):
    def test_topics_the_query_names(self):
        self.assertEqual(detect_topic("Đoàn người rước kiệu trong lễ hội làng"), "festival")
        self.assertEqual(detect_topic("Khách du lịch chụp ảnh trước ngôi chùa cổ"), "place")
        self.assertEqual(detect_topic("Lễ khai mạc hội chợ nông sản"), "event")
        self.assertIsNone(detect_topic("thuyền đỏ trên sông"))

    def test_keywords_from_a_lookup(self):
        self.assertEqual(keywords("...\nTỪ KHÓA: Hội Gióng (hội Phù Đổng); đền Sóc, Sóc Sơn; không rõ\nĐÁP ÁN: x"),
                         ["Hội Gióng", "đền Sóc", "Sóc Sơn"])
        self.assertEqual(keywords("TỪ KHÓA: không rõ"), [])
        self.assertEqual(keywords(None), [])

    def test_a_festival_query_is_researched_from_the_start_and_its_names_searched(self):
        gemini = WebGemini({("L01_V003", 300): 6},
                           web_text="Hội Gióng ở đền Sóc.\nTỪ KHÓA: Hội Gióng; đền Sóc\nĐÁP ÁN: không rõ")
        search = FakeSearch(HITS)
        result = make_agent(RewritingDeepSeek(), gemini, search, mode="fast").run(
            task_from_text("q-kis", "Đoàn người rước kiệu trong lễ hội làng"))

        self.assertIn("lễ hội", gemini.web_prompts[0])  # before the plan: the query's words were enough
        rewrite = search.batches[1]
        spoken = [q["text"] for q in rewrite if q["kind"] == "audio"]
        self.assertEqual(sorted(spoken), ["Hội Gióng", "đền Sóc"])  # each name once
        self.assertTrue(all(q["method"] == "keyword" for q in rewrite if q["kind"] == "audio"))
        self.assertIn("Hội Gióng", [q["text"] for q in rewrite if q["kind"] == "ocr"])
        self.assertTrue(any(line.startswith("Lễ hội") for line in result.strategy))

    def test_the_planners_topic_starts_research_the_words_missed(self):
        gemini = WebGemini({("L01_V002", 200): 9})
        make_agent(RewritingDeepSeek({**PLAN, "topic": "place"}), gemini, FakeSearch(HITS), mode="fast").run(
            task_from_text("q-kis", "thuyền đỏ trên sông"))
        self.assertEqual(len(gemini.web_prompts), 1)
        self.assertIn("địa danh", gemini.web_prompts[0])

    def test_plan_topic(self):
        task = task_from_text("q", "x")
        self.assertEqual(parse_plan({**PLAN, "topic": "Festival"}, task).topic, "festival")
        self.assertEqual(parse_plan({**PLAN, "topic": "spaceship"}, task).topic, "other")

    def test_trake_events_as_the_query_words_them(self):
        agent = make_agent(RewritingDeepSeek({**PLAN, "events_en": ["a", "b"]}), WebGemini(), FakeSearch(HITS),
                           mode="fast")
        result = agent.run(task_from_text("q-trake", "Múa lân.\nE1: lân nhảy lên\nE2: lân ngã xuống"))
        self.assertEqual(result.to_dict()["events"], ["lân nhảy lên", "lân ngã xuống"])


class ConfigTests(unittest.TestCase):
    BASE = {"DEEPSEEK_API_KEY": "d", "AIC_AGENT_URL": "http://x"}

    def settings(self, **env):
        from pathlib import Path
        return AgentSettings.from_env({**self.BASE, **env}, dotenv=Path("/nonexistent"))

    def test_the_qwen_key_turns_on_the_qwen_moment_judge(self):
        settings = self.settings(QWEN_API_KEY="sk-or-1")
        self.assertEqual((settings.vision, settings.judge_style, settings.openrouter_api_key),
                         ("openrouter", "moment", "sk-or-1"))
        self.assertEqual(openrouter_key({"OPENROUTER_API_KEY": "a", "QWEN_API_KEY": "b"}), "a")

    def test_an_explicit_backend_still_wins(self):
        settings = self.settings(QWEN_API_KEY="sk-or-1", AIC_AGENT_VISION="gemini", GEMINI_API_KEY="g")
        self.assertEqual((settings.vision, settings.judge_style), ("gemini", "keyframe"))
        self.assertEqual(default_vision({}), "gemini")
        with self.assertRaises(RuntimeError):
            self.settings(QWEN_API_KEY="k", AIC_AGENT_JUDGE_STYLE="pixels")


class ShotCacheTests(unittest.TestCase):
    def test_concurrent_moments_of_one_video_fetch_its_shots_once(self):
        class Slow:
            calls = 0

            def shots(self, video_id):
                Slow.calls += 1
                time.sleep(0.05)
                return [{"index": 0}]

        search = Slow()
        threads = [threading.Thread(target=evidence._shots_of, args=(search, "L01_V001")) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(Slow.calls, 1)

    def test_a_failure_is_not_retried_at_once(self):
        class Down:
            calls = 0

            def shots(self, video_id):
                Down.calls += 1
                raise TimeoutError("slow host")

        search = Down()
        self.assertEqual(evidence._shots_of(search, "L01_V001"), ())
        self.assertEqual(evidence._shots_of(search, "L01_V001"), ())
        self.assertEqual(Down.calls, 1)


if __name__ == "__main__":
    unittest.main()
