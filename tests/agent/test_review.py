import threading
import time
import unittest

from agent.review import HumanReview
from agent.tasks import task_from_text
from agent.vision import ANSWER_SCHEMA, VERIFY_SCHEMA
from tests.agent.test_orchestrator import HITS, PLAN, FakeDeepSeek, FakeGemini, FakeSearch, make_agent


class StalledGemini(FakeGemini):
    """Verification hangs until released, like Gemini on a congested venue network."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = threading.Event()

    def judge(self, images, prompt, schema, **kwargs):
        if schema in (VERIFY_SCHEMA, ANSWER_SCHEMA):
            self.release.wait(10)
        return super().judge(images, prompt, schema, **kwargs)


class HumanReviewTests(unittest.TestCase):
    def test_an_acceptance_or_a_stop_ends_the_run_a_rejection_does_not(self):
        review = HumanReview()
        review.decide("C1", "reject")
        self.assertFalse(review.decided())
        self.assertFalse(review.event.is_set())
        review.decide("C2", "accept")
        self.assertTrue(review.decided() and review.event.is_set())
        review.decide("C2", "clear")
        self.assertFalse(review.decided() or review.event.is_set())
        review.stop()
        self.assertTrue(review.decided())

    def test_answers_are_trimmed_and_unknown_decisions_refused(self):
        review = HumanReview()
        review.set_answer("  Đèo   Tà Pứa ")
        self.assertEqual(review.answer, "Đèo Tà Pứa")
        review.set_answer("   ")
        self.assertIsNone(review.answer)
        with self.assertRaises(ValueError):
            review.decide("C1", "maybe")


class HumanInTheLoopRunTests(unittest.TestCase):
    def test_accepting_a_candidate_ends_the_wait_on_a_stalled_gemini(self):
        gemini = StalledGemini({("L01_V002", 200): 9})
        review, snapshots = HumanReview(), []

        def watch(snapshot):
            snapshots.append(snapshot)
            judging = [c for c in snapshot["candidates"] if c["judging"]]
            if judging and not review.decided():  # the teammate picks while Gemini looks
                review.decide(next(c["id"] for c in judging if c["video_id"] == "L01_V003"), "accept")

        started = time.monotonic()
        try:
            result = make_agent(FakeDeepSeek(), gemini, FakeSearch(HITS), mode="fast").run(
                task_from_text("q-kis", "thuyền đỏ"), review=review, progress=watch)
        finally:
            gemini.release.set()

        self.assertLess(time.monotonic() - started, 5)  # did not sit out Gemini's 10 s
        top = result.ranking[0]
        self.assertEqual((top.video_id, top.verdict.by, top.verdict.score), ("L01_V003", "human", 10.0))
        self.assertIn("abandoned for a human decision",
                      next(e["summary"] for e in result.trace if e["tool"] == "verify"))
        self.assertEqual(snapshots[-1]["stage"], "done")

    def test_the_view_shows_what_gemini_is_judging_with_its_context(self):
        review, seen = HumanReview(), []
        search = FakeSearch(HITS)
        make_agent(FakeDeepSeek(), FakeGemini({("L01_V002", 200): 9}), search, mode="fast").run(
            task_from_text("q-kis", "thuyền đỏ"), review=review, progress=seen.append)

        judging = next(s for s in seen if any(c["judging"] for c in s["candidates"]))
        self.assertEqual(judging["stage"], "judging")
        self.assertEqual(judging["plan"]["scenes"], ["thuyền"])
        first = judging["candidates"][0]
        self.assertEqual(set(first), {"id", "video_id", "frame_idx", "time", "judging",
                                      "verdict", "context", "sources"})

    def test_a_rejection_outlasts_a_perfect_gemini_score(self):
        review = HumanReview()
        review.decide("C2", "reject")  # decided before Gemini ever saw it
        gemini = FakeGemini({("L01_V002", 200): 10, ("L01_V003", 300): 7})
        result = make_agent(FakeDeepSeek(), gemini, FakeSearch(HITS), mode="fast").run(
            task_from_text("q-kis", "thuyền đỏ"), review=review)

        rejected = next(c for c in result.ranking if c.cid == "C2")
        self.assertEqual((rejected.verdict.by, rejected.verdict.score), ("human", 0.0))
        self.assertEqual(result.ranking[-1].cid, "C2")
        self.assertNotIn("C2", gemini.judged)  # never sent to Gemini at all

    def test_withdrawing_a_decision_gives_gemini_its_verdict_back(self):
        review = HumanReview()
        gemini = FakeGemini({("L01_V002", 200): 8})
        changed_mind = []

        def watch(snapshot):
            judged = any(c["id"] == "C2" and c["verdict"] and c["verdict"]["by"] == "gemini"
                         for c in snapshot["candidates"])
            if judged and not changed_mind:
                changed_mind.append(True)
                review.decide("C2", "reject")  # overrules Gemini's 8...
            elif changed_mind == [True] and "C2" in review.decisions():
                review.decide("C2", "clear")   # ...then thinks better of it

        result = make_agent(FakeDeepSeek(), gemini, FakeSearch(HITS), mode="fast").run(
            task_from_text("q-kis", "thuyền đỏ"), review=review, progress=watch)
        restored = next(c for c in result.ranking if c.cid == "C2")
        self.assertEqual((restored.verdict.by, restored.verdict.score), ("gemini", 8))

    def test_a_typed_answer_replaces_the_stalled_reading(self):
        task = task_from_text("q-qa", "Con số trên cân là bao nhiêu?")
        plan = {**PLAN, "question_vi": "Con số trên cân?", "answer_source": "visual_count"}
        gemini = StalledGemini()
        gemini.release.set()  # verification answers; only the reading will stall
        answer_gate = threading.Event()
        original = gemini.judge

        def judge(images, prompt, schema, **kwargs):
            if schema is ANSWER_SCHEMA:
                answer_gate.wait(10)
            return original(images, prompt, schema, **kwargs)

        gemini.judge = judge
        review = HumanReview()

        def watch(snapshot):
            if snapshot["stage"] == "reading_answer" and not review.answer:
                review.set_answer("2.15")

        started = time.monotonic()
        try:
            result = make_agent(FakeDeepSeek(plan), gemini, FakeSearch(HITS), mode="fast").run(
                task, review=review, progress=watch)
        finally:
            answer_gate.set()
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual((result.answer, result.answer_evidence), ("2.15", "typed by the reviewer"))

    def test_stop_ends_the_run_with_the_current_ranking(self):
        gemini = StalledGemini()
        review = HumanReview()

        def watch(snapshot):
            if any(c["judging"] for c in snapshot["candidates"]):
                review.stop()

        started = time.monotonic()
        try:
            result = make_agent(FakeDeepSeek(), gemini, FakeSearch(HITS), mode="fast").run(
                task_from_text("q-kis", "thuyền đỏ"), review=review, progress=watch)
        finally:
            gemini.release.set()
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.ranking), 3)  # nothing lost, just not judged
        self.assertNotIn("rewrite", [e["tool"] for e in result.trace])


if __name__ == "__main__":
    unittest.main()
