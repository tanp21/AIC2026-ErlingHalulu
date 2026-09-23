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


class HumanJudgeReviewTests(unittest.TestCase):
    def test_a_person_judging_ends_the_run_only_by_finishing(self):
        review = HumanReview(judge="human")
        review.decide("C1", "accept")
        review.decide("C2", "accept")
        self.assertFalse(review.decided() or review.event.is_set())  # accepting only ranks
        review.finish()
        self.assertTrue(review.decided() and review.event.is_set())

    def test_scores_sit_between_accept_and_reject_and_replace_each_other(self):
        review = HumanReview(judge="human")
        review.decide("C1", "accept")
        review.score("C1", 5)
        self.assertEqual((review.verdicts(), review.decisions()), ({"C1": 5.0}, {}))
        review.decide("C1", "reject")
        self.assertEqual(review.verdicts(), {"C1": 0.0})
        review.decide("C1", "clear")
        self.assertEqual(review.verdicts(), {})
        with self.assertRaises(ValueError):
            review.score("C1", 11)

    def test_commands_are_taken_once_and_hints_trimmed(self):
        review = HumanReview(judge="human")
        review.push({"type": "more", "hint": "  đỏ   hơn  " + "x" * 400})
        review.push({"type": "scope", "video_id": "L01_V002"})
        self.assertEqual(review.snapshot()["pending_commands"], 2)
        first, second = review.take_commands()
        self.assertEqual(first["hint"][:6], "đỏ hơn")
        self.assertLessEqual(len(first["hint"]), 300)
        self.assertEqual(second["type"], "scope")
        self.assertEqual(review.take_commands(), [])
        with self.assertRaises(ValueError):
            review.push({"type": "delete_everything"})

    def test_the_order_on_the_reviewers_screen_is_kept_and_bounded(self):
        review = HumanReview(judge="human")
        review.set_order(["C7", "C3", "C1"])
        self.assertEqual(review.order, ["C7", "C3", "C1"])
        review.set_order([f"C{n}" for n in range(1000)])
        self.assertEqual(len(review.order), 300)

    def test_every_action_is_logged_with_its_second(self):
        now = [100.0]
        review = HumanReview(judge="human", clock=lambda: now[0])
        now[0] = 104.5
        review.decide("C3", "accept")
        now[0] = 110.0
        review.finish()
        log = review.log()
        self.assertEqual([(e["t"], e["action"]) for e in log], [(4.5, "decide"), (10.0, "finish")])


class ForbiddenVision(FakeGemini):
    """A human-judged run must never hand a frame to anything."""

    def judge(self, images, prompt, schema, **kwargs):
        raise AssertionError("a human-judged run sent a frame to a model")


def run_with_reviewer(script, *, task=None, plan=None, search=None, options=None, deepseek=None):
    """Run a human-judged query; `script(snapshot, review)` acts as the person on each publish."""
    review = HumanReview(judge="human")
    state = {"turn": 0}
    seen = []

    def watch(snapshot):
        seen.append(snapshot)
        if snapshot["stage"] == "awaiting_review":
            state["turn"] += 1
            script(state["turn"], snapshot, review)

    agent = make_agent(deepseek or FakeDeepSeek(plan or PLAN), ForbiddenVision(), search or FakeSearch(HITS),
                       mode="fast", judge="human", **(options or {}))
    result = agent.run(task or task_from_text("q-kis", "thuyền đỏ"), review=review, progress=watch)
    return result, review, seen


class HumanJudgedRunTests(unittest.TestCase):
    def test_the_run_waits_for_the_person_and_never_calls_a_model_on_a_frame(self):
        def script(turn, snapshot, review):
            if turn == 1:
                review.decide("C3", "accept")
                review.decide("C1", "reject")
            elif turn == 2:
                review.finish()

        result, review, seen = run_with_reviewer(script)
        self.assertIsNone(result.error)
        self.assertEqual([c.cid for c in result.ranking][:1], ["C3"])  # accepted first
        self.assertEqual(result.ranking[-1].cid, "C1")  # rejected last
        self.assertEqual({e["tool"] for e in result.trace} & {"verify", "falsify"}, set())
        self.assertGreaterEqual(len([s for s in seen if s["stage"] == "awaiting_review"]), 2)
        self.assertEqual([entry["action"] for entry in result.review_log][-1], "finish")

    def test_the_view_carries_channels_keywords_and_a_wide_pool(self):
        _, _, seen = run_with_reviewer(lambda turn, snapshot, review: review.finish())
        waiting = next(s for s in seen if s["stage"] == "awaiting_review")
        self.assertEqual(waiting["plan"]["keywords"]["vi"], ["thuyền đỏ"])
        self.assertIn("a red boat", waiting["plan"]["keywords"]["en"])
        first = waiting["candidates"][0]
        self.assertTrue(first["channels"] and {"kind", "source", "rank"} <= set(first["channels"][0]))
        self.assertEqual(waiting["total_candidates"], 3)

    def test_a_score_of_five_ranks_between_unjudged_and_accepted(self):
        def script(turn, snapshot, review):
            if turn == 1:
                review.score("C2", 5)
                review.decide("C3", "accept")
            else:
                review.finish()

        result, _, _ = run_with_reviewer(script)
        order = [c.cid for c in result.ranking]
        self.assertEqual(order[0], "C3")
        self.assertLess(order.index("C2"), order.index("C1"))  # "right video" beats unjudged
        self.assertEqual(next(c for c in result.ranking if c.cid == "C2").verdict.reason,
                         "Scored 5/10 by the reviewer")

    def test_more_runs_a_rewrite_with_the_hint_and_never_judges_it(self):
        deepseek = FakeDeepSeek()
        prompts = []
        original = deepseek.chat_json

        def chat_json(messages, **kwargs):
            prompts.append(messages[-1]["content"])
            return original(messages, **kwargs)

        deepseek.chat_json = chat_json
        search = FakeSearch(HITS)

        def script(turn, snapshot, review):
            if turn == 1:
                review.push({"type": "more", "hint": "chiếc thuyền phải màu đỏ"})
            else:
                review.finish()

        result, _, _ = run_with_reviewer(script, search=search, deepseek=deepseek)
        rewrite_prompt = prompts[-1]
        self.assertIn("chiếc thuyền phải màu đỏ", rewrite_prompt)
        self.assertIn("PERSON", rewrite_prompt)
        self.assertNotIn("Budget:", rewrite_prompt)
        self.assertTrue(any(q["id"].startswith("s1.scoped") for batch in search.batches for q in batch))
        self.assertEqual([e for e in result.trace if e["tool"] == "verify"], [])

    def test_scope_searches_inside_the_chosen_video_only(self):
        search = FakeSearch(HITS)

        def script(turn, snapshot, review):
            if turn == 1:
                review.push({"type": "scope", "video_id": "L01_V002"})
            else:
                review.finish()

        run_with_reviewer(script, search=search)
        scoped = [q for batch in search.batches for q in batch if q["id"].startswith("s1.scoped")]
        self.assertTrue(scoped)
        self.assertTrue(all(q["video_ids"] == ["L01_V002"] for q in scoped))

    def test_neighbours_and_an_exact_frame_join_the_pool_and_stay_in_view(self):
        def script(turn, snapshot, review):
            if turn == 1:
                review.push({"type": "neighbours", "cid": "C2"})
                review.push({"type": "add_frame", "video_id": "L01_V009", "frame_idx": 777,
                             "time": 77.7, "accept": True})
            else:
                review.finish()

        result, _, seen = run_with_reviewer(script)
        added = next(c for c in result.ranking if (c.video_id, c.frame_idx) == ("L01_V009", 777))
        self.assertEqual((added.verdict.by, added.verdict.score), ("human", 10.0))
        self.assertEqual(result.ranking[0].cid, added.cid)
        pinned = [c for c in seen[-2]["candidates"] if c["pinned"]]
        self.assertGreaterEqual(len(pinned), 5)  # C2's four neighbours and the exact frame

    def test_the_submission_follows_the_order_the_reviewer_saw(self):
        """C1 leads the agent's fusion; the screen put C3 first because its words said the details."""
        def script(turn, snapshot, review):
            review.set_order(["C3", "C1", "C2"])
            review.finish()

        result, _, _ = run_with_reviewer(script)
        self.assertEqual([c.cid for c in result.ranking], ["C3", "C1", "C2"])

    def test_what_was_accepted_still_leads_and_what_was_rejected_still_trails(self):
        def script(turn, snapshot, review):
            review.decide("C2", "accept")
            review.decide("C3", "reject")
            review.set_order(["C3", "C1", "C2"])    # the screen listed the rejected one first; it must not stay there
            review.finish()

        result, _, _ = run_with_reviewer(script)
        self.assertEqual([c.cid for c in result.ranking], ["C2", "C1", "C3"])

    def test_a_candidate_the_screen_never_listed_follows_the_listed_ones(self):
        def script(turn, snapshot, review):
            review.set_order(["C3"])
            review.finish()

        result, _, _ = run_with_reviewer(script)
        self.assertEqual([c.cid for c in result.ranking], ["C3", "C1", "C2"])   # then the agent's own order

    def test_without_an_order_the_agents_own_order_stands(self):
        result, _, _ = run_with_reviewer(lambda turn, snapshot, review: review.finish())
        self.assertEqual([c.cid for c in result.ranking], ["C1", "C2", "C3"])

    def test_a_command_that_fails_is_a_note_not_the_end_of_the_run(self):
        search = FakeSearch(HITS)

        def broken(video_id, frame_idx, radius=6):
            raise RuntimeError("HTTP 500")

        def script(turn, snapshot, review):
            if turn == 1:
                search.neighbours = broken
                review.push({"type": "neighbours", "cid": "C1"})
            else:
                review.finish()

        result, _, _ = run_with_reviewer(script, search=search)
        self.assertIsNone(result.error)
        self.assertIn("failed: RuntimeError", next(e["summary"] for e in result.trace if e["tool"] == "human"
                                                   and e["args"].get("type") == "neighbours"))

    def test_a_silent_reviewer_times_out_with_the_current_ranking(self):
        started = time.monotonic()
        result, _, _ = run_with_reviewer(lambda turn, snapshot, review: None,
                                         options={"human_idle_timeout": 0.6})
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.ranking), 3)
        self.assertIn("no reviewer activity", next(e["summary"] for e in result.trace if e["tool"] == "human"))

    def test_the_typed_answer_is_the_answer_and_no_model_reads_the_video(self):
        task = task_from_text("q-qa", "Con số trên cân là bao nhiêu?")
        plan = {**PLAN, "question_vi": "Con số trên cân?", "answer_source": "visual_count"}

        def script(turn, snapshot, review):
            review.set_answer("2.15")
            review.finish()

        result, _, _ = run_with_reviewer(script, task=task, plan=plan)
        self.assertEqual((result.answer, result.answer_evidence), ("2.15", "typed by the reviewer"))

    def test_the_reviewers_sequence_leads_the_trake_paths(self):
        task = task_from_text("q-trake", "E1: thuyền\nE2: cầu")
        plan = {**PLAN, "events_en": ["a boat", "a bridge"]}

        def script(turn, snapshot, review):
            review.set_sequence("L01_V002", [200, 260], [20.0, 26.0])
            review.finish()

        result, _, _ = run_with_reviewer(script, task=task, plan=plan)
        self.assertEqual((result.sequences[0].source, result.sequences[0].video_id, result.sequences[0].frames),
                         ("human", "L01_V002", [200, 260]))
        rows = __import__("agent.submission", fromlist=["result_rows"]).result_rows(result.to_dict())
        self.assertEqual(rows[0], ["L01_V002", "200", "260"])


if __name__ == "__main__":
    unittest.main()
