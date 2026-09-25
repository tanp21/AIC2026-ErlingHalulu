"""The parts of the human-mode agent benchmark that decide what gets run and how it is scored.

The run itself needs the live server, both model keys and three hours; these tests cover the
two places a mistake is silent -- which queries are excluded, and how the collaboration
database's rows become gold frames -- so a wrong benchmark is caught before it is paid for.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.bench_agent_human import gold_of, tasks_and_gold


def query(filename, kind, answers):
    return {"filename": filename, "kind": kind, "prompt": f"{filename} prompt", "answers": answers}


ROUND = {"queries": [
    query("query-p2-1-kis", "kis", [{"video_id": "L24_V035", "frame_id": 640},
                                    {"video_id": "L24_V035", "frame_id": 628, "auto_filled": True},
                                    {"video_id": "L24_V018", "frame_id": 692, "auto_filled": True}]),
    query("query-p2-7-qa", "qa", [{"video_id": "L21_V009", "frame_id": 21172, "answer": "1204"}]),
    query("query-p2-27-qa", "qa", [{"video_id": "L26_V001", "frame_id": 5, "answer": "x"}]),
    query("query-p2-9-kis", "kis", []),  # nobody answered: nothing to score against
]}


class GoldTests(unittest.TestCase):
    def test_every_accepted_frame_is_a_row_of_its_own(self):
        """`score_result` reads `frames[0]` of every row, so a collapsed row hides the rest."""
        rows = gold_of(ROUND["queries"][0])

        self.assertEqual([(row.video_id, row.frames) for row in rows],
                         [("L24_V035", (640,)), ("L24_V035", (628,)), ("L24_V018", (692,))])

    def test_a_qa_row_carries_the_answer_the_round_recorded(self):
        row = gold_of(ROUND["queries"][1])[0]
        self.assertEqual((row.video_id, row.frames, row.answer), ("L21_V009", (21172,), "1204"))

    def test_a_trake_row_is_one_frame_per_event_carried_in_frame_ids(self):
        """Reported by the first round-3 run: reading only `frame_id` dropped every TRAKE query."""
        rows = gold_of(query("query-p2-21-trake", "trake",
                             [{"video_id": "L26_V156", "frame_ids": [3261, 3767, 4785, 6728]}]))

        self.assertEqual([(row.video_id, row.frames) for row in rows],
                         [("L26_V156", (3261, 3767, 4785, 6728))])

    def test_a_row_without_a_video_id_is_left_out_rather_than_gold_with_no_video(self):
        self.assertEqual(gold_of(query("q", "kis", [{"frame_id": 3}, {"video_id": " "}])), [])


class TaskSelectionTests(unittest.TestCase):
    def test_the_rounds_queries_are_kept_and_the_named_ones_are_not_run_at_all(self):
        tasks, golds, kinds = tasks_and_gold(ROUND, exclude=["query-p2-27-qa"])

        self.assertEqual([task.query_id for task in tasks], ["query-p2-1-kis", "query-p2-7-qa"])
        self.assertNotIn("query-p2-27-qa", golds)
        self.assertEqual(kinds["query-p2-7-qa"], "qa")

    def test_a_query_with_no_answers_is_left_out(self):
        tasks, _, _ = tasks_and_gold(ROUND)
        self.assertNotIn("query-p2-9-kis", [task.query_id for task in tasks])

    def test_only_runs_the_named_pilot(self):
        tasks, _, _ = tasks_and_gold(ROUND, only=["query-p2-7-qa"])
        self.assertEqual([task.query_id for task in tasks], ["query-p2-7-qa"])

    def test_each_round_is_scored_against_its_own_answers(self):
        """Round 3 reuses round 2's file names: the same name must mean two different queries."""
        round3 = {"queries": [query("query-p2-1-kis", "kis",
                                    [{"video_id": "L26_V424", "frame_id": 5518}])]}
        _, round2_gold, _ = tasks_and_gold(ROUND)
        _, round3_gold, _ = tasks_and_gold(round3)

        self.assertEqual(round2_gold["query-p2-1-kis"][0].video_id, "L24_V035")
        self.assertEqual(round3_gold["query-p2-1-kis"][0].video_id, "L26_V424")


if __name__ == "__main__":
    unittest.main()
