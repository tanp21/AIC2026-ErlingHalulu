import csv
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from agent.evaluate import normalize_answer, score_result, summarize
from agent.report import render
from agent.submission import result_rows, to_csv, write_submission
from agent.tasks import GoldRow


def candidate(video, frame, score=None):
    return {"video_id": video, "frame_idx": frame, "time": 1.0,
            "verdict": None if score is None else {"score": score, "reason": "<b>lý do</b>"}}


class SubmissionTests(unittest.TestCase):
    def test_rows_follow_the_rules_for_each_kind(self):
        kis = {"query_id": "q-kis", "kind": "kis", "ranking": [
            candidate("L01_V001", 5), candidate("L01_V001", 5), candidate("L01_V002", 7)]}
        self.assertEqual(result_rows(kis), [["L01_V001", "5"], ["L01_V002", "7"]])

        qa = {"query_id": "q-qa", "kind": "qa", "answer": "Có 3 người, gồm \"nam\"",
              "ranking": [candidate("L01_V001", 5)]}
        text = to_csv(result_rows(qa))
        self.assertEqual(text, 'L01_V001,5,"Có 3 người, gồm ""nam"""\n')
        self.assertEqual(next(csv.reader(io.StringIO(text)))[2], 'Có 3 người, gồm "nam"')

        trake = {"query_id": "q-trake", "kind": "trake", "sequences": [
            {"video_id": "L01_V001", "frames": [1, 2, 3]},
            {"video_id": "L01_V001", "frames": [1, 2, 3]},
        ]}
        self.assertEqual(result_rows(trake), [["L01_V001", "1", "2", "3"]])

    def test_caps_rows_and_writes_zip_with_submission_folder(self):
        result = {"query_id": "q-kis", "kind": "kis",
                  "ranking": [candidate("L01_V001", i) for i in range(150)]}
        self.assertEqual(len(result_rows(result)), 100)
        with tempfile.TemporaryDirectory() as directory:
            archive = write_submission([result], Path(directory))
            with zipfile.ZipFile(archive) as opened:
                self.assertEqual(opened.namelist(), ["submission/q-kis.csv"])
                self.assertTrue(opened.read("submission/q-kis.csv").startswith(b"L01_V001,0\n"))


class EvaluateTests(unittest.TestCase):
    def test_hits_use_the_gold_range_and_tolerance(self):
        result = {"query_id": "q", "kind": "kis", "ranking": [
            candidate("L01_V002", 100), candidate("L01_V001", 3300), candidate("L01_V001", 3200)]}
        gold = [GoldRow("L01_V001", (3123,)), GoldRow("L01_V001", (3225,))]
        score = score_result(result, gold, tolerance=10)
        self.assertEqual((score.hit_rank, score.video_rank), (3, 2))
        self.assertEqual(score_result(result, gold, tolerance=80).hit_rank, 2)

    def test_qa_answers_compare_after_normalisation_and_trake_needs_every_event(self):
        qa = {"query_id": "q", "kind": "qa", "answer": " đèo tà pứa. ",
              "ranking": [candidate("L01_V001", 10)]}
        score = score_result(qa, [GoldRow("L01_V001", (12,), "Đèo Tà Pứa")], tolerance=5)
        self.assertTrue(score.answer_ok)
        self.assertEqual(normalize_answer('"2.15"'), "2 15")

        trake = {"query_id": "t", "kind": "trake", "sequences": [
            {"video_id": "L01_V001", "frames": [10, 200, 300]},
            {"video_id": "L01_V001", "frames": [10, 20, 30]},
        ]}
        self.assertEqual(score_result(trake, [GoldRow("L01_V001", (12, 22, 28))], 5).hit_rank, 2)

        summary = summarize([score, score_result(trake, [GoldRow("L01_V001", (1, 2, 3))], 5)])
        self.assertEqual(summary["qa"]["answer_acc"], 1.0)
        self.assertEqual(summary["all"]["R@1"], 0.5)


class ReportTests(unittest.TestCase):
    def test_escapes_model_text(self):
        page = render([{"query_id": "q", "kind": "kis", "query": "<script>", "elapsed": 1,
                        "trace": [], "ranking": [candidate("L01_V001", 5, 7)]}], "http://server")
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("&lt;b&gt;lý do&lt;/b&gt;", page)
        self.assertIn("http://server/thumb/L01_V001/5?w=320", page)


if __name__ == "__main__":
    unittest.main()
