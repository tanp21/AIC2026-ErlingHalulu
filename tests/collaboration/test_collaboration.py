import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from backend.collaboration import CollaborationError, CollaborationStore


def query_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class CollaborationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = CollaborationStore(Path(self.temp_dir.name) / "collab.sqlite3")
        self.owner = "owner@example.com"
        self.users = frozenset({self.owner, "mate@example.com"})

    def import_round(self):
        return self.store.import_round(
            "Round P1",
            query_zip({
                "packet/query-p1-1-kis.txt": "Tìm cảnh phóng tàu",
                "packet/query-p1-2-qa.txt": "Có bao nhiêu người?",
                "packet/query-p1-3-trake.txt": "E1: Bắt đầu\nE2: Tiếp theo\nE3: Kết thúc",
            }),
            self.owner,
        )

    def test_imports_queries_and_keeps_only_one_active_round(self):
        first = self.import_round()
        second = self.store.import_round(
            "Round P2", query_zip({"query-p2-1-kis.txt": "Cảnh khác"}), self.owner
        )

        rounds = self.store.list_rounds()
        self.assertEqual([row["name"] for row in rounds], ["Round P2", "Round P1"])
        self.assertEqual([row["active"] for row in rounds], [True, False])
        snapshot = self.store.get_round(first["id"])
        self.assertFalse(snapshot["active"])
        self.assertEqual(snapshot["revision"], 1)
        self.assertEqual([query["kind"] for query in snapshot["queries"]], ["kis", "qa", "trake"])
        self.assertIsNone(snapshot["queries"][2]["event_count"])
        self.assertGreater(second["id"], first["id"])

    def test_rejects_unsafe_or_malformed_archives_atomically(self):
        cases = [
            query_zip({"../query-1-kis.txt": "bad"}),
            query_zip({"query-1.txt": "bad"}),
        ]
        for payload in cases:
            with self.subTest(payload=payload[:20]):
                with self.assertRaises(CollaborationError):
                    self.store.import_round("Bad", payload, self.owner)
        self.assertEqual(self.store.list_rounds(), [])

    def test_does_not_parse_trake_markers(self):
        imported = self.store.import_round(
            "Typo labels",
            query_zip({"query-1-trake.txt": "E1: first\nE2: second\nE2: third\nE4: fourth"}),
            self.owner,
        )
        self.assertIsNone(imported["queries"][0]["event_count"])

    def test_imports_unstructured_trake_prompt(self):
        imported = self.store.import_round(
            "Organizer format",
            query_zip({"query-1-trake.txt": "E1 first\nE2 second\nE3 third"}),
            self.owner,
        )
        self.assertIsNone(imported["queries"][0]["event_count"])

    def test_does_not_parse_vietnamese_trake_labels(self):
        # SOTUYEN2 query-p2-21 labels its events "Cảnh 1:" … "Cảnh 4:" instead of "E1:".
        prompt = (
            "4 cảnh này xảy ra liên tiếp nhau. \n"
            "Cảnh 1: Hai người phụ nữ cùng nhau dán niêm phong một thùng carton.\n"
            "Cảnh 2: Các thùng mì tôm và bọc bánh mì được sắp xếp ngay ngắn.\n"
            "Cảnh 3: Một người đàn ông nhấc thùng mì tôm lên.\n"
            "Cảnh 4: Cảnh quay cận cảnh các thùng mì được xếp chồng trên xe tải."
        )
        imported = self.store.import_round(
            "Vietnamese labels", query_zip({"query-1-trake.txt": prompt}), self.owner
        )
        self.assertIsNone(imported["queries"][0]["event_count"])

    def test_keeps_event_count_empty_regardless_of_prompt_format(self):
        prompt = (
            "Cảnh 2 người phụ nữ đang đi bộ trong công viên.\n"
            "E1: Khoảnh khắc đầu tiên.\n"
            "E2: Khoảnh khắc thứ hai."
        )
        imported = self.store.import_round(
            "Prose guard", query_zip({"query-1-trake.txt": prompt}), self.owner
        )
        self.assertIsNone(imported["queries"][0]["event_count"])

    def test_saves_drafts_derives_status_and_detects_stale_versions(self):
        round_row = self.import_round()
        query = self.store.get_round(round_row["id"])["queries"][0]

        claimed = self.store.update_query(
            query["id"], query["version"], self.owner, self.users, assignee="mate@example.com"
        )
        self.assertEqual(claimed["status"], "in_progress")
        ready = self.store.update_query(
            query["id"], claimed["version"], self.owner, self.users,
            answers=[{"video_id": "L00_V001", "frame_id": 123}],
        )
        self.assertEqual(ready["status"], "ready")
        with self.assertRaisesRegex(CollaborationError, "stale") as raised:
            self.store.update_query(
                query["id"], claimed["version"], self.owner, self.users, answers=[]
            )
        self.assertEqual(raised.exception.status_code, 409)

    def test_review_requires_valid_answers_and_answer_edit_clears_review(self):
        round_row = self.import_round()
        query = self.store.get_round(round_row["id"])["queries"][1]
        draft = self.store.update_query(
            query["id"], 0, self.owner, self.users,
            answers=[{"video_id": "L00_V001", "frame_id": None, "answer": "5"}],
        )
        with self.assertRaisesRegex(CollaborationError, "valid"):
            self.store.set_review(query["id"], draft["version"], self.owner, True)

        ready = self.store.update_query(
            query["id"], draft["version"], self.owner, self.users,
            answers=[{"video_id": "L00_V001", "frame_id": 10, "answer": "5"}],
        )
        reviewed = self.store.set_review(query["id"], ready["version"], "mate@example.com", True)
        self.assertEqual(reviewed["status"], "reviewed")
        edited = self.store.update_query(
            query["id"], reviewed["version"], self.owner, self.users,
            answers=[{"video_id": "L00_V001", "frame_id": 11, "answer": "5"}],
        )
        self.assertIsNone(edited["reviewed_by"])
        self.assertEqual(edited["status"], "ready")

    def test_validates_trake_event_count_and_order(self):
        round_row = self.import_round()
        query = self.store.get_round(round_row["id"])["queries"][2]
        query = self.store.update_query(
            query["id"], query["version"], self.owner, self.users, event_count=3
        )
        for frames in ([1, 2], [1, None, 3], [1, 3, 2]):
            draft = self.store.update_query(
                query["id"], query["version"], self.owner, self.users,
                answers=[{"video_id": "L00_V001", "frame_ids": frames}],
            )
            self.assertEqual(draft["status"], "in_progress")
            query = draft

        ready = self.store.update_query(
            query["id"], query["version"], self.owner, self.users,
            answers=[{"video_id": "L00_V001", "frame_ids": [1, 2, 3]}],
        )
        self.assertEqual(ready["status"], "ready")

    def test_builds_rule_compliant_zip_and_forced_export_omits_invalid_queries(self):
        round_row = self.import_round()
        queries = self.store.get_round(round_row["id"])["queries"]
        kis = self.store.update_query(
            queries[0]["id"], 0, self.owner, self.users,
            answers=[{"video_id": "L00_V001", "frame_id": 123}],
        )
        qa = self.store.update_query(
            queries[1]["id"], 0, self.owner, self.users,
            answers=[
                {"video_id": "L00_V002", "frame_id": 456, "answer": "5"},
                {"video_id": "L00_V003", "frame_id": 457, "answer": 'Ba, nói "xin chào"'},
                {"video_id": "L00_V004", "frame_id": 458, "answer": "Dòng 1\nDòng 2"},
            ],
        )
        self.store.set_review(kis["id"], kis["version"], self.owner, True)

        preview = self.store.export_preview(round_row["id"])
        self.assertEqual(preview["missing"], ["query-p1-3-trake"])
        self.assertEqual(preview["unreviewed"], ["query-p1-2-qa"])
        with self.assertRaisesRegex(CollaborationError, "warnings"):
            self.store.build_export(round_row["id"], allow_incomplete=False)

        payload, filename = self.store.build_export(round_row["id"], allow_incomplete=True)
        self.assertEqual(filename, "RoundP1.zip")
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(archive.namelist(), [
                "submission/query-p1-1-kis.csv", "submission/query-p1-2-qa.csv"
            ])
            kis_text = archive.read("submission/query-p1-1-kis.csv").decode("utf-8")
            qa_text = archive.read("submission/query-p1-2-qa.csv").decode("utf-8")
        self.assertEqual(kis_text, "L00_V001,123\n")
        self.assertEqual(
            qa_text,
            'L00_V002,456,"5"\n'
            'L00_V003,457,"Ba, nói ""xin chào"""\n'
            'L00_V004,458,"Dòng 1\nDòng 2"\n',
        )
        self.assertEqual(
            list(csv.reader(io.StringIO(qa_text))),
            [
                ["L00_V002", "456", "5"],
                ["L00_V003", "457", 'Ba, nói "xin chào"'],
                ["L00_V004", "458", "Dòng 1\nDòng 2"],
            ],
        )

    def test_export_uses_one_snapshot_and_ascii_filename(self):
        imported = self.store.import_round(
            "Vòng Đặc Biệt", query_zip({"query-1-kis.txt": "Tìm cảnh"}), self.owner
        )
        query = imported["queries"][0]
        self.store.update_query(
            query["id"], 0, self.owner, self.users,
            answers=[{"video_id": "L00_V001", "frame_id": 1}],
        )
        with patch.object(self.store, "export_preview", side_effect=AssertionError("second snapshot")):
            _payload, filename = self.store.build_export(imported["id"], allow_incomplete=True)
        self.assertEqual(filename, "VongDacBiet.zip")


if __name__ == "__main__":
    unittest.main()
