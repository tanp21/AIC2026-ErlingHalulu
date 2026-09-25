import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.ocr import merge_ocr_kaggle_shards as merger


def write_shard(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


LINE = {"text": "abc", "box": [[0, 0], [1, 0], [1, 1], [0, 1]], "conf": 0.9}


class IterShardRecordsTests(unittest.TestCase):
    def test_yields_key_and_lines_skipping_blank_lines(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.jsonl"
            write_shard(path, [
                {"key": "M01/M01_V001/3", "lines": [LINE]},
                {"key": "M01/M01_V001/11", "lines": []},
            ])
            path.write_text(path.read_text() + "\n")  # trailing blank line

            records = list(merger.iter_shard_records(path))

            self.assertEqual(records, [
                ("M01/M01_V001/3", [LINE]),
                ("M01/M01_V001/11", []),
            ])

    def test_malformed_json_line_raises(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "shard.jsonl"
            path.write_text("not json\n")

            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                list(merger.iter_shard_records(path))


class MergeShardsTests(unittest.TestCase):
    def test_merges_disjoint_shards(self):
        with TemporaryDirectory() as tmp:
            shard_a = Path(tmp) / "a.jsonl"
            shard_b = Path(tmp) / "b.jsonl"
            write_shard(shard_a, [{"key": "M01/M01_V001/3", "lines": [LINE]}])
            write_shard(shard_b, [{"key": "M02/M02_V001/5", "lines": []}])

            merged = merger.merge_shards([shard_a, shard_b])

            self.assertEqual(merged, {
                "M01/M01_V001/3": {"lines": [LINE]},
                "M02/M02_V001/5": {"lines": []},
            })

    def test_duplicate_key_across_shards_raises(self):
        with TemporaryDirectory() as tmp:
            shard_a = Path(tmp) / "a.jsonl"
            shard_b = Path(tmp) / "b.jsonl"
            write_shard(shard_a, [{"key": "M01/M01_V001/3", "lines": [LINE]}])
            write_shard(shard_b, [{"key": "M01/M01_V001/3", "lines": []}])

            with self.assertRaisesRegex(ValueError, "duplicate key.*M01/M01_V001/3"):
                merger.merge_shards([shard_a, shard_b])


class ValidateMergedTests(unittest.TestCase):
    def test_passes_on_well_formed_merged_dict(self):
        merger.validate_merged({"k": {"lines": [LINE]}}, expected_count=1)

    def test_expected_count_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "expected 2 keyframes, merged 1"):
            merger.validate_merged({"k": {"lines": []}}, expected_count=2)

    def test_lines_not_a_list_raises(self):
        with self.assertRaisesRegex(ValueError, "'lines' is not a list"):
            merger.validate_merged({"k": {"lines": "oops"}})

    def test_malformed_box_raises(self):
        bad = {"text": "x", "box": [[0, 0], [1, 0]], "conf": 0.5}
        with self.assertRaisesRegex(ValueError, "malformed box"):
            merger.validate_merged({"k": {"lines": [bad]}})

    def test_conf_out_of_range_raises(self):
        bad = dict(LINE, conf=1.5)
        with self.assertRaisesRegex(ValueError, "malformed conf"):
            merger.validate_merged({"k": {"lines": [bad]}})


if __name__ == "__main__":
    unittest.main()
