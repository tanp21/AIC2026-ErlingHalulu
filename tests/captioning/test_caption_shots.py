import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from scripts.captioning import caption_shots


def write_video(root, video_id="L21_V001"):
    group = video_id[:3]
    directory = root / "2026" / "videos" / group
    directory.mkdir(parents=True)
    (directory / f"{video_id}.mp4").touch()


def write_shots(root, video_id="L21_V001", bounds=((0, 9), (10, 29))):
    group = video_id[:3]
    directory = root / "2026" / "shots" / group
    directory.mkdir(parents=True)
    (directory / f"{video_id}.json").write_text(json.dumps({
        "fps": 10.0, "shots": bounds,
    }))


def write_keyframes(root, names, video_id="L21_V001"):
    group = video_id[:3]
    directory = root / "2026" / "keyframes" / group / video_id
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).touch()


class CaptionShotsTests(unittest.TestCase):
    def test_remote_worker_can_skip_keyframe_mapping(self):
        args = caption_shots.parse_args(["--skip-keyframe-mapping"])
        self.assertTrue(args.skip_keyframe_mapping)

    def test_strips_model_timestamp_prefixes(self):
        caption = (
            "[00:00 - 00:02] A presenter enters. "
            "[00:03] She points at a chart."
        )
        self.assertEqual(
            caption_shots.strip_timestamps(caption),
            "A presenter enters. She points at a chart.",
        )

    def test_maps_every_keyframe_to_one_shot_and_ignores_temporary_pngs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_video(root)
            write_shots(root)
            write_keyframes(root, [
                "000001.png", "000005.png", "000010.png", "000020.png",
                "000025.png", "000015.tmp.png", "notes.txt",
            ])

            records = caption_shots.load_shot_records(root)
            rows, summary = caption_shots.build_keyframe_mapping(root, "2026", records)
            target = root / "mapping.parquet"
            caption_shots.atomic_write_table(rows, caption_shots.KEYFRAME_SCHEMA, target)

            self.assertEqual(
                [(row["frame_idx"], row["shot_index"]) for row in rows],
                [(1, 0), (5, 0), (10, 1), (20, 1), (25, 1)],
            )
            self.assertEqual(summary["ignored_temporary"], 1)
            self.assertEqual(summary["keyframes_per_shot"], {2: 1, 3: 1})
            self.assertEqual(pq.read_table(target).schema, caption_shots.KEYFRAME_SCHEMA)

    def test_mapping_rejects_a_shot_without_keyframes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_video(root)
            write_shots(root)
            write_keyframes(root, ["000001.png"])

            records = caption_shots.load_shot_records(root)
            with self.assertRaisesRegex(ValueError, "1 shots have no keyframes"):
                caption_shots.build_keyframe_mapping(root, "2026", records)

    def test_benchmark_sample_is_balanced_and_repeatable(self):
        records = []
        durations = (1, 3, 10, 20, 60, 180)
        for bucket, duration in enumerate(durations):
            for item in range(3):
                frames = int(duration * 10)
                records.append(caption_shots.ShotRecord(
                    "2026", "L21", f"L21_V{bucket:03d}", item,
                    item * frames, (item + 1) * frames - 1, 10.0, "/video.mp4",
                ))

        first = caption_shots.select_benchmark_shots(records, 12, seed=7)
        second = caption_shots.select_benchmark_shots(records, 12, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(
            {name: sum(caption_shots.duration_bucket(r.duration_sec) == name for r in first)
             for name, _, _ in caption_shots.DURATION_BINS},
            {name: 2 for name, _, _ in caption_shots.DURATION_BINS},
        )

    def test_benchmark_uses_only_duration_buckets_present_in_a_group(self):
        records = [
            caption_shots.ShotRecord(
                "2026", "L23", "L23_V001", index,
                index * 100, index * 100 + frames - 1, 10.0, "/video.mp4",
            )
            for index, frames in enumerate((10, 30, 100, 200, 600))
        ]

        selected = caption_shots.select_benchmark_shots(records, 5)

        self.assertEqual(selected, records)

    def test_shards_keep_videos_whole_and_balance_shots(self):
        records = []
        for video, count in enumerate((5, 4, 3, 2, 1)):
            records.extend(
                caption_shots.ShotRecord(
                    "2026", "L21", f"L21_V{video:03d}", shot,
                    shot * 10, shot * 10 + 9, 10.0, "/video.mp4",
                )
                for shot in range(count)
            )

        shards = [caption_shots.select_shard(records, 3, index) for index in range(3)]

        self.assertEqual([len(shard) for shard in shards], [5, 5, 5])
        self.assertEqual(
            sorted((row.video_id, row.shot_index) for shard in shards for row in shard),
            sorted((row.video_id, row.shot_index) for row in records),
        )
        owners = {}
        for index, shard in enumerate(shards):
            for row in shard:
                owners.setdefault(row.video_id, set()).add(index)
        self.assertTrue(all(len(indices) == 1 for indices in owners.values()))


if __name__ == "__main__":
    unittest.main()
