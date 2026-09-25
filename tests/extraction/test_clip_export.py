import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import numpy as np

from backend import app as server
from backend import frame_maps
from backend import shots
from backend.access_control import AccessConfig, LatestWinsCoordinator
from scripts.extraction import detect_shots

OWNER = "owner@example.com"
ROW = {"video_id": "L21_V001", "video_idx": 0, "group": "L21", "year": "2026",
       "path": "/data/2026/videos/L21/L21_V001.mp4", "fps": 30.0}
# On disk but outside the embedded profiles, so it carries no fps from the index.
UNINDEXED_ROW = {"video_id": "K01_V001", "group": "K01", "year": "2025",
                 "path": "/data/2025/videos/K01/K01_V001.mp4"}


def index(*bounds, fps=30.0):
    return shots.ShotIndex(fps, np.asarray(bounds, dtype=np.int64).reshape(-1, 2))


def mapping(count=1000, delta=3003, time_base_den=90000):
    return frame_maps.FrameMap(np.arange(count, dtype=np.int64) * delta, 1, time_base_den)


class ShotIndexTests(unittest.TestCase):
    def test_shot_at_resolves_interior_and_boundary_frames(self):
        found = index([0, 9], [10, 24], [30, 39])

        self.assertEqual(found.shot_at(0), 0)
        self.assertEqual(found.shot_at(9), 0)
        self.assertEqual(found.shot_at(10), 1)
        self.assertEqual(found.shot_at(24), 1)
        self.assertEqual(found.shot_at(39), 2)

    def test_shot_at_returns_none_outside_any_shot(self):
        found = index([0, 9], [30, 39])

        self.assertIsNone(found.shot_at(-1))
        self.assertIsNone(found.shot_at(20))
        self.assertIsNone(found.shot_at(40))
        self.assertIsNone(index().shot_at(0))

    def test_clamped_trims_shots_that_run_past_the_frame_map(self):
        clamped = index([0, 9], [10, 24], [30, 60]).clamped(frame_count=40)

        self.assertEqual(clamped.bounds.tolist(), [[0, 9], [10, 24], [30, 39]])

    def test_clamped_drops_shots_starting_beyond_the_frame_map(self):
        clamped = index([0, 9], [50, 60]).clamped(frame_count=40)

        self.assertEqual(clamped.bounds.tolist(), [[0, 9]])
        self.assertEqual(len(index([50, 60]).clamped(frame_count=40)), 0)

    def test_load_path_reads_the_transnetv2_json_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "L21_V001.json"
            path.write_text(json.dumps(
                {"fps": 25.0, "shots": [[0, 26], [27, 289]], "shots_sec": [[0.0, 1.04]]}
            ))

            loaded = shots.load_path(path)

        self.assertEqual(loaded.fps, 25.0)
        self.assertEqual(loaded.bounds.tolist(), [[0, 26], [27, 289]])

    def test_shots_path_mirrors_the_video_layout(self):
        self.assertEqual(
            shots.shots_path(ROW, data_dir="/data"),
            Path("/data/2026/shots/L21/L21_V001.json"),
        )

    def test_scan_media_finds_videos_from_every_dataset_year(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for year, group, name in (("2025", "K01", "K01_V001"),
                                      ("2026", "L21", "L21_V001"),
                                      ("2026", "L21", "L21_V002")):
                (root / year / "videos" / group).mkdir(parents=True, exist_ok=True)
                (root / year / "videos" / group / f"{name}.mp4").touch()
            (root / "2026" / "videos" / "N023").mkdir(parents=True)
            (root / "2026" / "videos" / "N023" / "N023_V002.mov").touch()
            (root / "2026" / "keyframes" / "L21").mkdir(parents=True)
            (root / "2026" / "keyframes" / "L21" / "decoy.mp4").touch()

            catalogue = shots.scan_media(root)

        self.assertEqual(sorted(catalogue), ["K01_V001", "L21_V001", "L21_V002", "N023_V002"])
        self.assertEqual(catalogue["K01_V001"]["group"], "K01")
        self.assertEqual(catalogue["K01_V001"]["year"], "2025")
        self.assertTrue(catalogue["K01_V001"]["path"].endswith("2025/videos/K01/K01_V001.mp4"))
        self.assertTrue(catalogue["N023_V002"]["path"].endswith("2026/videos/N023/N023_V002.mov"))

    def test_rejects_unordered_bounds(self):
        with self.assertRaises(ValueError):
            index([10, 24], [0, 9])


class ClipCommandTests(unittest.TestCase):
    def test_bounds_sit_half_a_frame_outside_the_shot_on_both_edges(self):
        """Landing exactly on a PTS lets %.6f rounding add or drop a boundary frame."""
        start, end = shots.shot_bounds_seconds(mapping(), start_frame=30, end_frame=59)

        self.assertAlmostEqual(start, 29.5 * 3003 / 90000)
        self.assertAlmostEqual(end, 59.5 * 3003 / 90000)

    def test_first_frame_bound_never_goes_negative(self):
        start, _ = shots.shot_bounds_seconds(mapping(), start_frame=0, end_frame=9)

        self.assertEqual(start, 0.0)

    def test_final_frame_duration_falls_back_to_the_previous_delta(self):
        _, end = shots.shot_bounds_seconds(mapping(count=10), start_frame=0, end_frame=9)

        self.assertAlmostEqual(end, 9.5 * 3003 / 90000)

    def test_command_seeks_on_the_source_clock_and_re_encodes_with_audio(self):
        command = shots.clip_command("video.mp4", 1.001, 2.002, "out.mp4")
        joined = " ".join(command)

        self.assertIn("-copyts", command)
        self.assertNotIn("-c copy", joined)
        self.assertIn("1.001000", joined)
        self.assertIn("2.002000", joined)
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[-1], "out.mp4")

    def test_export_runs_ffmpeg_under_the_clip_semaphore(self):
        with patch("backend.shots.subprocess.run") as run:
            shots.export("video.mp4", 0.0, 1.0, "out.mp4")

        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][0], "ffmpeg")
        self.assertTrue(run.call_args.kwargs["check"])


class ShotDetectionTests(unittest.TestCase):
    """The pure half of scripts.extraction.detect_shots (no TransNetV2 needed)."""

    def detect(self, predictions, **kwargs):
        model = Mock()
        model.predict_video.return_value = (None, np.asarray(predictions, dtype=np.float32), None)
        return detect_shots.detect_shots(model, "video.mp4", **kwargs)

    def test_cuts_at_the_peak_of_each_transition_run(self):
        predictions = np.zeros(120)
        predictions[39:42] = [0.6, 0.9, 0.7]
        predictions[79:81] = [0.95, 0.8]

        self.assertEqual(self.detect(predictions), [(0, 39), (40, 78), (79, 119)])

    def test_ignores_transitions_that_would_make_a_shot_too_short(self):
        predictions = np.zeros(100)
        predictions[20] = 0.9
        predictions[25] = 0.9  # only 5 frames later

        self.assertEqual(self.detect(predictions), [(0, 19), (20, 99)])

    def test_covers_every_frame_with_no_transition_at_all(self):
        self.assertEqual(self.detect(np.zeros(90)), [(0, 89)])

    def test_collect_jobs_skips_existing_output_and_honours_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for group, name in (("L21", "L21_V001"), ("L21", "L21_V002"), ("L22", "L22_V001")):
                (root / "videos" / group).mkdir(parents=True, exist_ok=True)
                (root / "videos" / group / f"{name}.mp4").touch()
            (root / "shots" / "L21").mkdir(parents=True)
            (root / "shots" / "L21" / "L21_V001.json").write_text("{}")

            jobs = detect_shots.collect_jobs(
                root / "videos", root / "shots", groups={"L21"}, force=False
            )
            forced = detect_shots.collect_jobs(
                root / "videos", root / "shots", groups=set(), force=True
            )

        self.assertEqual([Path(video).name for video, _ in jobs], ["L21_V002.mp4"])
        self.assertEqual(Path(jobs[0][1]).name, "L21_V002.json")
        self.assertEqual(len(forced), 3)


class ClipRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.app.state.access_config = AccessConfig(
            frozenset({OWNER, "b@x.com", "c@x.com", "d@x.com", "e@x.com"}), OWNER
        )
        server.app.state.search_coordinator = LatestWinsCoordinator()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )
        self.headers = {"X-AIC-User": OWNER}
        meta = patch("backend.app.core.load_meta", return_value=(None, None, {0: ROW}, {"L21_V001": ROW}))
        exists = patch("backend.app.os.path.exists", return_value=True)
        frame_sources = patch("backend.app.core.list_frame_sources", return_value=[])
        self.addCleanup(meta.stop)
        self.addCleanup(exists.stop)
        self.addCleanup(frame_sources.stop)
        meta.start()
        exists.start()
        frame_sources.start()

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_video_list_flags_missing_shot_boundaries(self):
        with patch("backend.app.shots.has_shots", return_value=False), \
                patch("backend.app.shots.media_catalogue", return_value={}):
            response = await self.client.get("/videos", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [
            {"video_id": "L21_V001", "group": "L21", "year": "2026",
             "fps": 30.0, "indexed": True, "has_shots": False},
        ])

    async def test_video_list_includes_datasets_outside_the_search_index(self):
        with patch("backend.app.shots.has_shots", return_value=True), \
                patch("backend.app.shots.media_catalogue", return_value={"K01_V001": UNINDEXED_ROW}):
            response = await self.client.get("/videos", headers=self.headers)

        body = response.json()
        self.assertEqual([row["video_id"] for row in body], ["K01_V001", "L21_V001"])
        self.assertEqual([row["indexed"] for row in body], [False, True])
        self.assertIsNone(body[0]["fps"])

    async def test_unindexed_video_resolves_through_the_media_catalogue(self):
        with patch("backend.app.shots.media_catalogue", return_value={"K01_V001": UNINDEXED_ROW}), \
                patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9], [10, 24])):
            response = await self.client.get("/shots/K01_V001", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["shots"]), 2)

    async def test_video_missing_from_disk_and_index_is_still_404(self):
        with patch("backend.app.shots.media_catalogue", return_value={}):
            response = await self.client.get("/shots/K01_V001", headers=self.headers)

        self.assertEqual(response.status_code, 404)

    async def test_shots_endpoint_converts_frames_through_the_frame_map(self):
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9], [10, 24])):
            response = await self.client.get("/shots/L21_V001", headers=self.headers)

        body = response.json()
        self.assertEqual(body["frameCount"], 1000)
        self.assertEqual([shot["startFrame"] for shot in body["shots"]], [0, 10])
        self.assertAlmostEqual(body["shots"][1]["start"], 10 * 3003 / 90000)
        self.assertAlmostEqual(body["shots"][1]["end"], 25 * 3003 / 90000)
        self.assertEqual(body["shots"][1]["reviewFrames"], [10, 17, 24])
        self.assertEqual(body["labelRevision"], 0)

    async def test_shots_endpoint_returns_review_labels(self):
        labels = {"version": 1, "video_id": "L21_V001", "shots_digest": "x",
                  "revision": 4, "labels": {"1": {
                      "state": "suggested", "reason": "presenter", "confidence": .96,
                  }}}
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9], [10, 24])), \
                patch("backend.app.shot_labels.load", return_value=labels):
            response = await self.client.get("/shots/L21_V001", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["labelRevision"], 4)
        self.assertEqual(response.json()["shots"][1]["label"]["reason"], "presenter")

    async def test_owner_can_review_a_suggested_shot(self):
        saved = {"revision": 5, "labels": {"1": {
            "state": "excluded", "reason": "outro", "reviewed_by": OWNER,
        }}}
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9], [10, 24])), \
                patch("backend.app.shot_labels.update", return_value=saved) as update:
            response = await self.client.patch(
                "/shots/L21_V001/labels", headers=self.headers,
                json={"expected_revision": 4, "updates": [{
                    "shot_index": 1, "state": "excluded", "reason": "outro",
                }]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["labelRevision"], 5)
        self.assertEqual(update.call_args.args[2], 4)
        self.assertEqual(update.call_args.args[4], OWNER)

    async def test_teammate_cannot_change_shot_labels(self):
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9])):
            response = await self.client.patch(
                "/shots/L21_V001/labels", headers={"X-AIC-User": "b@x.com"},
                json={"expected_revision": 0, "updates": [{
                    "shot_index": 0, "state": "excluded", "reason": "intro",
                }]},
            )

        self.assertEqual(response.status_code, 403)

    async def test_shots_endpoint_assigns_indexed_keyframes_to_their_shots(self):
        sources = [{"name": "l2026", "label": "L 2026"}]
        with patch("backend.app.frame_maps.load", return_value=mapping(count=30)), \
                patch("backend.app.shots.load", return_value=index([0, 9], [10, 24])), \
                patch("backend.app.core.list_frame_sources", return_value=sources), \
                patch("backend.app.core.list_profiles", return_value=[{"name": "l2026"}]), \
                patch("backend.app.core.default_profile", return_value="l2026"), \
                patch("backend.app.core.frame_ids", return_value=np.array([-1, 0, 9, 10, 24, 25, 30])):
            response = await self.client.get(
                "/shots/L21_V001?keyframe_source=l2026", headers=self.headers
            )

        body = response.json()
        self.assertEqual(body["keyframeSource"], "l2026")
        self.assertEqual(body["keyframeSources"], sources)
        self.assertEqual(
            [[keyframe["frameId"] for keyframe in shot["keyframes"]] for shot in body["shots"]],
            [[0, 9], [10, 24]],
        )
        self.assertEqual(body["shots"][1]["keyframes"][0]["frame"], "/frame/L21_V001/10")

    async def test_shots_endpoint_rejects_an_unknown_keyframe_source(self):
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9])):
            response = await self.client.get(
                "/shots/L21_V001?keyframe_source=missing", headers=self.headers
            )

        self.assertEqual(response.status_code, 400)

    async def test_shots_endpoint_is_unavailable_without_boundaries(self):
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", side_effect=FileNotFoundError):
            response = await self.client.get("/shots/L21_V001", headers=self.headers)

        self.assertEqual(response.status_code, 503)

    async def test_clip_download_names_the_file_after_the_shot(self):
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9], [10, 24])), \
                patch("backend.app.shots.export") as export:
            response = await self.client.get("/clip/L21_V001/1", headers=self.headers)

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'filename="L21_V001_shot0001_f000010-000024.mp4"',
            response.headers["content-disposition"],
        )
        self.assertAlmostEqual(export.call_args.args[1], 9.5 * 3003 / 90000)
        self.assertAlmostEqual(export.call_args.args[2], 24.5 * 3003 / 90000)

    async def test_clip_rejects_a_shot_index_past_the_end(self):
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9])):
            response = await self.client.get("/clip/L21_V001/7", headers=self.headers)

        self.assertEqual(response.status_code, 422)

    async def test_clip_refuses_a_shot_longer_than_the_cap(self):
        long_shot = index([0, 999])
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=long_shot), \
                patch("backend.app.shots.MAX_CLIP_SECONDS", 5.0):
            response = await self.client.get("/clip/L21_V001/0", headers=self.headers)

        self.assertEqual(response.status_code, 422)

    async def test_clip_cleans_up_when_ffmpeg_fails(self):
        error = subprocess.CalledProcessError(1, "ffmpeg")
        with patch("backend.app.frame_maps.load", return_value=mapping()), \
                patch("backend.app.shots.load", return_value=index([0, 9])), \
                patch("backend.app.shots.export", side_effect=error), \
                patch("backend.app.os.unlink") as unlink:
            response = await self.client.get("/clip/L21_V001/0", headers=self.headers)

        self.assertEqual(response.status_code, 500)
        unlink.assert_called_once()


if __name__ == "__main__":
    unittest.main()
