"""Polars object data alignment and position query checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import HTTPException

from backend import app, object_search
from backend.access_control import AccessConfig, LatestWinsCoordinator
from scripts.indexing import build_object_index


class ObjectIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = self.root / "current"
        (self.index / "labels").mkdir(parents=True)
        playback = self.index / "playback" / "video_id=L21_V001"
        playback.mkdir(parents=True)
        (self.index / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "labels": ["car", "person"],
            "profiles": {"l2025": {"mask": 1, "keyframes": 1},
                         "l2026-v2": {"mask": 2, "keyframes": 2}},
            "alignment_max_seconds": .25,
        }))
        pq.write_table(pa.Table.from_pylist([
            {"id": 1, "video_id": "L21_V001", "frame_idx": 20, "time": 2., "pts": 20,
             "source_frame_idx": 18, "source_time": 1.9, "delta": .1, "profile_mask": 3},
            {"id": 2, "video_id": "L21_V001", "frame_idx": 40, "time": 4., "pts": 40,
             "source_frame_idx": 39, "source_time": 3.9, "delta": .1, "profile_mask": 2},
        ], schema=build_object_index.FRAME_SCHEMA), self.index / "keyframes.parquet")
        def detection(frame, mask, score, x0, y0, x1, y1, det_idx):
            return {"frame_id": frame, "profile_mask": mask, "score": score,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "det_idx": det_idx}
        pq.write_table(pa.Table.from_pylist([
            detection(1, 3, .7, .4, .6, .7, .9, 2),
        ], schema=build_object_index.DETECTION_SCHEMA), self.index / "labels" / "000.parquet")
        pq.write_table(pa.Table.from_pylist([
            detection(1, 3, .9, .1, .2, .3, .6, 0),
            detection(1, 3, .8, .7, .2, .9, .6, 1),
            detection(2, 2, .95, .7, .2, .9, .6, 0),
        ], schema=build_object_index.DETECTION_SCHEMA), self.index / "labels" / "001.parquet")
        pq.write_table(pa.Table.from_pylist([
            {"video_id": "L21_V001", "frame_idx": 18, "time": 1.9,
             "labels": ["person"], "scores": [.9], "boxes": [[.1, .2, .3, .6]]},
            {"video_id": "L21_V001", "frame_idx": 19, "time": 2.,
             "labels": [], "scores": [], "boxes": []},
        ], schema=build_object_index.PLAYBACK_SCHEMA), playback / "part.parquet")
        object_search._load_store.cache_clear()
        self.addCleanup(object_search._load_store.cache_clear)
        self.patch = patch.object(object_search, "DATA_DIR", self.index)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_position_count_and_profile(self):
        query = object_search.ObjectQuery(clauses=[
            object_search.ObjectClause(label="person", region=(0, 0, 1, 1), min_count=2),
            object_search.ObjectClause(label="car", region=(0.4, 0.5, 0.8, 1)),
        ])
        result = object_search.search("l2026-v2", query, 10)
        self.assertEqual([(hit["video_id"], hit["frame_idx"]) for hit in result["hits"]],
                         [("L21_V001", 20)])
        self.assertEqual(result["hits"][0]["score"], 0.7)
        self.assertEqual(len(result["hits"][0]["matched_objects"]), 3)
        self.assertEqual(object_search.match_frames("l2025", query, [
            {"video_id": "L21_V001", "frame_idx": 20},
            {"video_id": "L21_V001", "frame_idx": 40},
        ])[1], None)

    def test_same_detection_cannot_satisfy_two_positions(self):
        query = object_search.ObjectQuery(clauses=[
            object_search.ObjectClause(label="person", region=(0.6, 0, 1, 1)),
            object_search.ObjectClause(label="person", region=(0.6, 0, 1, 1)),
        ])
        self.assertEqual(object_search.search("l2026-v2", query, 10)["hits"], [])

    def test_single_object_minimum_count_uses_distinct_instances(self):
        query = object_search.ObjectQuery(clauses=[
            object_search.ObjectClause(label="person", region=(0, 0, 1, 1), min_count=2),
        ])
        hits = object_search.search("l2026-v2", query, 10)["hits"]
        self.assertEqual([hit["frame_idx"] for hit in hits], [20])
        self.assertEqual(hits[0]["score"], 0.8)
        self.assertEqual(len(hits[0]["matched_objects"]), 2)

    def test_video_detections_include_full_parquet_samples(self):
        response = object_search.video_detections("L21_V001")
        self.assertEqual([frame["frame_idx"] for frame in response["frames"]], [18, 19])
        self.assertEqual(response["frames"][0]["detections"][0]["label"], "person")
        self.assertEqual(response["frames"][0]["detections"][0]["box"], [0.2, 0.1, 0.6, 0.3])
        self.assertEqual(response["frames"][1]["detections"], [])

    def test_builder_aligns_by_time_and_normalizes_video_id(self):
        profiles = self.root / "profiles"
        profile = profiles / "demo"
        profile.mkdir(parents=True)
        (profiles / "manifest.json").write_text(json.dumps({
            "profiles": {"demo": {"directory": "demo"}}, "default": "demo",
        }))
        pq.write_table(pa.Table.from_pylist([{"video_idx": 0, "video_id": "S01_V001"}]),
                       profile / "video_meta.parquet")
        np.save(profile / "id_video.npy", np.array([0], dtype=np.int32))
        np.save(profile / "id_frame.npy", np.array([2], dtype=np.int32))
        maps = self.root / "maps"
        maps.mkdir()
        np.savez(maps / "S01_V001.npz", pts=np.arange(10, dtype=np.int64),
                 time_base_num=1, time_base_den=10)
        source = self.root / "S.parquet"
        pq.write_table(pa.Table.from_pylist([{
            "video_id": "S01-V001", "frame_idx": 1, "time": 0.1,
            "labels": ["person"], "scores": [0.9], "boxes": [[0.1, 0.2, 0.3, 0.6]],
        }]), source)
        output = self.root / "built"
        stats = build_object_index.build([("S", source)], profiles, maps, output)
        self.assertEqual(stats["keyframes"], 1)
        self.assertEqual((stats["parquet_frames"], stats["parquet_detections"]), (1, 1))
        built = output / "current"
        self.assertEqual(pl.read_parquet(built / "keyframes.parquet")["source_frame_idx"].to_list(), [1])
        self.assertEqual(pl.read_parquet(built / "labels" / "000.parquet")["cx"].to_list(), [0.4])
        self.assertTrue((built / "playback" / "video_id=S01_V001").is_dir())
        object_search._load_store.cache_clear()
        with patch.object(object_search, "DATA_DIR", built):
            query = object_search.ObjectQuery(clauses=[
                object_search.ObjectClause(label="person", region=(0, 0, 1, 1)),
            ])
            self.assertEqual(object_search.search("demo", query, 10)["hits"][0]["frame_idx"], 2)


class ObjectSearchApiTests(unittest.TestCase):
    def test_standalone_and_filter(self):
        query = object_search.ObjectQuery(clauses=[
            object_search.ObjectClause(label="person", region=(0, 0, 1, 1)),
        ])
        hit = {"video_id": "L21_V001", "frame_idx": 20, "pts": 20, "time": 2.0,
               "score": 0.9, "matched_objects": [{"label": "person", "confidence": 0.9,
                                                   "box": [0.1, 0.2, 0.3, 0.6]}], "delta_ms": 100}
        with patch.object(app.search_core, "search_objects", return_value={"hits": [hit], "truncated": False}):
            response = app.search(app.SearchReq(mode="object", object_query=query))
        self.assertEqual(response["results"][0]["source"], "object")
        self.assertEqual(response["results"][0]["frame_idx"], 20)
        primary = {"results": [{"video_id": "L21_V001", "frame_idx": 20,
                                "score": 0.5, "source": "visual", "scores": {}}],
                   "latency_ms": 1.0, "mode": "visual"}
        with patch.object(app, "_search_without_objects", return_value=primary), \
             patch.object(app.search_core, "match_objects", return_value=[{
                 "score": 0.9, "matched_objects": hit["matched_objects"], "delta_ms": 100,
             }]):
            filtered = app.search(app.SearchReq(mode="visual", visual_text="person", object_query=query))
        self.assertEqual(filtered["results"][0]["score"], 0.5)
        self.assertEqual(filtered["results"][0]["scores"]["object_score"], 0.9)
        with patch.object(app, "_search_without_objects", return_value=primary), \
             patch.object(app.search_core, "match_objects", side_effect=FileNotFoundError("missing")):
            with self.assertRaises(HTTPException) as raised:
                app.search(app.SearchReq(mode="visual", visual_text="person", object_query=query))
        self.assertEqual(raised.exception.status_code, 503)

    def test_object_events_form_temporal_keyframe_sequence(self):
        query = object_search.ObjectQuery(clauses=[
            object_search.ObjectClause(label="person", region=(0, 0, 1, 1)),
        ])
        def hit(frame, second):
            return {"video_id": "L21_V001", "frame_idx": frame, "pts": frame,
                    "time": second, "score": 0.9, "delta_ms": 0,
                    "matched_objects": [{"label": "person", "confidence": 0.9,
                                         "box": [0.1, 0.1, 0.3, 0.4]}]}
        request = app.TemporalSearchReq(events=["event", "event"], max_gap_seconds=20)
        with patch.object(app.search_core, "search_objects", side_effect=[
            {"hits": [hit(20, 2.0)], "truncated": False},
            {"hits": [hit(50, 5.0)], "truncated": False},
        ]):
            response = app.temporal_search_response(request, 10, "raw_mean", event_specs=[
                {"kind": "object", "query": query}, {"kind": "object", "query": query},
            ])
        self.assertEqual(response["total_sequences"], 1)
        events = response["videos"][0]["sequences"][0]["events"]
        self.assertEqual([event["frame_idx"] for event in events], [20, 50])
        self.assertTrue(all(event["source"] == "object" for event in events))


class ObjectTemporalEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_multipart_object_events_need_no_visual_encoder(self):
        app.app.state.access_config = AccessConfig(
            frozenset({"one@example.com", "two@example.com"}), "one@example.com"
        )
        app.app.state.search_coordinator = LatestWinsCoordinator()
        app.app.state.temporal_snapshots = app.TemporalSnapshotStore(page_size=20)
        query = {"clauses": [{"label": "person", "region": [0, 0, 1, 1]}]}
        async def inline(function, *args):
            return function(*args)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app),
                                     base_url="http://test") as client:
            with patch.object(app, "load_temporal_config", return_value=(10, "raw_mean", 0)), \
                 patch.object(app, "run_in_threadpool", side_effect=inline), \
                 patch.object(app, "temporal_search_response", return_value={
                     "videos": [], "total_sequences": 0, "latency_ms": 1,
                 }) as search, \
                 patch.object(app.search_core, "encode_texts") as encode:
                response = await client.post("/search/temporal/image",
                    headers={"X-AIC-User": "one@example.com"},
                    data={"payload": json.dumps({"events": [
                        {"kind": "object", "object_query": query},
                        {"kind": "object", "object_query": query},
                    ]})})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([event["kind"] for event in search.call_args.args[5]],
                         ["object", "object"])
        encode.assert_not_called()
