import math
import unittest

import numpy as np

from backend.dres_payload import (
    build_kis, build_qa, build_trake, frame_timestamp_ms, normalize_answer,
    validate_kis_payload, video_id_from_source,
)
from backend.frame_maps import FrameMap


class PayloadTests(unittest.TestCase):
    def setUp(self):
        # Original video starts at nonzero PTS; the selected frame is 123456 ms
        # into that video. Shot metadata is deliberately unrelated.
        self.mapping = FrameMap(np.array([90000, 210000, 213456, 216000]), 1, 1000)

    def test_kis_exact_selected_frame_not_shot_boundary(self):
        shot_start_ms, shot_end_ms = 120000, 126000
        selected_frame_ms = frame_timestamp_ms(self.mapping, 2)
        self.assertEqual(selected_frame_ms, 123456)
        self.assertNotEqual(selected_frame_ms, shot_start_ms)
        self.assertNotEqual(selected_frame_ms, shot_end_ms)
        self.assertEqual(build_kis("L01_V001", selected_frame_ms), {
            "answerSets": [{"answers": [{"mediaItemName": "L01_V001", "start": "123456", "end": "123456"}]}],
        })

    def test_every_valid_kis_has_equal_start_end_selected_frame(self):
        for frame in range(len(self.mapping.pts)):
            ms = frame_timestamp_ms(self.mapping, frame)
            answer = build_kis("L01_V001", ms)["answerSets"][0]["answers"][0]
            self.assertEqual(answer["start"], answer["end"])
            self.assertEqual(answer["start"], str(ms))

    def test_kis_rejects_interval(self):
        payload = build_kis("L01_V001", 123456)
        payload["answerSets"][0]["answers"][0]["end"] = "126000"
        with self.assertRaisesRegex(ValueError, "must equal"):
            validate_kis_payload(payload, 123456)

    def test_frame_time_uses_pts_and_millisecond_precision(self):
        mapping = FrameMap(np.array([45000, 45001, 45002, 45003]), 1, 3)
        self.assertEqual([frame_timestamp_ms(mapping, i) for i in range(4)], [0, 333, 667, 1000])
        for invalid in (-1, 4, None, True, math.nan, math.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                frame_timestamp_ms(mapping, invalid)

    def test_missing_invalid_video_and_time(self):
        for video in ("", "L01_V001.mp4", "shot_12"):
            with self.subTest(video=video), self.assertRaises(ValueError):
                build_kis(video, 0)
        for time in (-1, None, math.nan, math.inf, True):
            with self.subTest(time=time), self.assertRaises(ValueError):
                build_kis("L01_V001", time)

    def test_video_name_removes_only_actual_extension(self):
        for path, expected in (
            ("L01_V001.mp4", "L01_V001"),
            ("/data/2026/videos/L01/L01_V001.MP4", "L01_V001"),
            ("N001_V001.mov", "N001_V001"),
            ("/data/2026/videos/N001/N001_V001.MOV", "N001_V001"),
        ):
            self.assertEqual(video_id_from_source(path, expected), expected)
        for path, expected in (("L01_V001.mp4.mp4", "L01_V001"), ("L01_V001.mp4", "L01"), ("L01_V001.mkv", "L01_V001")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                video_id_from_source(path, expected)

    def test_qa_answer_is_normalized_to_uppercase_ascii_without_spaces(self):
        self.assertEqual(build_qa("example answer", "L01_V001", 123456), {
            "answerSets": [{"answers": [{"text": "QA-EXAMPLEANSWER-L01_V001-123456"}]}],
        })
        # A Vietnamese answer keeps its words and loses only its diacritics: `đ` has no canonical
        # decomposition, so it is folded on its own.
        for answer, expected in (("Năm người", "NAMNGUOI"), ("Đội bóng đá", "DOIBONGDA"),
                                 ("  A-B  ", "A-B"), ("Có 3 người, gồm nam và nữ", "CO3NGUOI,GOMNAMVANU"),
                                 ("5", "5")):
            with self.subTest(answer=answer):
                self.assertEqual(normalize_answer(answer), expected)
                self.assertEqual(build_qa(answer, "L01_V001", 0)["answerSets"][0]["answers"][0]["text"],
                                 f"QA-{expected}-L01_V001-0")
        for answer in ("", "   ", None):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                build_qa(answer, "L01_V001", 0)
        self.assertEqual(build_qa("answer", "N001_V001", 123456)["answerSets"][0]["answers"][0]["text"],
                         "QA-ANSWER-N001_V001-123456")

    def test_trake_uses_ordered_official_frame_ids(self):
        self.assertEqual(build_trake("L01_V001", [100, 250, 400], 3), {
            "answerSets": [{"answers": [{"text": "TR-L01_V001-100,250,400"}]}],
        })
        self.assertEqual(build_trake("L01_V001", [400, 100])["answerSets"][0]["answers"][0]["text"],
                         "TR-L01_V001-400,100")
        self.assertEqual(build_trake("L01_V001", [400, 100], 2)["answerSets"][0]["answers"][0]["text"], "TR-L01_V001-400,100")
        self.assertEqual(build_trake("L01_V001", [100], 1)["answerSets"][0]["answers"][0]["text"], "TR-L01_V001-100")
        for frames, count in (([], 1), ([1, 1], 2), ([-1], 1), ([None], 1), ([1, 2], 3)):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                build_trake("L01_V001", frames, count)
        self.assertEqual(build_trake("N001_V001", [100], 1)["answerSets"][0]["answers"][0]["text"],
                         "TR-N001_V001-100")
