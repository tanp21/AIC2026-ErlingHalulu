import unittest
from unittest.mock import Mock

from scripts.captioning.caption_shots_gemini import (
    DESCRIPTION_PROMPT, EXPERIMENT_SCHEMA, GEMINI_CAPTION_SCHEMA, TEXT_PROMPT,
    PILOT_SHOTS, broadcast_overlay_leaks, clean_ocr, image_parts, make_request,
    parse_combined_response, request_json, temporal_action_failures,
    parse_batch_response, sample_positions, select_keyframes, validate_required_text, video_sampling_fps,
    verify_batch_access,
)
from scripts.captioning.caption_shots import ShotRecord


class GeminiShotCaptionTests(unittest.TestCase):
    def test_prompt_excludes_broadcast_overlays(self):
        folded = (DESCRIPTION_PROMPT + TEXT_PROMPT).casefold()
        for text in ("scrolling red text", "60 giay", "tuoitretv", "htv"):
            self.assertIn(text.casefold(), folded)
        for text in ("every visible person", "spatial order", "chronological action",
                     "every distinct readable", "never collapse values",
                     "merge uninterrupted repetition", "only the verbatim visible text"):
            self.assertIn(text, folded)
        self.assertNotIn("short physical location", folded)
        for overfit in ("robotic arm", "hand crank", "folding/unfolding"):
            self.assertNotIn(overfit, folded)

    def test_pilot_has_all_requested_shots(self):
        self.assertEqual(len(PILOT_SHOTS), 10)
        self.assertEqual(len(set(PILOT_SHOTS)), 10)
        self.assertIn(("K07_V025", 225), PILOT_SHOTS)
        self.assertIn(("K07_V025", 226), PILOT_SHOTS)
        self.assertIn(("K07_V025", 227), PILOT_SHOTS)
        self.assertIn(("L23_V013", 20), PILOT_SHOTS)
        self.assertNotIn(("L23_V013", 21), PILOT_SHOTS)

    def test_dense_sampling_includes_boundaries(self):
        positions = sample_positions(22166, 22223, 12)
        self.assertEqual((positions[0], positions[-1]), (22166, 22223))
        self.assertEqual(len(positions), 12)

    def test_required_text_gate_is_accent_insensitive(self):
        validate_required_text([{
            "video_id": "L23_V013", "shot_index": 20,
            "ocr": "Da Nang - Tam Ky (Quang Nam); TON DONG A",
        }])

    def test_required_text_gate_rejects_missing_scene_text(self):
        with self.assertRaisesRegex(RuntimeError, "Quảng Nam"):
            validate_required_text([{
                "video_id": "L23_V013", "shot_index": 20,
                "ocr": "Tam Ky; TON DONG A",
            }])

    def test_gemini_schema_has_separate_ocr_field(self):
        self.assertEqual(GEMINI_CAPTION_SCHEMA.names.count("ocr"), 1)
        self.assertIn("caption", GEMINI_CAPTION_SCHEMA.names)
        self.assertIn("input_tokens", EXPERIMENT_SCHEMA.names)
        self.assertIn("strategy", EXPERIMENT_SCHEMA.names)

    def test_video_sampling_targets_twelve_frames_and_caps_at_24_fps(self):
        normal = ShotRecord("2026", "X", "X_V001", 1, 0, 89, 30, "/x.mp4")
        short = ShotRecord("2026", "X", "X_V001", 2, 0, 5, 30, "/x.mp4")
        self.assertAlmostEqual(video_sampling_fps(normal), 4.0)
        self.assertEqual(video_sampling_fps(short), 24.0)

    def test_batch_request_serializes_as_generate_content(self):
        request = make_request(
            image_parts([b"jpeg"]), "prompt",
            resolution="MEDIA_RESOLUTION_HIGH", max_output_tokens=10,
            metadata={"pass": "description"},
        )
        value = request_json(request)
        self.assertEqual(value["contents"][0]["role"], "user")
        self.assertEqual(value["contents"][0]["parts"][-1]["text"], "prompt")
        self.assertIn("generation_config", value)
        self.assertNotIn("config", value)
        self.assertNotIn("metadata", value)

    def test_batch_parser_ignores_new_usage_metadata(self):
        result = parse_batch_response({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "OK"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 1,
                "serviceTier": "SERVICE_TIER_STANDARD",
            },
        })
        self.assertEqual(result["text"], "OK")
        self.assertEqual(result["input_tokens"], 3)
        self.assertFalse(result["truncated"])

    def test_batch_parser_can_record_truncation_for_a_benchmark(self):
        result = parse_batch_response({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": "MAX_TOKENS",
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
        }, allow_truncated=True)
        self.assertTrue(result["truncated"])

    def test_clean_ocr_removes_broadcast_items_only(self):
        self.assertEqual(
            clean_ocr("DÍCH; HTV; 60 giây; 06:50:11; top of a physical sign; Quảng Nam"),
            "DÍCH; top of a physical sign; Quảng Nam",
        )

    def test_combined_response_keeps_ocr_independent(self):
        caption, ocr = parse_combined_response(
            '{"caption":"Detailed visual description: A pillar.",'
            '"ocr":"HỎA HỒNG; HTV"}'
        )
        self.assertEqual(caption, "Detailed visual description: A pillar.")
        self.assertEqual(ocr, "HỎA HỒNG")

    def test_temporal_gate_requires_each_action_and_object_group(self):
        complete = [{
            "video_id": "K07_V025", "shot_index": 226,
            "caption": "A worker begins to turn the hand crank.",
        }]
        incomplete = [{
            "video_id": "K07_V025", "shot_index": 226,
            "caption": "A worker inspects the dashboard.",
        }]
        self.assertEqual(temporal_action_failures(complete), [])
        self.assertEqual(temporal_action_failures(incomplete), ["K07_V025/226"])

    def test_overlay_gate_checks_full_caption_not_only_ocr(self):
        self.assertEqual(broadcast_overlay_leaks([{
            "video_id": "L23_V013", "shot_index": 20,
            "caption": "Finish arch branded HTV. Visible scene text: ĐÍCH",
            "ocr": "ĐÍCH",
        }]), ["L23_V013/20"])

    def test_batch_preflight_explains_paid_tier_requirement(self):
        client = Mock()
        client.batches.create.side_effect = RuntimeError("400 FAILED_PRECONDITION")
        with self.assertRaisesRegex(RuntimeError, "paid billing tier"):
            verify_batch_access(client, "gemini-3.1-flash-lite")

    def test_batch_preflight_explains_denied_project(self):
        client = Mock()
        client.batches.create.side_effect = RuntimeError("403 PERMISSION_DENIED")
        with self.assertRaisesRegex(RuntimeError, "denied this project access"):
            verify_batch_access(client, "gemini-3.1-flash-lite")

    def test_keyframes_are_uniformly_bounded(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            record = ShotRecord("2026", "L27", "L27_V010", 83, 10, 19, 25, "/x.mp4")
            frames = root / "2026/keyframes/L27/L27_V010"
            frames.mkdir(parents=True)
            for index in range(5, 25):
                (frames / f"{index:06d}.png").touch()
            selected = select_keyframes(record, root, limit=3)
            self.assertEqual([int(path.stem) for path in selected], [10, 14, 19])


if __name__ == "__main__":
    unittest.main()
