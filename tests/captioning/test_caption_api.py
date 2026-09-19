import unittest

from scripts.captioning import caption_api, windows


def unit(duration=3.13, fps=25.0, frame_count=8):
    return windows.CaptionUnit(
        "2025", "K17", "K17_V003", 102, 0, int(duration * fps) - 1, fps,
        "/video.mp4", 0, 1, 372.59, 372.59 + duration, frame_count,
    )


class RequestBodyTests(unittest.TestCase):
    """The "1024-default-8frames" run is the quality baseline the user picked (it reads
    "Mộc Châu" correctly on K17_V003 shot 102) -- these assertions exist so a refactor
    cannot silently drift the request away from that exact recipe again."""

    def test_defaults_are_the_tuned_production_recipe(self):
        args = caption_api.parse_args(["--dry-run"])

        # Tuned on the 137-unit pilot against 25 fidelity checks on known on-screen
        # text: 0.8/0.9/40 read 21/25 on every seed, versus 17-22 for the checkpoint's
        # own generation_config.json (0.7/0.8/20).
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.top_p, 0.9)
        self.assertEqual(args.top_k, 40)
        # The baseline ran max_tokens 1024, but a third of the pilot captions hit that
        # cap; 2048 lets them finish.
        self.assertEqual(args.max_tokens, 2048)
        self.assertEqual(args.frame_width, 0)  # native resolution, do_resize=False

    def test_top_k_is_a_top_level_field_not_wrapped_in_extra_body(self):
        # A raw httpx POST is not the OpenAI SDK: "extra_body" is meaningless to vLLM
        # here and would silently drop top_k, leaving generation unbounded.
        args = caption_api.parse_args(["--dry-run"])

        body = caption_api.build_request(unit(), [b"\xff\xd8\xff\xd9"] * 8, args)

        self.assertEqual(body["top_k"], 40)
        self.assertNotIn("extra_body", body)

    def test_request_uses_the_baseline_prompt_verbatim(self):
        args = caption_api.parse_args(["--dry-run"])

        body = caption_api.build_request(unit(), [b"\xff\xd8\xff\xd9"] * 8, args)

        text_parts = [part["text"] for part in body["messages"][0]["content"]
                     if part["type"] == "text"]
        self.assertEqual(text_parts, [caption_api.RECIPE_PROMPT])

    def test_recipe_prompt_hash_is_the_baseline_artifact_hash(self):
        # artifacts/captions/1024-default-8frames/captions.parquet records this hash;
        # any edit to RECIPE_PROMPT that is not re-validated against that artifact fails.
        self.assertEqual(caption_api.RECIPE_PROMPT_SHA256,
                         caption_api.BASELINE_PROMPT_SHA256)

    def test_request_has_no_repetition_penalty_key(self):
        # The baseline call never set repetition_penalty either; sending vLLM's own
        # default (1.0) is achieved by omitting the field, not by pinning it to 1.0 --
        # pinning it would only coincidentally match if vLLM's default ever changes.
        args = caption_api.parse_args(["--dry-run"])

        body = caption_api.build_request(unit(), [b"\xff\xd8\xff\xd9"] * 8, args)

        self.assertNotIn("repetition_penalty", body)

    def test_repetition_penalty_is_top_level_when_changed(self):
        args = caption_api.parse_args(["--dry-run", "--repetition-penalty", "1.05"])

        body = caption_api.build_request(unit(), [b"\xff\xd8\xff\xd9"] * 8, args)

        self.assertEqual(body["repetition_penalty"], 1.05)
        self.assertNotIn("extra_body", body)


class RowTests(unittest.TestCase):
    def test_row_stores_the_baseline_style_single_caption_field(self):
        args = caption_api.parse_args(["--dry-run"])
        payload = {
            "choices": [{"message": {"content": " [00:01] A man walks. "},
                        "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4246, "completion_tokens": 914,
                     "total_tokens": 5160},
        }

        row = caption_api.build_row(unit(), [b"\xff\xd8\xff\xd9"] * 8, payload, 37.6, args)

        self.assertEqual(row["caption"], "A man walks.")
        self.assertEqual(row["summary"], "")
        self.assertEqual(row["onscreen_text"], "")
        self.assertEqual(row["raw_output"], "[00:01] A man walks.")


if __name__ == "__main__":
    unittest.main()
