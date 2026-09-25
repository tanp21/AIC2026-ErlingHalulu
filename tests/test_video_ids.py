"""One video-id format, shared by every route, ZIP parser and label document.

The regression this pins: the pattern was re-typed per module and most copies said ``\\d{2}``,
so every media request for an N (traffic-camera) video -- ``/thumb``, ``/video``, ``/shots``,
``/frame-map`` -- answered 422 while the index behind it was correct: a keyframe that never
loads and a player that never opens.
"""
import re
import unittest
from pathlib import Path

from backend import video_ids


class VideoIdFormatTests(unittest.TestCase):
    def test_accepts_two_and_three_digit_groups(self):
        for video_id in ("L21_V001", "S01_V012", "M10_V031", "N029_V002", "N100_V003"):
            with self.subTest(video_id=video_id):
                self.assertIsNotNone(video_ids.VIDEO_ID.fullmatch(video_id))

    def test_accepts_the_hyphen_the_organisers_write(self):
        # Detector exports and map-keyframes name the same videos N029-V002; dres_payload
        # normalises it, so the format has to let it through validation.
        self.assertIsNotNone(video_ids.VIDEO_ID.fullmatch("N029-V002"))

    def test_rejects_anything_else(self):
        for video_id in ("n029_V002", "N029V002", "N029_V02", "L21_V0001", "N029_V002_x", ""):
            with self.subTest(video_id=video_id):
                self.assertIsNone(video_ids.VIDEO_ID.fullmatch(video_id))


class PatternSharingTests(unittest.TestCase):
    BACKEND = Path(video_ids.__file__).resolve().parent
    #: The agent is a client of the server but reads the same ids, from models and reviewers.
    AGENT = BACKEND.parent / "agent"

    #: What a re-typed copy looks like in source (the shapes that were in the tree).
    LITERALS = ("\\d{2}[_-]V", "\\d{2}_V", "\\d{2,3}[_-]V", "\\d{2,3}_V")

    def test_every_backend_module_uses_the_shared_constant(self):
        # A narrower literal anywhere is the bug: it rejects N ids at that one door.
        offenders = []
        for path in sorted([*self.BACKEND.glob("*.py"), *self.AGENT.glob("*.py")]):
            if path.name == "video_ids.py":
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if any(literal in line for literal in self.LITERALS):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [], "re-typed video-id pattern:\n" + "\n".join(offenders))

    def test_every_route_that_takes_a_video_id_uses_the_shared_pattern(self):
        from backend import app as server

        patterns = {}
        for route in server.app.routes:
            dependant = getattr(route, "dependant", None)
            if dependant is None:
                continue
            for parameter in dependant.path_params:
                pattern = self._pattern(parameter)
                if pattern and "V\\d{3}" in pattern:
                    patterns.setdefault(pattern, []).append(route.path)
        self.assertTrue(patterns, "no route takes a video id")
        self.assertEqual(list(patterns), [video_ids.VIDEO_ID_PATTERN],
                         f"routes disagree on the video-id pattern: {patterns}")

    @staticmethod
    def _pattern(parameter):
        # Pydantic v2 keeps a path parameter's constraint in the field's metadata.
        for item in getattr(parameter.field_info, "metadata", ()):
            pattern = getattr(item, "pattern", None)
            if pattern:
                return pattern
        return getattr(parameter.field_info, "pattern", None)


if __name__ == "__main__":
    unittest.main()
