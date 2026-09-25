"""The keyframe archives land at the frame the N index would place them at."""
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.indexing.build_keyframe_thumbnails import build


class BuildKeyframeThumbnailsTests(unittest.TestCase):
    #: 2 s apart at a nominal 25 fps is 50 frames, but these videos are VFR: the frames that
    #: carry 0 s and 2 s are 0 and 1, which is exactly what the index resolved them to.
    TIMES = [0, 2, 6, 10, 14]

    def _fixture(self, root):
        base = root / "indexes"
        maps = base / "frame_maps"
        maps.mkdir(parents=True)
        np.savez_compressed(maps / "N001_V001.npz",
                            pts=np.asarray(self.TIMES, dtype=np.int64) * 1000,
                            time_base_num=1, time_base_den=1000)
        points = root / "points.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"video_id": "N001_V001", "group": "N001", "keyframe_n": 1, "frame_idx": 0, "pts_ms": 0},
            {"video_id": "N001_V001", "group": "N001", "keyframe_n": 2, "frame_idx": 50, "pts_ms": 2000},
        ]), points)
        archive = root / "Keyframes_N001-N010.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for ordinal in (1, 2):
                buffer = io.BytesIO()
                Image.new("RGB", (1920, 1080), (ordinal * 40, 0, 0)).save(buffer, format="JPEG")
                bundle.writestr(f"keyframes/N001-V001/{ordinal:03d}.jpg", buffer.getvalue())
        return base, points, archive

    def test_writes_every_width_at_the_resolved_frame(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, points, archive = self._fixture(root)
            out = root / "thumbs"

            result = build([archive], out, points, base, widths=(160, 320), workers=2)

            self.assertEqual(result["written"], 2)
            # ordinal 2 is 2 s in, and 2 s is frame 1 of this video, not frame 50.
            for width, expected in ((160, 160), (320, 320)):
                for frame in (0, 1):
                    path = out / str(width) / "N001" / "N001_V001" / f"{frame}.jpg"
                    self.assertTrue(path.exists(), path)
                    with Image.open(path) as image:
                        self.assertEqual(image.width, expected)

    def test_a_second_run_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, points, archive = self._fixture(root)
            out = root / "thumbs"
            build([archive], out, points, base, widths=(320,), workers=2)

            result = build([archive], out, points, base, widths=(320,), workers=2)

            self.assertEqual(result["written"], 0)
            self.assertEqual(result["skipped"], 2)


if __name__ == "__main__":
    unittest.main()
