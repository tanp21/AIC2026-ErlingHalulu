import unittest

import numpy as np

from scripts.ocr.alt_engines import _mask_ignore_regions


class MaskIgnoreRegionsTests(unittest.TestCase):
    def test_no_regions_returns_the_same_pixels(self):
        image = np.full((10, 10, 3), 255, dtype="uint8")
        masked = _mask_ignore_regions(image, None)
        np.testing.assert_array_equal(masked, image)

    def test_does_not_mutate_the_input(self):
        image = np.full((10, 10, 3), 255, dtype="uint8")
        _mask_ignore_regions(image, [[0.0, 0.0, 1.0, 1.0]])
        self.assertTrue((image == 255).all())

    def test_paints_the_normalised_box_black(self):
        image = np.full((10, 10, 3), 255, dtype="uint8")
        masked = _mask_ignore_regions(image, [[0.0, 0.0, 0.5, 0.5]])
        self.assertTrue((masked[0:5, 0:5] == 0).all())
        self.assertTrue((masked[5:, 5:] == 255).all())

    def test_multiple_regions_are_all_masked(self):
        image = np.full((10, 10, 3), 255, dtype="uint8")
        masked = _mask_ignore_regions(image, [[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]])
        self.assertTrue((masked[0:2, 0:2] == 0).all())
        self.assertTrue((masked[8:, 8:] == 0).all())
        self.assertTrue((masked[4:6, 4:6] == 255).all())


if __name__ == "__main__":
    unittest.main()
