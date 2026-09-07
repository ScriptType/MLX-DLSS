import unittest
import importlib.util
import numpy as np

from mlxdlss.motion_quality import assess_motion


@unittest.skipUnless(importlib.util.find_spec("cv2"), "video extra is required")
class MotionQualityTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(1).random((64, 80, 3), dtype=np.float32)
        self.zero = np.zeros((64, 80, 2), np.float32)

    def test_stationary_correspondence_is_fully_trusted(self):
        result = assess_motion(self.frame, self.frame, self.zero, self.zero, scene_cut_threshold=0.3)
        np.testing.assert_array_equal(result.confidence, 1)
        self.assertFalse(result.reset)

    def test_out_of_bounds_and_inconsistent_flow_are_rejected(self):
        for b, f in ((np.ones_like(self.zero) * 2, self.zero), (np.ones_like(self.zero) * 0.1, self.zero)):
            result = assess_motion(self.frame, self.frame, b, f, scene_cut_threshold=0)
            np.testing.assert_array_equal(result.confidence, 0)
            self.assertFalse(result.reset)

    def test_pan_is_not_a_cut_but_equal_histogram_scene_change_is(self):
        current = np.roll(self.frame, 4, axis=1)
        b = self.zero.copy(); b[..., 0] = -4 / 80
        result = assess_motion(current, self.frame, b, -b, scene_cut_threshold=0.3)
        self.assertFalse(result.reset)
        np.testing.assert_array_equal(result.confidence[:, 8:-8], 1)
        unrelated = self.frame[::-1].copy()
        cut = assess_motion(unrelated, self.frame, self.zero, self.zero, scene_cut_threshold=0.3)
        self.assertTrue(cut.reset)
        disabled = assess_motion(unrelated, self.frame, self.zero, self.zero, scene_cut_threshold=0)
        self.assertFalse(disabled.reset)

    def test_new_foreground_is_rejected_without_resetting_background(self):
        current = self.frame.copy(); current[20:40, 30:50] = 1
        result = assess_motion(current, self.frame, self.zero, self.zero, scene_cut_threshold=0.3)
        self.assertFalse(result.reset)
        np.testing.assert_array_equal(result.confidence[24:36, 34:46], 0)
        np.testing.assert_array_equal(result.confidence[:10, :10], 1)

    def test_nonfinite_and_mismatched_maps_are_errors(self):
        for b in (np.full_like(self.zero, np.nan), self.zero[..., :1]):
            with self.assertRaises(ValueError):
                assess_motion(self.frame, self.frame, b, self.zero, scene_cut_threshold=0.3)

    def test_high_contrast_pan_with_valid_correspondence_is_not_cut(self):
        previous = np.repeat((np.indices((64, 80)).sum(0) % 2)[..., None], 3, axis=-1).astype(np.float32)
        current = np.roll(previous, 1, axis=1)
        b = self.zero.copy(); b[..., 0] = -1 / 80
        result = assess_motion(current, previous, b, -b, scene_cut_threshold=0.3)
        self.assertFalse(result.reset)
