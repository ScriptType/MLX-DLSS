import json
import stat
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from Tools import benchmark_video_temporal as benchmark
from mlxdlss.motion_quality import assess_motion
from mlxdlss.temporal import TemporalOptions, TemporalSession, make_temporal_features


def textured_frame(height=64, width=80):
    yy, xx = np.indices((height, width), dtype=np.float32)
    return np.stack(
        (
            0.45 + 0.22 * np.sin(xx * 0.31) + 0.08 * np.cos(yy * 0.17),
            0.48 + 0.18 * np.cos(xx * 0.13 + yy * 0.19),
            0.42 + 0.16 * np.sin(xx * 0.23 - yy * 0.11),
        ),
        axis=-1,
    ).astype(np.float32)


def translate(image, dx, dy):
    matrix = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def constant_motion(frame, dx, dy):
    motion = np.empty((*frame.shape[:2], 2), np.float32)
    motion[..., 0] = -dx / frame.shape[1]
    motion[..., 1] = -dy / frame.shape[0]
    return motion


class CheapPipeline:
    def run_features(self, features):
        head = np.zeros((*features.shape[:2], 4), np.float32)
        head[..., 0] = features[..., 4] * np.float32(3)
        head[..., 1] = features[..., 5] * np.float32(-2)
        head[..., 2] = features[..., 6]
        return head


class KnownMotionCorpusTests(unittest.TestCase):
    def test_integer_and_subpixel_horizontal_and_vertical_pans_reproject_history(self):
        previous = textured_frame()
        height, width = previous.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)

        for dx, dy in ((3, 0), (12, 0), (0.5, 0), (0, -9), (0, 0.5)):
            with self.subTest(dx=dx, dy=dy):
                current = translate(previous, dx, dy)
                features = make_temporal_features(
                    current,
                    previous,
                    constant_motion(previous, dx, dy),
                    frame_index=1,
                )
                reprojected = features[..., 7:10] * np.float32(8) + np.float32(0.5)
                valid = (
                    (xx - dx >= 8)
                    & (xx - dx < width - 8)
                    & (yy - dy >= 8)
                    & (yy - dy < height - 8)
                )
                self.assertGreater(float(valid.mean()), 0.5)
                self.assertLess(float(np.abs(reprojected[valid] - current[valid]).mean()), 0.002)

    def test_moving_foreground_trusts_object_motion_and_rejects_disocclusion(self):
        previous = textured_frame(72, 96)
        current = previous.copy()
        previous[22:50, 20:48] = (0.9, 0.1, 0.2)
        current[22:50, 28:56] = (0.9, 0.1, 0.2)
        backward = np.zeros((72, 96, 2), np.float32)
        forward = np.zeros_like(backward)
        backward[22:50, 28:56, 0] = -8 / 96
        forward[22:50, 20:48, 0] = 8 / 96

        result = assess_motion(current, previous, backward, forward, scene_cut_threshold=0.3)

        self.assertFalse(result.reset)
        self.assertGreater(float(result.confidence[29:43, 35:49].mean()), 0.95)
        self.assertEqual(float(result.confidence[29:43, 21:27].max()), 0.0)
        self.assertGreater(float(result.confidence[4:14, 4:14].mean()), 0.95)

    def test_entering_and_leaving_objects_reject_changed_pixels_but_keep_background(self):
        background = textured_frame(72, 96)
        foreground = background.copy()
        foreground[22:50, 32:60] = (0.05, 0.95, 0.1)
        zero = np.zeros((72, 96, 2), np.float32)

        for current, previous in ((foreground, background), (background, foreground)):
            with self.subTest(direction="enter" if current is foreground else "leave"):
                result = assess_motion(current, previous, zero, zero, scene_cut_threshold=0.3)
                self.assertFalse(result.reset)
                self.assertEqual(float(result.confidence[29:43, 39:53].max()), 0.0)
                self.assertGreater(float(result.confidence[4:14, 4:14].mean()), 0.95)

    def test_equal_histogram_cut_resets_while_flash_and_fade_have_explicit_behavior(self):
        previous = textured_frame(72, 96)
        zero = np.zeros((72, 96, 2), np.float32)
        rng = np.random.default_rng(17)
        cut_previous = rng.random(previous.shape, dtype=np.float32)
        pixels = cut_previous.reshape(-1, 3)
        permutation = rng.permutation(len(pixels))
        equal_histogram_cut = pixels[permutation].reshape(cut_previous.shape)
        np.testing.assert_array_equal(
            np.sort(equal_histogram_cut, axis=None),
            np.sort(cut_previous, axis=None),
        )

        cut = assess_motion(equal_histogram_cut, cut_previous, zero, zero, scene_cut_threshold=0.3)
        self.assertTrue(cut.reset)
        self.assertLess(float((cut.confidence > 0.5).mean()), 0.15)

        flash = np.clip(previous + 0.45, 0, 1)
        self.assertTrue(assess_motion(flash, previous, zero, zero, scene_cut_threshold=0.3).reset)

        fade_frames = [previous * value for value in (1.0, 0.96, 0.92, 0.88, 0.84)]
        fade_results = [
            assess_motion(current, prior, zero, zero, scene_cut_threshold=0.3)
            for prior, current in zip(fade_frames, fade_frames[1:])
        ]
        self.assertTrue(all(not result.reset for result in fade_results))
        self.assertTrue(all(float((result.confidence > 0.5).mean()) > 0.85 for result in fade_results))

    def test_300_frame_recurrence_is_finite_and_detail_does_not_feed_history(self):
        frame = textured_frame(16, 16)
        sessions = [
            TemporalSession(
                CheapPipeline(),
                options=TemporalOptions(detail_strength=detail, detail_radius=1),
                motion="zero",
            )
            for detail in (1, 2)
        ]
        penultimate = None
        outputs = None
        for index in range(300):
            outputs = [session.process(frame) for session in sessions]
            np.testing.assert_array_equal(sessions[0].history, sessions[1].history)
            self.assertTrue(all(np.isfinite(output).all() for output in outputs))
            if index == 298:
                penultimate = sessions[0].history.copy()

        self.assertEqual([session.frame_index for session in sessions], [300, 300])
        self.assertLess(float(np.abs(sessions[0].history - penultimate).max()), 1e-7)
        self.assertGreater(float(np.abs(outputs[0] - outputs[1]).mean()), 1e-5)


class BenchmarkContractTests(unittest.TestCase):
    def test_known_motion_metric_rewards_advected_effect_and_penalizes_screen_space_stale_effect(self):
        previous_source = textured_frame(64, 80)
        current_source = translate(previous_source, 3, 0)
        yy, xx = np.indices(previous_source.shape[:2], dtype=np.float32)
        previous_effect = (0.025 * np.sin(xx * 0.47) * np.cos(yy * 0.29))[..., None]
        previous_effect = np.repeat(previous_effect, 3, axis=-1).astype(np.float32)
        current_effect = translate(previous_effect, 3, 0)
        backward = constant_motion(previous_source, 3, 0)
        forward = -backward

        aligned_error, coverage = benchmark.warped_effect_error(
            current_source + current_effect,
            previous_source + previous_effect,
            current_source,
            previous_source,
            backward,
            forward,
        )
        stale_error, stale_coverage = benchmark.warped_effect_error(
            current_source + previous_effect,
            previous_source + previous_effect,
            current_source,
            previous_source,
            backward,
            forward,
        )

        self.assertGreater(coverage, 0.5)
        self.assertAlmostEqual(coverage, stale_coverage)
        self.assertLess(aligned_error, 1e-5)
        self.assertGreater(stale_error, 0.01)

    def test_empty_visibility_has_null_error_and_strict_finite_json(self):
        frame = textured_frame(16, 16)
        zero = np.zeros((16, 16, 2), np.float32)
        error, coverage = benchmark.warped_effect_error(frame, frame, frame, frame, zero, zero)
        metrics = benchmark.aggregate_warp_metrics([error], [coverage])

        self.assertIsNone(error)
        self.assertEqual(metrics, {"warped_effect_error": None, "coverage": 0.0, "valid_frames": 0})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            benchmark.write_report(path, {"cases": {"empty": metrics}})
            self.assertEqual(json.loads(path.read_text())["cases"]["empty"]["warped_effect_error"], None)
            self.assertNotIn("NaN", path.read_text())

    def test_preview_uses_exact_fractional_source_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "fake-ffmpeg"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "sys.stdin.buffer.read()\n"
                "pathlib.Path(sys.argv[-1]).write_text(sys.argv[sys.argv.index('-r') + 1])\n"
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            output = directory / "preview.mp4"

            benchmark.encode_preview(np.zeros((2, 8, 8, 3), np.float32), output, Fraction(24000, 1001), ffmpeg=str(executable))

            self.assertEqual(output.read_text(), "24000/1001")

    def test_failed_preview_removes_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            executable = directory / "fake-ffmpeg"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib, sys\n"
                "sys.stdin.buffer.read()\n"
                "pathlib.Path(sys.argv[-1]).write_bytes(b'partial')\n"
                "raise SystemExit(7)\n"
            )
            executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
            output = directory / "preview.mp4"

            with self.assertRaises(Exception):
                benchmark.encode_preview(np.zeros((2, 8, 8, 3), np.float32), output, Fraction(30, 1), ffmpeg=str(executable))

            self.assertFalse(output.exists())
            self.assertEqual(list(directory.glob("*.partial.*")), [])


if __name__ == "__main__":
    unittest.main()
