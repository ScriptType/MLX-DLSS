import pathlib
import shutil
import tempfile
import unittest

import numpy as np

from .synthetic import synthetic_weights, write_logical_safetensors

try:
    from mlxdlss.mlxdlss_stream import MLXDLSSStreamSession, find_mlxdlss
    MLXDLSS_BINARY = find_mlxdlss()
except Exception:  # binary absent (Linux, Windows, or not built)
    MLXDLSS_BINARY = None


@unittest.skipUnless(MLXDLSS_BINARY, "mlxdlss binary is required (macOS build)")
class MLXDLSSStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mlxdlss.tools import cli as weights_cli

        cls.directory = tempfile.mkdtemp()
        weights = pathlib.Path(cls.directory) / "weights.safetensors"
        write_logical_safetensors(weights, synthetic_weights())
        cls.package = pathlib.Path(cls.directory) / "Synthetic.dlssmodel"
        assert weights_cli.main(["mlx", str(weights), str(cls.package)]) == 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_temporal_stream_returns_one_frame_per_input_and_resets_on_cut(self):
        session = MLXDLSSStreamSession(self.package, 64, 48, temporal=True, motion="zero", scene_cut_threshold=0.3, mlxdlss=MLXDLSS_BINARY)
        frame = np.random.default_rng(0).random((48, 64, 3)).astype(np.float32) * 0.2
        first = session.process_frame(frame)
        second = session.process_frame(frame)
        third = session.process_frame(np.clip(frame + 0.7, 0, 1))
        summary = session.close()
        self.assertEqual(first.shape, (48, 64, 3)); self.assertTrue(np.isfinite(second).all() and np.isfinite(third).all())
        self.assertEqual(session.scene_cuts, 1)
        self.assertEqual(summary.get("frames"), 3)

    def test_first_frame_stream_applies_the_recipe_on_the_swift_side(self):
        session = MLXDLSSStreamSession(self.package, 64, 48, temporal=False, mlxdlss=MLXDLSS_BINARY, processing_scale=2, detail_strength=2)
        frame = np.random.default_rng(1).random((48, 64, 3)).astype(np.float32)
        output = session.process_frame(frame)
        summary = session.close()
        self.assertEqual(output.shape, (48, 64, 3)); self.assertEqual(summary.get("mode"), "first-frame")

    def test_scaled_temporal_confidence_and_invalid_frame_preserve_pipe_order(self):
        frame = np.random.default_rng(4).random((17, 19, 3), dtype=np.float32)
        for scale in (1.5, 2, 4):
            with self.subTest(scale=scale):
                session = MLXDLSSStreamSession(self.package, 19, 17, motion="zero", processing_scale=scale, mlxdlss=MLXDLSS_BINARY)
                try:
                    with self.assertRaises(ValueError):
                        session.process_frame(frame, motion=np.full((17, 19, 2), np.nan))
                    first = session.process_frame(frame)
                    second = session.process_frame(frame, history_confidence=np.zeros((17, 19, 1), np.float32))
                    self.assertEqual(first.shape, frame.shape)
                    self.assertTrue(np.isfinite(second).all())
                    summary = session.close()
                    self.assertEqual(summary["frames"], 2)
                    self.assertEqual(summary["shape"], [round(17 * scale), round(19 * scale), 3])
                finally:
                    session.abort()

    def test_compact_integer_and_resampled_frames_match_legacy_protocol(self):
        from mlxdlss.motion_quality import MotionEstimate
        from mlxdlss.temporal import prepare_temporal_frame

        for dtype in (np.uint8, np.uint16):
            for scale in (1, 1.5):
                with self.subTest(dtype=dtype, scale=scale):
                    raw = np.random.default_rng(19).integers(0, np.iinfo(dtype).max + 1, (17, 19, 3), dtype=dtype)
                    raw = raw[:, ::-1]  # exercise packed noncontiguous source input
                    motion = np.zeros((17, 19, 2), np.float32)
                    motion[..., 0] = -1 / 19
                    confidence = np.full((17, 19, 1), 0.75, np.float32)
                    confidence[:, :3] = 0
                    legacy = MLXDLSSStreamSession(self.package, 19, 17, motion="zero", processing_scale=scale,
                                                 scene_cut_threshold=0, mlxdlss=MLXDLSS_BINARY, protocol_version=2)
                    compact = MLXDLSSStreamSession(self.package, 19, 17, motion="zero", processing_scale=scale,
                                                  scene_cut_threshold=0, mlxdlss=MLXDLSS_BINARY)
                    try:
                        for index in range(3):
                            if index == 2:
                                legacy.reset(); compact.reset()
                            source = np.roll(raw, index, axis=1)
                            frame = source.astype(np.float32) / np.float32(np.iinfo(dtype).max)
                            expected = legacy.process_frame(frame, motion=motion, history_confidence=confidence)
                            prepared = prepare_temporal_frame(frame, MotionEstimate(motion, confidence, False), scale,
                                                              packed_color=source)
                            actual = compact._process_prepared(prepared)
                            np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(legacy.close()["frames"], 3)
                        self.assertEqual(compact.close()["frames"], 3)
                    finally:
                        legacy.abort(); compact.abort()
