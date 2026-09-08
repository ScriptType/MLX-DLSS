import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import patch

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
        from safetensors.torch import save_file
        from .test_framegen import synthetic_framegen_weights

        cls.fg_weights = pathlib.Path(cls.directory) / "framegen.safetensors"
        save_file(synthetic_framegen_weights(), str(cls.fg_weights))

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
                                                  scene_cut_threshold=0, mlxdlss=MLXDLSS_BINARY, protocol_version=3)
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

    def test_metal_display_matches_cpu_recipe_without_feeding_display_into_history(self):
        frame = np.random.default_rng(49).random((17, 19, 3), dtype=np.float32)
        for scale, detail, colour, radius in ((1, 1, 1, 4), (2, 1, 1, 4), (4, 2, 0.5, 1.25),
                                               (1.5, 2, 1, 4), (2, 0, 0, 0.5)):
            with self.subTest(scale=scale, detail=detail):
                kwargs = dict(motion="zero", processing_scale=scale, detail_strength=detail,
                              colour_strength=colour, detail_radius=radius, scene_cut_threshold=0,
                              mlxdlss=MLXDLSS_BINARY)
                legacy = MLXDLSSStreamSession(self.package, 19, 17, protocol_version=3, **kwargs)
                metal = MLXDLSSStreamSession(self.package, 19, 17, **kwargs)
                try:
                    for index in range(4):
                        if index == 3:
                            legacy.reset(); metal.reset()
                        source = np.roll(frame, index, axis=1)
                        confidence = np.full((17, 19, 1), 0.75 if index == 2 else 1, np.float32)
                        expected = legacy.process_frame(source, history_confidence=confidence)
                        actual = metal.process_frame(source, history_confidence=confidence)
                        np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-7)
                    self.assertEqual(metal.close()["output_shape"], [17, 19, 3])
                    legacy.close()
                finally:
                    legacy.abort(); metal.abort()

    def test_float32_metal_fused_preserves_reference_precision(self):
        frame = np.random.default_rng(73).random((17, 19, 3), dtype=np.float32)
        eager = MLXDLSSStreamSession(self.package, 19, 17, motion="zero", precision="float32",
                                    execution="eager", mlxdlss=MLXDLSS_BINARY)
        fused = MLXDLSSStreamSession(self.package, 19, 17, motion="zero", precision="float32",
                                    execution="metal-fused", mlxdlss=MLXDLSS_BINARY)
        try:
            for index in range(2):
                source = np.roll(frame, index, axis=1)
                np.testing.assert_array_equal(fused.process_frame(source), eager.process_frame(source))
            eager.close(); fused.close()
        finally:
            eager.abort(); fused.abort()

    def test_gpu_chain_keeps_float_originals_order_and_short_final_window(self):
        from mlxdlss.mlxdlss_stream import MLXDLSSFrameGenStream
        from mlxdlss.motion_quality import MotionEstimate
        from mlxdlss.temporal import prepare_temporal_frame

        rng = np.random.default_rng(83)
        frames = [np.full((17, 19, 3), 0.5001, np.float32)] * 2
        frames += [rng.random((17, 19, 3), dtype=np.float32) * 0.2 + 0.3 for _ in range(4)]
        for factor, batch, count, format in ((2, 1, 0, "f32"), (2, 4, 1, "f32"), (3, 4, 2, "f32"),
                                            (4, 4, 6, "f32"), (3, 2, 6, "u8"), (2, 4, 6, "u16")):
            with self.subTest(factor=factor, batch=batch, count=count, format=format):
                precision = "float32" if format == "u16" else "float16"
                kwargs = dict(motion="zero", processing_scale=2, detail_strength=2,
                              mlxdlss=MLXDLSS_BINARY, precision=precision)
                nr = MLXDLSSStreamSession(self.package, 19, 17, **kwargs)
                fg = MLXDLSSFrameGenStream(self.fg_weights, 19, 17, factor=factor, batch=batch,
                                          format="f32", mlxdlss=MLXDLSS_BINARY, precision=precision)
                chain = MLXDLSSStreamSession(self.package, 19, 17, framegen_weights=self.fg_weights,
                                             framegen_factor=factor, framegen_batch=batch, framegen_precision=precision,
                                             output_format=format, **kwargs)
                try:
                    originals, generated, actual = [], [], []
                    for index, frame in enumerate(frames[:count]):
                        prepared = prepare_temporal_frame(frame, MotionEstimate(
                            np.zeros((17, 19, 2), np.float32), None, index == 3), 2)
                        original = nr._process_prepared(prepared)
                        originals.append(original)
                        generated.extend(fg.push(original))
                        returned = chain.push_prepared(prepared)
                        expected_count = 1 if index == 0 else batch * factor if index % batch == 0 else 0
                        self.assertEqual(len(returned), expected_count)
                        actual.extend(returned)
                    generated.extend(fg.finish())
                    actual.extend(chain.finish())
                    self.assertEqual(chain.finish(), [])
                    expected = originals[:1]
                    for i, original in enumerate(originals[1:]):
                        expected += generated[i * (factor - 1):(i + 1) * (factor - 1)] + [original]
                    self.assertEqual(len(actual), max(0, count - 1) * factor + int(count > 0))
                    for before, after in zip(expected, actual):
                        if format != "f32":
                            maximum = 255 if format == "u8" else 65535
                            before = (np.clip(before, 0, 1) * np.float32(maximum) + 0.5).astype(after.dtype)
                        np.testing.assert_array_equal(after, before)
                    summary = chain.close()
                    self.assertTrue(summary["gpu_framegen"])
                    self.assertEqual(summary["output_frames"], len(actual))
                    nr.close(); fg.close()
                    with self.assertRaises(RuntimeError):
                        chain.push_prepared(None)
                finally:
                    nr.abort(); fg.abort(); chain.abort()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
    def test_native_video_chain_preserves_audio_rate_and_closes_on_cancel(self):
        from mlxdlss.framegen_video import FrameGenOptions
        from mlxdlss.video import ConvertOptions, probe
        from mlxdlss.video_pipeline import NeuralRenderStage, FrameGenerationStage, run_video
        from .test_video import make_clip

        root = pathlib.Path(self.directory)
        source = root / "chain-input.mp4"
        make_clip(source, frames=6)
        sessions = []
        def create(*args, **kwargs):
            session = MLXDLSSStreamSession(*args, **kwargs)
            sessions.append(session)
            return session
        for mode, audio, pixel_format, cancel in (("fps", "copy", "rgb24", False),
                                                   ("slowmo", "stretch", "rgb48le", False),
                                                   ("fps", "copy", "rgb24", True)):
            nr = NeuralRenderStage(None, ConvertOptions(backend="mlxdlss", model_package=str(self.package),
                                  motion="zero", mlxdlss=MLXDLSS_BINARY, enhance={"processing_scale": 2}))
            fg = FrameGenerationStage(None, FrameGenOptions(backend="mlxdlss", mlxdlss=MLXDLSS_BINARY,
                                      mlxdlss_weights=str(self.fg_weights), factor=3, batch=4, mode=mode, audio=audio), floating=True)
            outputs = []
            target = root / f"chain-{mode}-{cancel}.mp4"
            with patch("mlxdlss.mlxdlss_stream.MLXDLSSStreamSession", side_effect=create), \
                 patch("mlxdlss.mlxdlss_stream.MLXDLSSFrameGenStream", side_effect=AssertionError("separate FG process started")):
                run_video(source, target, [nr, fg], ConvertOptions(pixel_format=pixel_format, overwrite=True),
                          log=lambda _: None, progress=lambda count, total: outputs.append(count),
                          should_stop=lambda: cancel and bool(outputs))
            session = sessions[-1]
            self.assertTrue(session._closed)
            self.assertIsNotNone(session.process.poll())
            self.assertTrue(session.process.stdin.closed and session.process.stdout.closed)
            self.assertEqual(session.framegen_factor, 3)
            info = probe(target)
            self.assertEqual(info.frame_count, 1 if cancel else 16)
            self.assertEqual(info.fps, 10 if mode == "slowmo" else 30)
            self.assertTrue(info.has_audio)
