import threading
import unittest
import pathlib
import shutil
import tempfile
from unittest.mock import patch

import numpy as np

from mlxdlss.temporal import TemporalOptions, TemporalSession
from mlxdlss.video import ConvertOptions, probe
from mlxdlss.video_pipeline import FrameGenerationStage, NeuralRenderStage, run_video

from .test_video_temporal_quality import ProbePipeline


class VideoPipelineTests(unittest.TestCase):
    def test_device_chain_selection_preserves_backend_order_and_float_contract(self):
        from mlxdlss.framegen_video import FrameGenOptions

        nr = NeuralRenderStage(None, ConvertOptions(backend="mlxdlss", mlxdlss="/native"))
        fg = FrameGenerationStage(None, FrameGenOptions(backend="mlxdlss", mlxdlss="/native"), floating=True)
        with patch("mlxdlss.mlxdlss_stream.find_mlxdlss", side_effect=lambda path: path):
            self.assertTrue(nr.can_generate_on_device(fg))
            self.assertFalse(nr.can_generate_on_device(nr))
            for field, value in (("backend", "torch"), ("mlxdlss", "/another")):
                with patch.object(fg.options, field, value):
                    self.assertFalse(nr.can_generate_on_device(fg))
            with patch.object(fg, "floating", False):
                self.assertFalse(nr.can_generate_on_device(fg))
            with patch.object(nr.options, "temporal", False):
                self.assertFalse(nr.can_generate_on_device(fg))

    def test_prefetch_preserves_temporal_outputs_cuts_and_frame_order(self):
        rng = np.random.default_rng(27)
        dark = rng.random((17, 19, 3), dtype=np.float32) * 0.2
        frames = [dark, dark.copy(), dark + 0.7, dark + 0.71]
        options = ConvertOptions(motion="zero", enhance={"processing_scale": 1.5, "detail_strength": 2})
        reference = TemporalSession(ProbePipeline(), motion="zero",
                                    options=TemporalOptions(processing_scale=1.5, detail_strength=2))
        expected = [reference.process(frame) for frame in frames]
        stage = NeuralRenderStage(ProbePipeline(), options)
        actual = list(stage.apply(iter(frames), 19, 17, prefetch=True))
        self.assertEqual(stage.scene_cuts, 1)
        self.assertEqual(len(actual), len(expected))
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(after, before)

    def test_motion_preparation_overlaps_render_but_history_stays_on_consumer(self):
        rendering, prepared = threading.Event(), threading.Event()
        consumer = threading.get_ident()
        calls = []

        class OverlapPipeline(ProbePipeline):
            def run_features(self, features):
                calls.append(threading.get_ident())
                rendering.set()
                if len(calls) == 1 and not prepared.wait(5):
                    raise AssertionError("next motion did not overlap rendering")
                return super().run_features(features)

        def motion(current, previous):
            self.assertTrue(rendering.wait(5))
            self.assertNotEqual(threading.get_ident(), consumer)
            prepared.set()
            return np.zeros((*current.shape[:2], 2), np.float32)

        stage = NeuralRenderStage(OverlapPipeline(), ConvertOptions(motion=motion))
        frame = np.full((16, 16, 3), 0.25, np.float32)
        upstream_threads = []
        def upstream():
            for _ in range(2):
                upstream_threads.append(threading.get_ident())
                yield frame
        self.assertEqual(len(list(stage.apply(upstream(), 16, 16, prefetch=True))), 2)
        self.assertEqual(calls, [consumer, consumer])
        self.assertEqual(upstream_threads, [consumer, consumer], "upstream FG must stay on the GPU consumer thread")

    def test_prefetch_propagates_upstream_failure_and_closes_source(self):
        closed = threading.Event()

        def frames():
            try:
                yield np.zeros((16, 16, 3), np.float32)
                raise ValueError("broken decoder")
            finally:
                closed.set()

        stage = NeuralRenderStage(ProbePipeline(), ConvertOptions(motion="zero"))
        with self.assertRaisesRegex(ValueError, "broken decoder"):
            list(stage.apply(frames(), 16, 16, prefetch=True))
        self.assertTrue(closed.is_set())

    def test_worker_motion_failure_closes_source_after_last_valid_frame(self):
        closed, cancelled = threading.Event(), threading.Event()
        def frames():
            try:
                for _ in range(3):
                    yield np.full((16, 16, 3), 0.25, np.float32)
            finally:
                closed.set()
        def failed_motion(current, previous):
            raise ValueError("motion preparation failed")
        pipeline = ProbePipeline()
        stage = NeuralRenderStage(pipeline, ConvertOptions(motion=failed_motion))
        outputs = stage.apply(frames(), 16, 16, prefetch=True, cancel=cancelled.set)
        self.assertEqual(next(outputs).shape, (16, 16, 3))
        with self.assertRaisesRegex(ValueError, "motion preparation failed"):
            next(outputs)
        self.assertEqual(len(pipeline.inputs), 1)
        self.assertTrue(closed.is_set())
        self.assertTrue(cancelled.is_set())

    def test_float_frame_generation_keeps_sub_byte_detail_and_originals(self):
        from mlxdlss.framegen import FrameGenerator
        from mlxdlss.framegen_video import FrameGenOptions
        from .test_framegen import synthetic_framegen_weights

        generator = FrameGenerator(synthetic_framegen_weights(), device="cpu")
        frames = [np.full((17, 19, 3), value, np.float32) for value in (0.5001, 0.5001, 0.7501, 0.7501)]
        for factor in (2, 3, 4):
            stage = FrameGenerationStage(generator, FrameGenOptions(factor=factor, batch=2), floating=True)
            result = list(stage.apply(iter(frames), 19, 17))
            self.assertEqual(len(result), 3 * factor + 1)
            for index, original in enumerate(frames):
                np.testing.assert_array_equal(result[index * factor], original)
            np.testing.assert_allclose(result[1], frames[0], atol=1e-7, rtol=0)
            self.assertEqual(result[1].dtype, np.float32)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VideoChainIOTests(unittest.TestCase):
    def test_both_effect_orders_keep_frame_rate_audio_and_short_final_batch(self):
        from mlxdlss.framegen import FrameGenerator
        from mlxdlss.framegen_video import FrameGenOptions
        from .test_framegen import synthetic_framegen_weights
        from .test_video import make_clip

        generator = FrameGenerator(synthetic_framegen_weights(), device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            make_clip(root / "in.mp4", frames=3)
            for nr_first in (True, False):
                for mode, audio in (("fps", "copy"), ("slowmo", "stretch")):
                    nr = NeuralRenderStage(ProbePipeline(), ConvertOptions(motion="zero"))
                    fg = FrameGenerationStage(generator, FrameGenOptions(factor=3, batch=4, mode=mode, audio=audio), floating=True)
                    stages = [nr, fg] if nr_first else [fg, nr]
                    result = root / f"{nr_first}-{mode}.mp4"
                    run_video(root / "in.mp4", result, stages, ConvertOptions(), log=lambda _: None)
                    info = probe(result)
                    self.assertEqual(info.frame_count, 7)
                    self.assertEqual(info.fps, 30 if mode == "fps" else 10)
                    self.assertTrue(info.has_audio)
