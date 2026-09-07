import unittest
from unittest.mock import patch

import numpy as np

from mlxdlss.features import scaled_color
from mlxdlss.temporal import TemporalOptions, TemporalSession, compose_temporal, make_temporal_features
from mlxdlss.video_cli import build_parser
try:
    from mlxdlss.web.effects import parse_effects
except ImportError:
    parse_effects = None


class ProbePipeline:
    def __init__(self):
        self.inputs = []

    def run_features(self, features):
        self.inputs.append(features.copy())
        head = np.zeros((*features.shape[:2], 4), np.float32)
        head[..., 0] = 0.5
        return head


class TemporalQualityTests(unittest.TestCase):
    @unittest.skipUnless(parse_effects, "web extra is required")
    def test_new_video_defaults_and_legacy_choices(self):
        raw = [{"kind": "nr"}]
        self.assertTrue(parse_effects(raw, kind="video")[0].temporal)
        self.assertFalse(parse_effects(raw, kind="image")[0].temporal)
        self.assertFalse(parse_effects(raw)[0].temporal)
        self.assertNotIn("temporal", raw[0])
        self.assertFalse(parse_effects([{"kind": "nr", "temporal": False}], kind="video")[0].temporal)
        parser = build_parser()
        self.assertTrue(parser.parse_args(["convert", "a.mp4", "b.mp4"]).temporal)
        self.assertFalse(parser.parse_args(["convert", "a.mp4", "b.mp4", "--no-temporal"]).temporal)

    def test_history_conditions_the_next_model_input(self):
        pipeline = ProbePipeline()
        session = TemporalSession(pipeline, motion="zero")
        frame = np.full((16, 16, 3), 0.5, np.float32)
        first = session.process(frame)
        session.process(frame)
        np.testing.assert_array_equal(pipeline.inputs[1][:16, :16, 7:10], scaled_color(first))
        self.assertGreater(float(np.abs(first - frame).max()), 0.01)

    @unittest.skipUnless(parse_effects, "web extra is required")
    def test_malformed_effects_remain_validation_errors(self):
        for value in (None, 42, False):
            with self.assertRaises(ValueError):
                parse_effects([value], kind="video")

    def test_callers_cannot_modify_retained_history(self):
        session = TemporalSession(ProbePipeline(), motion="zero")
        frame = np.full((16, 16, 3), 0.5, np.float32)
        out = session.process(frame)
        expected = session.history.copy()
        out[:] = 0
        np.testing.assert_array_equal(session.history, expected)

    def test_rejected_history_cannot_tint_prediction(self):
        current = np.full((4, 4, 3), 0.5, np.float32)
        history = np.zeros_like(current)
        history[..., 0] = 1
        confidence = np.zeros((4, 4, 1), np.float32)
        features = make_temporal_features(current, history, np.zeros((4, 4, 2), np.float32),
                                          frame_index=1, history_confidence=confidence)
        np.testing.assert_array_equal(features[..., 7:10], scaled_color(current))
        result = compose_temporal(np.zeros((4, 4, 4), np.float32), current, features,
                                  history_confidence=confidence)
        np.testing.assert_array_equal(result, current)

    def test_full_confidence_preserves_reference_and_half_confidence_has_known_blend(self):
        current = np.full((4, 4, 3), 0.5, np.float32)
        history = np.ones_like(current)
        motion = np.zeros((4, 4, 2), np.float32)
        reference = make_temporal_features(current, history, motion, frame_index=1)
        head = np.zeros((4, 4, 4), np.float32)
        for c in (0.0, 0.5, 1.0):
            confidence = np.full((4, 4, 1), c, np.float32)
            features = make_temporal_features(current, history, motion, frame_index=1, history_confidence=confidence)
            result = compose_temporal(head, current, features, blend_scale=0.5, history_confidence=confidence)
            np.testing.assert_allclose(result, 0.5 + 0.125 * c * c, atol=1e-7)
            if c == 1:
                np.testing.assert_array_equal(features, reference)

    def test_bad_maps_fail_even_on_first_frame(self):
        frame = np.full((8, 8, 3), 0.5, np.float32)
        session = TemporalSession(ProbePipeline(), motion="zero")
        for motion in (np.zeros((8, 8, 1)), np.full((8, 8, 2), np.nan)):
            with self.assertRaises(ValueError):
                session.process(frame, motion=motion)
        for confidence in (np.ones((8, 8)), np.full((8, 8, 1), -0.1), np.full((8, 8, 1), np.inf)):
            with self.assertRaises(ValueError):
                session.process(frame, history_confidence=confidence)
        self.assertEqual(session.frame_index, 0)

    def test_scale_keeps_output_extent_and_unsharpened_history(self):
        frame = np.random.default_rng(3).random((81, 83, 3), dtype=np.float32)
        for scale in (1, 1.5, 2, 4):
            pipeline = ProbePipeline()
            session = TemporalSession(pipeline, options=TemporalOptions(processing_scale=scale, detail_strength=2), motion="zero")
            out = session.process(frame)
            self.assertEqual(out.shape, frame.shape)
            self.assertEqual(session.history.shape, (round(81 * scale), round(83 * scale), 3))
            reference_pipeline = ProbePipeline()
            reference = TemporalSession(reference_pipeline, options=TemporalOptions(processing_scale=scale), motion="zero")
            reference.process(frame)
            np.testing.assert_array_equal(session.history, reference.history)
            session.process(frame)
            reference.process(frame)
            np.testing.assert_array_equal(pipeline.inputs[1], reference_pipeline.inputs[1])

    def test_motion_scale_changes_grid_without_scaling_uv(self):
        from mlxdlss.temporal import resize_guide
        flow = np.full((17, 19, 2), (-3 / 19, 2 / 17), np.float32)
        scaled = resize_guide(flow, 38, 34)
        np.testing.assert_allclose(scaled[..., 0] * 38, -6, atol=1e-6)
        np.testing.assert_allclose(scaled[..., 1] * 34, 4, atol=1e-6)

    def test_invalid_scale_fails_before_starting_metal(self):
        from mlxdlss.mlxdlss_stream import MLXDLSSStreamSession
        with patch("mlxdlss.mlxdlss_stream.subprocess.Popen") as popen:
            for scale in (0, 5, float("nan")):
                with self.assertRaises(ValueError):
                    MLXDLSSStreamSession("model", 32, 32, motion="zero", processing_scale=scale)
            popen.assert_not_called()
