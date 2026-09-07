"""Metal backend for the converter: frames stream through ``mlxdlss stream`` over pipes.

Protocol (per frame): little-endian uint32 flags (bit 0 = reset history), then
float32 colour (H, W, 3); in temporal mode also motion (H, W, 2) as normalised
history-UV offsets and depth (H, W, 1). One float32 RGB frame comes back per
input frame. Protocol 2 adds flag bit 1 for an optional float32 H×W×1 history
confidence plane after depth; the temporal adapter explicitly requests it.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from .composition import compose_detail, resample
from .temporal import BLEND_SCALE, FlowMotionEstimator, zero_motion, resolve_motion, resize_guide

RESET_FLAG = 1


def find_mlxdlss(explicit: str | None = None) -> str:
    import os

    candidates = [explicit, os.environ.get("MLXDLSS_BINARY"), shutil.which("mlxdlss")]
    root = Path(__file__).resolve().parents[2]
    candidates += [str(root / ".build" / "release" / "mlxdlss"), str(root / ".build" / "debug" / "mlxdlss")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("mlxdlss binary not found: build it with `swift build -c release --product mlxdlss`, pass --mlxdlss PATH or set MLXDLSS_BINARY")


class MLXDLSSStreamSession:
    """Sequential Metal renderer; source-size frames, processing-size history."""

    def __init__(self, model_package: str | Path, width: int, height: int, *, temporal: bool = True,
                 motion: Callable | str = "flow", scene_cut_threshold: float = 0.3, mlxdlss: str | None = None,
                 profile: str = "standard", intensity: float = 1.0, execution: str = "metal-fused",
                 precision: str = "float16", processing_scale: float = 1.0, detail_strength: float = 1.0,
                 colour_strength: float = 1.0, detail_radius: float = 4.0, blend_scale: float = BLEND_SCALE,
                 robust_motion: bool = True):
        if not 1 <= processing_scale <= 4:
            raise ValueError("processing_scale must be within [1, 4]")
        if width <= 0 or height <= 0:
            raise ValueError("frame dimensions must be positive")
        self.width, self.height, self.temporal = width, height, temporal
        self.processing_width = round(width * processing_scale) if temporal else width
        self.processing_height = round(height * processing_scale) if temporal else height
        self.scene_cut_threshold = scene_cut_threshold
        self.robust_motion = robust_motion
        self.detail = (detail_strength, colour_strength, detail_radius)
        if not temporal or motion == "zero":
            self.motion = zero_motion
        elif motion == "flow":
            self.motion = FlowMotionEstimator()
        elif callable(motion):
            self.motion = motion
        else:
            raise ValueError("motion must be 'flow', 'zero' or a callable")
        command = [find_mlxdlss(mlxdlss), "stream", str(model_package),
                   "--width", str(self.processing_width), "--height", str(self.processing_height),
                   "--mode", "temporal" if temporal else "first-frame", "--execution", execution,
                   "--precision", precision, "--profile", profile, "--intensity", str(intensity)]
        if temporal:
            command += ["--protocol-version", "2", "--blend-scale", str(blend_scale)]
        else:
            command += ["--processing-scale", str(processing_scale), "--detail-strength", str(detail_strength),
                        "--colour-strength", str(colour_strength), "--detail-radius", str(detail_radius)]
        self._stderr = tempfile.TemporaryFile()
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr)
        except BaseException:
            self._stderr.close()
            raise
        self.previous: np.ndarray | None = None
        self.frame_index = self.scene_cuts = self.frames = 0
        self._reset_pending = False
        self._closed = False
        self._summary = {}
        self._depth = np.ones((self.processing_height, self.processing_width, 1), dtype="<f4").tobytes()

    def reset(self):
        self.previous = None
        self.frame_index = 0
        self._reset_pending = True

    def _error_text(self):
        self._stderr.seek(0)
        return self._stderr.read().decode(errors="replace")[-1000:]

    def process_frame(self, frame: np.ndarray, *, motion: np.ndarray | None = None,
                      history_confidence: np.ndarray | None = None) -> np.ndarray:
        if self._closed:
            raise RuntimeError("mlxdlss stream is closed")
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != (self.height, self.width, 3) or not np.isfinite(frame).all():
            raise ValueError(f"frame must be finite ({self.height}, {self.width}, 3)")
        if not self.temporal and history_confidence is not None:
            raise ValueError("history confidence requires temporal mode")
        estimate = resolve_motion(self.motion, frame, self.previous, motion, history_confidence,
                                  scene_cut_threshold=self.scene_cut_threshold, robust_motion=self.robust_motion)
        flags = RESET_FLAG if self._reset_pending or estimate.reset else 0
        if estimate.reset:
            self.scene_cuts += 1
        width, height = self.processing_width, self.processing_height
        processing = resample(frame, width, height) if self.temporal else frame
        confidence = estimate.confidence if self.temporal else None
        if confidence is not None:
            confidence = resize_guide(confidence, width, height, confidence=True)
            flags |= 2
        payload = [struct.pack("<I", flags), np.ascontiguousarray(processing, dtype="<f4").tobytes()]
        if self.temporal:
            mv = resize_guide(estimate.motion_uv, width, height)
            payload += [np.ascontiguousarray(mv, dtype="<f4").tobytes(), self._depth]
            if confidence is not None:
                payload.append(np.ascontiguousarray(confidence, dtype="<f4").tobytes())
        try:
            self.process.stdin.write(b"".join(payload)); self.process.stdin.flush()
            expected = height * width * 3 * 4
            data = bytearray()
            while len(data) < expected:
                chunk = self.process.stdout.read(expected - len(data))
                if not chunk:
                    raise RuntimeError(f"mlxdlss stream ended early (rebuild the binary if its protocol is outdated): {self._error_text()}")
                data += chunk
        except (BrokenPipeError, OSError) as error:
            detail = self._error_text()
            self.abort()
            raise RuntimeError(f"mlxdlss stream failed (rebuild the binary if its protocol is outdated): {detail}") from error
        except BaseException:
            self.abort()
            raise
        output = np.frombuffer(bytes(data), dtype="<f4").reshape(height, width, 3)
        self.previous = frame.copy()
        self.frame_index = 1 if flags & RESET_FLAG else self.frame_index + 1
        self.frames += 1
        self._reset_pending = False
        if self.temporal:
            detail, colour, radius = self.detail
            output = compose_detail(frame, resample(output, self.width, self.height),
                                    detail_strength=detail, colour_strength=colour, radius=radius)
        return output

    def abort(self):
        if self._closed:
            return
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self._release()

    def _release(self):
        for pipe in (self.process.stdin, self.process.stdout):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        self._stderr.close()
        self._closed = True

    def close(self) -> dict:
        if self._closed:
            return self._summary
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            self.process.wait(timeout=30)
            stderr = self._error_text()
            if self.process.returncode:
                raise RuntimeError(f"mlxdlss stream exited with {self.process.returncode}: {stderr}")
            for line in stderr.splitlines():
                if line.startswith("{"):
                    try:
                        self._summary = json.loads(line)
                    except json.JSONDecodeError:
                        pass
            return self._summary
        finally:
            self.abort()


class MLXDLSSFrameGenStream:
    """Frame generation through ``mlxdlss framegen-stream`` (Metal): push frames in order and collect the
    ``factor - 1`` generated frames of every consecutive pair, in stream order, as uint8 arrays.

    The server computes ``batch`` pairs per pass, so a pair's frames come back once its window is
    complete (``push`` returns them) or when the input ends (``finish`` returns the rest)."""

    def __init__(self, weights: str | Path, width: int, height: int, *, factor: int = 2, precision: str = "float16",
                 mlxdlss: str | None = None, batch: int = 4):
        self.width, self.height, self.factor, self.batch = width, height, factor, max(1, int(batch))
        command = [find_mlxdlss(mlxdlss), "framegen-stream", "--weights", str(weights), "--width", str(width), "--height", str(height),
                   "--factor", str(factor), "--batch", str(self.batch), "--format", "u8", "--precision", precision]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.frames = 0
        self.pending_pairs = 0

    def _read_frames(self, count: int) -> list[np.ndarray]:
        expected = self.height * self.width * 3
        outputs = []
        for _ in range(count):
            data = bytearray()
            while len(data) < expected:
                chunk = self.process.stdout.read(expected - len(data))
                if not chunk:
                    raise RuntimeError(f"mlxdlss framegen-stream ended early: {self.process.stderr.read().decode(errors='replace')[-500:]}")
                data += chunk
            outputs.append(np.frombuffer(bytes(data), dtype=np.uint8).reshape(self.height, self.width, 3).copy())
        return outputs

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Send one uint8 (H, W, 3) frame; returns the generated frames of every pair whose window
        just completed (``batch * (factor - 1)`` frames, or none)."""
        frame = np.asarray(frame)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame must be ({self.height}, {self.width}, 3)")
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        payload = np.ascontiguousarray(frame).tobytes()
        try:
            self.process.stdin.write(payload); self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(f"mlxdlss framegen-stream failed: {self.process.stderr.read().decode(errors='replace')[-500:]}") from error
        self.frames += 1
        if self.frames == 1:
            return []
        self.pending_pairs += 1
        if self.pending_pairs < self.batch:
            return []
        outputs = self._read_frames(self.pending_pairs * (self.factor - 1))
        self.pending_pairs = 0
        return outputs

    def finish(self) -> list[np.ndarray]:
        """End the input and return the generated frames of the pairs still pending."""
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            self.process.stdin = None
        outputs = self._read_frames(self.pending_pairs * (self.factor - 1)) if self.pending_pairs else []
        self.pending_pairs = 0
        return outputs

    def close(self) -> dict:
        """Finish the stream (discarding any pending output) and return the server's JSON summary."""
        try:
            self.finish()
        except RuntimeError:
            pass
        stderr = self.process.stderr.read().decode(errors="replace")
        self.process.wait()
        for pipe in (self.process.stdout, self.process.stderr):
            if pipe is not None:
                pipe.close()
        for line in stderr.strip().split("\n"):
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        return {}
