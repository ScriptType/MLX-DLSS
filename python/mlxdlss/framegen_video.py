"""Frame generation for whole videos through FFmpeg.

Two modes: ``fps`` keeps the duration and multiplies the frame rate (generated
frames are inserted between the originals; audio is copied), ``slowmo`` keeps
the frame rate and stretches the duration by ``factor`` (audio is stretched
with FFmpeg's ``atempo`` — pitch-preserving WSOLA — copied as-is, or dropped).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .framegen import FrameGenerator
from .video import VideoToolError

AUDIO_MODES = ("copy", "stretch", "none")


def atempo_chain(ratio: float) -> str:
    """FFmpeg ``atempo`` filter chain for a playback-speed ``ratio`` (0.5 = twice as long).

    A single ``atempo`` accepts 0.5..100, so slower ratios are split into 0.5 steps
    (``atempo=0.5,atempo=0.5`` for x4) with one final factor in range."""
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    steps: list[float] = []
    remaining = ratio
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    while remaining > 100.0:
        steps.append(100.0)
        remaining /= 100.0
    steps.append(remaining)
    return ",".join(f"atempo={step:g}" for step in steps)


@dataclass
class FrameGenOptions:
    mode: str = "fps"                 # "fps" (rate x factor) or "slowmo" (duration x factor)
    factor: int = 2
    audio: str = "copy"               # copy | stretch (slowmo only) | none
    frame_limit: int | None = None
    encode_args: list[str] | None = None
    decode_args: list[str] = field(default_factory=list)
    overwrite: bool = False
    status_interval: float = 60.0
    backend: str = "torch"            # torch | mlxdlss (Metal, macOS: frames stream through `mlxdlss framegen-stream`)
    mlxdlss: str | None = None
    mlxdlss_weights: str | None = None    # dense safetensors for the Swift runtime (defaults to the torch weights path)
    mlxdlss_precision: str = "float16"
    batch: int = 4                    # consecutive pairs generated per pass (both backends)


@dataclass
class FrameGenResult:
    input_frames: int
    output_frames: int
    seconds: float
    width: int
    height: int
    input_fps: float
    output_fps: float
    output: Path


def interpolate_video(
    source: str | Path,
    destination: str | Path,
    generator: FrameGenerator | None,
    options: FrameGenOptions | None = None,
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    log: Callable[[str], None] | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> FrameGenResult:
    """Decode ``source``, generate ``factor - 1`` frames between every consecutive pair, encode.

    Pairs are generated ``options.batch`` at a time; the output keeps the stream order
    (frame, generated frames, next frame, ...) by holding input frames until their pair is done.
    ``progress(input_frames_done, input_frames_expected)`` is called per input frame;
    ``should_stop()`` is polled per input frame and ends the job early."""
    options = options or FrameGenOptions()
    log = log or (lambda message: print(message, file=sys.stderr, flush=True))
    source = Path(source); destination = Path(destination)
    if options.mode not in ("fps", "slowmo"):
        raise VideoToolError("mode must be 'fps' or 'slowmo'")
    if options.factor < 2:
        raise VideoToolError("factor must be at least 2")
    if options.audio not in AUDIO_MODES:
        raise VideoToolError(f"audio must be one of {AUDIO_MODES}")
    if options.backend not in ("torch", "mlxdlss"):
        raise VideoToolError("backend must be 'torch' or 'mlxdlss'")
    if options.backend == "torch" and generator is None:
        raise VideoToolError("the torch backend needs a FrameGenerator")
    if options.backend == "mlxdlss" and not options.mlxdlss_weights:
        raise VideoToolError("the mlxdlss backend needs mlxdlss_weights (dense frame generation safetensors)")
    if options.audio == "stretch" and options.mode != "slowmo":
        raise VideoToolError("audio 'stretch' only applies to slowmo (fps mode keeps the duration; use 'copy')")
    from .video import ConvertOptions
    from .video_pipeline import FrameGenerationStage, run_video

    io = ConvertOptions(pixel_format="rgb24", audio=options.audio, frame_limit=options.frame_limit,
                        decode_args=options.decode_args, encode_args=options.encode_args,
                        overwrite=options.overwrite, status_interval=options.status_interval)
    info, frames_in, frames_out, seconds, rate = run_video(
        source, destination, [FrameGenerationStage(generator, options)], io,
        ffmpeg=ffmpeg, ffprobe=ffprobe, log=log, progress=progress, should_stop=should_stop, progress_input=True,
    )
    return FrameGenResult(frames_in, frames_out, seconds, info.width, info.height, info.fps, float(rate), destination)
