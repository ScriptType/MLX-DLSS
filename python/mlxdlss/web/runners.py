"""Executes a job's effect chain with the package's own pipelines.

Native Metal media handles supported macOS inputs; other paths use the
portable pipelines. FFmpeg supplies optional completed-video comparisons.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from ..video import find_tool, probe
from .effects import FrameGen, NeuralRender, OutputOptions, parse_effects
from .jobs import Job
from .settings import Settings

Report = Callable[[str, float, int, int | None], None]


class ModelCache:
    """Loaded pipelines keyed by their configuration; one load per process."""

    def __init__(self):
        self._lock = threading.Lock()
        self.execution_lock = threading.RLock()
        self._nr: dict[tuple, object] = {}
        self._fg: dict[tuple, object] = {}

    def neural_rendering(self, weights: str, device: str, precision: str):
        from ..pipeline import NeuralRenderingPipeline

        key = (str(Path(weights).expanduser()), device, precision)
        with self._lock:
            if key not in self._nr:
                self._nr[key] = NeuralRenderingPipeline.from_safetensors(key[0], device=device, precision=precision)
            return self._nr[key]

    def frame_generator(self, weights: str, device: str, precision: str):
        from ..framegen import FrameGenerator

        key = (str(Path(weights).expanduser()), device, precision)
        with self._lock:
            if key not in self._fg:
                self._fg[key] = FrameGenerator.from_safetensors(key[0], device=device, precision=precision)
            return self._fg[key]


class Cancelled(Exception):
    pass


class JobRunner:
    def __init__(self, settings_provider: Callable[[], Settings], cache: ModelCache | None = None):
        self.settings_provider = settings_provider
        self.cache = cache or ModelCache()

    # -- entry point ------------------------------------------------------------
    def __call__(self, job: Job, folder: Path, report: Report, should_stop: Callable[[], bool]) -> list[Path]:
        with self.cache.execution_lock:
            if should_stop():
                raise Cancelled()
            return self._run(job, folder, report, should_stop)

    def _run(self, job: Job, folder: Path, report: Report, should_stop: Callable[[], bool]) -> list[Path]:
        settings = self.settings_provider()
        job.backend = "/".join(sorted({settings.resolved_backend(e.get("kind", "nr")) for e in job.effects}))
        effects = parse_effects(job.effects)
        source = folder / ("input" + Path(job.input_name).suffix.lower())
        if job.kind == "image":
            return [self._image(source, folder, effects, settings, report, should_stop)]
        return self._video(source, folder, effects, settings, report, should_stop, job)

    # -- images -------------------------------------------------------------------
    def _image(self, source: Path, folder: Path, effects, settings: Settings, report: Report, should_stop=lambda: False) -> Path:
        from PIL import Image, ImageOps

        nr = next(e for e in effects if isinstance(e, NeuralRender))
        out = folder / "result.png"
        if settings.resolved_backend("nr") == "mlxdlss":
            # Metal: the whole frame stays on the GPU; the PyTorch graph below keeps the
            # activations of the entire frame in memory and needs tens of GB at 4K.
            from ..mlxdlss_stream import find_mlxdlss

            if not settings.nr_model:
                raise ValueError("set the .dlssmodel package for the Metal backend in Settings")
            report("neural rendering", 0.2, 0, 1)
            from . import native
            binary = find_mlxdlss(settings.mlxdlss_binary or None)
            arguments = native.rendering_arguments(nr, settings, video=False)
            if native.available(settings, [nr], source):
                command = [binary, "process-image", str(source), "--output", str(out)] + arguments
            else:
                command = [binary, "render-image", str(source), arguments[1], "--output", str(out),
                           "--execution", "metal-fused", "--precision", "float16"] + arguments[2:]
            native.run_media(command, report, should_stop)
            report("done", 1.0, 1, 1)
            return out
        if not settings.nr_weights:
            raise ValueError("set the neural rendering weights (logical safetensors) in Settings")
        report("loading the network", 0.05, 0, None)
        pipeline = self.cache.neural_rendering(settings.nr_weights, settings.device, settings.precision)
        with Image.open(source) as original:
            image = np.asarray(ImageOps.exif_transpose(original).convert("RGB"), np.float32) / 255.0
        report("neural rendering", 0.2, 0, 1)
        result = pipeline.enhance(
            image, profile=nr.profile, processing_scale=nr.processing_scale, detail_strength=nr.detail_strength,
            colour_strength=nr.colour_strength, detail_radius=nr.detail_radius, intensity=nr.intensity,
        )
        Image.fromarray((np.clip(result.image, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out)
        report("done", 1.0, 1, 1)
        return out

    # -- videos -------------------------------------------------------------------
    def _video(self, source: Path, folder: Path, effects, settings: Settings, report: Report, should_stop, job: Job) -> list[Path]:
        from . import native
        from ..video import ConvertOptions
        from ..video_pipeline import NeuralRenderStage, run_video

        output = OutputOptions.model_validate(job.output_options)
        target = folder / ("result.mov" if output.codec == "prores" else "result.mp4")
        # An explicit legacy slowmo/copy choice keeps its original audio semantics.
        legacy_audio = any(isinstance(e, FrameGen) and e.mode == "slowmo" and e.audio == "copy"
                           and output.include_audio for e in effects)
        if native.available(settings, effects, source) and not legacy_audio:
            report("native media processing", 0.02, 0, None)
            result = native.run_media(native.video_arguments(source, target, effects, settings, output), report, should_stop)
            nr = next((e for e in effects if isinstance(e, NeuralRender)), None)
            job.diagnostics = {"temporal": bool(nr and nr.temporal), "motion": result["motionBackend"],
                               "processing_scale": nr.processing_scale if nr else 1,
                               "scene_cuts": result["sceneResets"], "timing": result.get("timing"), "pipeline": "native"}
            report("done", 1, result["inputFrames"], result["inputFrames"])
            aligned = not output.start_frame and not any(isinstance(e, FrameGen) and e.mode == "slowmo" for e in effects)
            if aligned:
                report("rendering the comparison", 0.98, result["inputFrames"], result["inputFrames"])
                preview = self._preview(source, target, folder)
                if preview is not None:
                    job.preview = preview.name
            if should_stop():
                raise Cancelled()
            return [target]
        stages = []
        for effect in effects:
            if should_stop():
                raise Cancelled()
            if isinstance(effect, NeuralRender):
                stages.append(self._neural_rendering_stage(effect, settings))
            else:
                if not output.include_audio:
                    effect = effect.model_copy(update={"audio": "none"})
                stages.append(self._frame_generation_stage(effect, settings, floating=len(effects) > 1))
        name = " → ".join("neural rendering" if isinstance(e, NeuralRender) else f"frame generation x{e.factor}" for e in effects)
        def progress(done, total):
            fraction = done / total if total else 0.0
            report(f"{name} {done}/{total if total is not None else '?'}", min(0.97, fraction * 0.97), done, total)
        codecs = {"h264": None,
                  "hevc": ["-c:v", "libx265", "-crf", "18", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-movflags", "+faststart"],
                  "prores": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]}
        run_video(source, target, stages, ConvertOptions(overwrite=True, status_interval=1e9,
                  start_frame=output.start_frame, frame_limit=output.frame_limit, encode_args=codecs[output.codec],
                  audio="copy" if output.include_audio else "none"),
                  log=lambda _m: None, progress=progress, should_stop=should_stop)
        for stage in stages:
            if isinstance(stage, NeuralRenderStage):
                options = stage.options
                job.diagnostics = {"temporal": options.temporal, "motion": options.motion if options.temporal else "none",
                                   "processing_scale": options.enhance.get("processing_scale", 1.0), "scene_cuts": stage.scene_cuts}
        if should_stop():
            raise Cancelled()
        report("rendering the preview", 0.98, job.frames_done, job.frames_total)
        # A trimmed or slowed result no longer aligns with the original's clock.
        aligned = not output.start_frame and not any(isinstance(e, FrameGen) and e.mode == "slowmo" for e in effects)
        preview = self._preview(source, target, folder) if aligned else None
        if preview is not None:
            job.preview = preview.name
        return [target]

    def _neural_rendering_stage(self, nr: NeuralRender, settings: Settings):
        from ..video import ConvertOptions
        from ..video_pipeline import NeuralRenderStage

        backend = settings.resolved_backend("nr")
        if nr.motion in {"vision", "videotoolbox"} and nr.temporal:
            raise ValueError("Vision/VideoToolbox motion needs native Metal media on macOS 26; choose Automatic or OpenCV for this input")
        motion = "flow" if nr.motion == "automatic" else nr.motion
        enhance = {"profile": nr.profile, "processing_scale": nr.processing_scale, "detail_strength": nr.detail_strength,
                   "colour_strength": nr.colour_strength, "detail_radius": nr.detail_radius, "intensity": nr.intensity}
        if backend == "mlxdlss":
            if not settings.nr_model:
                raise ValueError("set the .dlssmodel package for the Metal backend in Settings")
            options = ConvertOptions(backend="mlxdlss", model_package=str(Path(settings.nr_model).expanduser()), mlxdlss=settings.mlxdlss_binary or None,
                                     temporal=nr.temporal, motion=motion, scene_cut_threshold=nr.scene_cut_threshold,
                                     overwrite=True, status_interval=1e9, enhance=enhance)
            pipeline = None
        else:
            if not settings.nr_weights:
                raise ValueError("set the neural rendering weights (logical safetensors) in Settings")
            pipeline = self.cache.neural_rendering(settings.nr_weights, settings.device, settings.precision)
            options = ConvertOptions(temporal=nr.temporal, motion=motion, scene_cut_threshold=nr.scene_cut_threshold,
                                     overwrite=True, status_interval=1e9, enhance=enhance)
        return NeuralRenderStage(pipeline, options)

    def _frame_generation_stage(self, fg: FrameGen, settings: Settings, *, floating=False):
        from ..framegen_video import FrameGenOptions
        from ..video_pipeline import FrameGenerationStage

        if not settings.fg_weights:
            raise ValueError("set the frame generation weights (mlxdlss-weights extract-fg) in Settings")
        backend = settings.resolved_backend("fg")
        options = FrameGenOptions(mode=fg.mode, factor=fg.factor, audio=fg.audio, overwrite=True, status_interval=1e9,
                                  backend=backend, mlxdlss=settings.mlxdlss_binary or None, mlxdlss_weights=str(Path(settings.fg_weights).expanduser()))
        generator = None
        if backend == "torch":
            precision = "fast" if settings.precision == "fast" else "reference"
            generator = self.cache.frame_generator(settings.fg_weights, settings.device, precision)
        return FrameGenerationStage(generator, options, floating=floating)

    # -- preview --------------------------------------------------------------------
    @staticmethod
    def _preview(source: Path, result: Path, folder: Path, *, height: int = 540, seconds: float = 12.0) -> Path | None:
        """Original | result side by side, at the result's frame rate, first ``seconds`` seconds."""
        try:
            ffmpeg = find_tool("ffmpeg", None)
            info = probe(result)
        except Exception:
            return None
        out = folder / "preview.mp4"
        graph = (f"[0:v]fps={info.frame_rate},scale=-2:{height}[a];[1:v]scale=-2:{height}[b];"
                 f"[a][b]hstack=inputs=2:shortest=1,format=yuv420p[v]")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-i", str(result),
                   "-filter_complex", graph, "-map", "[v]", "-an", "-t", str(seconds), "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
                   "-movflags", "+faststart", str(out)]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except (subprocess.CalledProcessError, OSError):
            return None
        return out
