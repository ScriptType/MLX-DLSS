"""Executes a job's effect chain with the package's own pipelines.

Images take neural rendering (torch); videos run the selected effects in one
decode/effect/encode stream, using torch or the Swift Metal frame servers. A side-by-side preview is rendered with
FFmpeg when a video job is done.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from ..video import find_tool, probe
from .effects import FrameGen, NeuralRender, parse_effects
from .jobs import Job
from .settings import Settings

Report = Callable[[str, float, int, int | None], None]


class ModelCache:
    """Loaded pipelines keyed by their configuration; one load per process."""

    def __init__(self):
        self._lock = threading.Lock()
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
        settings = self.settings_provider()
        job.backend = "/".join(sorted({settings.resolved_backend(e.get("kind", "nr")) for e in job.effects}))
        effects = parse_effects(job.effects)
        source = folder / ("input" + Path(job.input_name).suffix.lower())
        if job.kind == "image":
            return [self._image(source, folder, effects, settings, report)]
        return self._video(source, folder, effects, settings, report, should_stop, job)

    # -- images -------------------------------------------------------------------
    def _image(self, source: Path, folder: Path, effects, settings: Settings, report: Report) -> Path:
        from PIL import Image

        nr = next(e for e in effects if isinstance(e, NeuralRender))
        out = folder / "result.png"
        if settings.resolved_backend("nr") == "mlxdlss":
            # Metal: the whole frame stays on the GPU; the PyTorch graph below keeps the
            # activations of the entire frame in memory and needs tens of GB at 4K.
            from ..mlxdlss_stream import find_mlxdlss

            if not settings.nr_model:
                raise ValueError("set the .dlssmodel package for the Metal backend in Settings")
            report("neural rendering", 0.2, 0, 1)
            command = [find_mlxdlss(settings.mlxdlss_binary or None), "render-image", str(source), str(Path(settings.nr_model).expanduser()),
                       "--output", str(out), "--execution", "metal-fused", "--precision", "float16", "--profile", nr.profile,
                       "--processing-scale", f"{nr.processing_scale:g}", "--detail-strength", f"{nr.detail_strength:g}",
                       "--colour-strength", f"{nr.colour_strength:g}", "--detail-radius", f"{nr.detail_radius:g}", "--intensity", f"{nr.intensity:g}"]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f"mlxdlss render-image failed: {completed.stderr.strip()[-500:]}")
            report("done", 1.0, 1, 1)
            return out
        if not settings.nr_weights:
            raise ValueError("set the neural rendering weights (logical safetensors) in Settings")
        report("loading the network", 0.05, 0, None)
        pipeline = self.cache.neural_rendering(settings.nr_weights, settings.device, settings.precision)
        image = np.asarray(Image.open(source).convert("RGB"), np.float32) / 255.0
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
        from ..video import ConvertOptions
        from ..video_pipeline import NeuralRenderStage, run_video

        stages = []
        for effect in effects:
            if should_stop():
                raise Cancelled()
            if isinstance(effect, NeuralRender):
                stages.append(self._neural_rendering_stage(effect, settings))
            else:
                stages.append(self._frame_generation_stage(effect, settings, floating=len(effects) > 1))
        name = " → ".join("neural rendering" if isinstance(e, NeuralRender) else f"frame generation x{e.factor}" for e in effects)
        def progress(done, total):
            fraction = done / total if total else 0.0
            report(f"{name} {done}/{total if total is not None else '?'}", min(0.97, fraction * 0.97), done, total)
        target = folder / "result.mp4"
        run_video(source, target, stages, ConvertOptions(overwrite=True, status_interval=1e9),
                  log=lambda _m: None, progress=progress, should_stop=should_stop)
        for stage in stages:
            if isinstance(stage, NeuralRenderStage):
                options = stage.options
                job.diagnostics = {"temporal": options.temporal, "motion": options.motion if options.temporal else "none",
                                   "processing_scale": options.enhance.get("processing_scale", 1.0), "scene_cuts": stage.scene_cuts}
        if should_stop():
            raise Cancelled()
        report("rendering the preview", 0.98, job.frames_done, job.frames_total)
        preview = self._preview(source, target, folder)
        if preview is not None:
            job.preview = preview.name
        return [target]

    def _neural_rendering_stage(self, nr: NeuralRender, settings: Settings):
        from ..video import ConvertOptions
        from ..video_pipeline import NeuralRenderStage

        backend = settings.resolved_backend("nr")
        enhance = {"profile": nr.profile, "processing_scale": nr.processing_scale, "detail_strength": nr.detail_strength,
                   "colour_strength": nr.colour_strength, "detail_radius": nr.detail_radius, "intensity": nr.intensity}
        if backend == "mlxdlss":
            if not settings.nr_model:
                raise ValueError("set the .dlssmodel package for the Metal backend in Settings")
            options = ConvertOptions(backend="mlxdlss", model_package=str(Path(settings.nr_model).expanduser()), mlxdlss=settings.mlxdlss_binary or None,
                                     temporal=nr.temporal, overwrite=True, status_interval=1e9, enhance=enhance)
            pipeline = None
        else:
            if not settings.nr_weights:
                raise ValueError("set the neural rendering weights (logical safetensors) in Settings")
            pipeline = self.cache.neural_rendering(settings.nr_weights, settings.device, settings.precision)
            options = ConvertOptions(temporal=nr.temporal, overwrite=True, status_interval=1e9, enhance=enhance)
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
