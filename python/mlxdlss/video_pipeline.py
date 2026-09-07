"""Bounded frame stages shared by the CLI and the web effect chain."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .temporal import TemporalOptions, TemporalSession, prepare_temporal_frame, resolve_motion
from .video import DEFAULT_ENCODE_ARGS, PIXEL_FORMATS, VideoToolError, find_tool, probe


def _close(iterator):
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _float_frame(frame):
    if frame.dtype.kind == "u":
        return frame.astype(np.float32) / np.float32(np.iinfo(frame.dtype).max)
    return np.asarray(frame, np.float32)


def _prefetched(iterator, prepare, cancel):
    """Fetch upstream on the consumer; overlap only CPU preparation with rendering."""
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlxdlss-prepare")
    end = object()
    future = None
    try:
        raw = next(iterator, end)
        if raw is end:
            return
        future = executor.submit(prepare, raw, None)
        while future is not None:
            item = future.result()
            # A preceding FG stage may perform GPU work while yielding its next
            # frame. Keep that work on this thread, including across batch edges.
            raw = next(iterator, end)
            future = None if raw is end else executor.submit(prepare, raw, item.source)
            yield item
    finally:
        if future is not None:
            cancel()
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        _close(iterator)


class NeuralRenderStage:
    def __init__(self, pipeline, options):
        self.pipeline, self.options = pipeline, options
        self.scene_cuts = 0

    def apply(self, frames, width, height, *, prefetch=False, cancel=lambda: None):
        options, enhance = self.options, self.options.enhance
        stream = renderer = prepared = None
        try:
            if options.backend == "mlxdlss":
                from .mlxdlss_stream import MLXDLSSStreamSession

                if not options.model_package:
                    raise VideoToolError("backend 'mlxdlss' needs --model MODEL.dlssmodel")
                stream = MLXDLSSStreamSession(
                    options.model_package, width, height, temporal=options.temporal, motion=options.motion,
                    scene_cut_threshold=options.scene_cut_threshold, robust_motion=options.robust_motion,
                    blend_scale=options.blend_scale, mlxdlss=options.mlxdlss, execution=options.execution,
                    precision=options.precision, **enhance,
                )
                renderer = stream
            elif options.temporal:
                renderer = TemporalSession(self.pipeline, motion=options.motion, options=TemporalOptions(
                    robust_motion=options.robust_motion, blend_scale=options.blend_scale,
                    scene_cut_threshold=options.scene_cut_threshold, **enhance,
                ))
            if options.temporal:
                def prepare(raw, previous):
                    frame = _float_frame(raw)
                    if frame.shape != (height, width, 3) or not np.isfinite(frame).all():
                        raise ValueError(f"frame must be finite ({height}, {width}, 3)")
                    estimate = resolve_motion(renderer.motion, frame, previous, None, None,
                                              scene_cut_threshold=options.scene_cut_threshold,
                                              robust_motion=options.robust_motion)
                    return prepare_temporal_frame(frame, estimate, enhance.get("processing_scale", 1.0), packed_color=raw)
                def sequential():
                    previous = None
                    for raw in frames:
                        item = prepare(raw, previous)
                        previous = item.source
                        yield item
                if prefetch and options.prefetch:
                    prepared = _prefetched(frames, prepare, cancel)
                else:
                    prepared = sequential()
                for frame in prepared:
                    yield renderer._process_prepared(frame)
                self.scene_cuts = renderer.scene_cuts
            elif stream is not None:
                for frame in frames:
                    yield stream.process_frame(_float_frame(frame))
            else:
                finish_keys = {"detail_strength", "colour_strength", "detail_radius", "intensity"}
                finish = {k: v for k, v in enhance.items() if k in finish_keys}
                prepare_options = {k: v for k, v in enhance.items() if k not in finish_keys}
                pending = []
                def flush():
                    heads = self.pipeline.run_features_batch(np.stack([p.features for p in pending]))
                    for item, head in zip(pending, heads):
                        yield self.pipeline.finish(item, head, network_seconds=0, **finish).image
                for index, frame in enumerate(frames, start=options.start_frame):
                    pending.append(self.pipeline.prepare(_float_frame(frame), frame_index=index, **prepare_options))
                    if len(pending) == options.batch:
                        yield from flush()
                        pending.clear()
                if pending:
                    yield from flush()
            if stream is not None:
                stream.close()
        finally:
            if prepared is not None:
                _close(prepared)
            if renderer is not None:
                self.scene_cuts = renderer.scene_cuts
            if stream is not None:
                stream.abort()
            _close(frames)


class FrameGenerationStage:
    def __init__(self, generator, options, *, floating=False):
        self.generator, self.options, self.floating = generator, options, floating

    def apply(self, frames, width, height, *, prefetch=False, cancel=lambda: None):
        from .mlxdlss_stream import MLXDLSSFrameGenStream

        options = self.options
        batch, per_pair = max(1, int(options.batch)), options.factor - 1
        stream = None
        held, window = [], []
        def emit(generated):
            if len(generated) != len(held) * per_pair:
                raise VideoToolError("frame generation returned an incomplete batch")
            for index, original in enumerate(held):
                yield from generated[index * per_pair:(index + 1) * per_pair]
                yield original
            held.clear()
        try:
            if options.backend == "mlxdlss":
                stream = MLXDLSSFrameGenStream(
                    options.mlxdlss_weights, width, height, factor=options.factor,
                    precision=options.mlxdlss_precision, mlxdlss=options.mlxdlss, batch=batch,
                    format="f32" if self.floating else "u8",
                )
            for frame in frames:
                if self.floating:
                    frame = _float_frame(frame)
                if not window:
                    # Push before yielding, so a following NR stage can never outlive an unsent first frame.
                    if stream is not None:
                        stream.push(frame)
                    window.append(frame)
                    yield frame
                    continue
                held.append(frame)
                if stream is not None:
                    generated = stream.push(frame)
                    if generated:
                        yield from emit(generated)
                else:
                    window.append(frame)
                    if len(window) == batch + 1:
                        generated = self.generator.generate_pairs(window, options.factor, as_uint8=not self.floating)
                        yield from emit([g for pair in generated for g in pair])
                        window = [window[-1]]
            if stream is not None:
                yield from emit(stream.finish())
                stream.close()
            elif len(window) >= 2:
                generated = self.generator.generate_pairs(window, options.factor, as_uint8=not self.floating)
                yield from emit([g for pair in generated for g in pair])
        finally:
            if stream is not None:
                stream.abort()
            _close(frames)


def run_video(source, destination, stages, options, *, ffmpeg=None, ffprobe=None,
              log, progress=None, should_stop=None, progress_input=False):
    """Decode and encode once, keeping the selected effect order in memory."""
    source, destination = Path(source), Path(destination)
    if destination.exists() and not options.overwrite:
        raise VideoToolError(f"destination exists: {destination} (pass overwrite)")
    if options.pixel_format not in PIXEL_FORMATS:
        raise VideoToolError(f"pixel format must be one of {tuple(PIXEL_FORMATS)}")
    if options.batch < 1:
        raise VideoToolError("batch must be at least 1")
    tool, info = find_tool("ffmpeg", ffmpeg), probe(source, ffprobe=ffprobe)
    dtype, scale, bytes_per_pixel = PIXEL_FORMATS[options.pixel_format]
    expected = None if info.frame_count is None else max(0, info.frame_count - options.start_frame)
    if options.frame_limit is not None and expected is not None:
        expected = min(expected, options.frame_limit)
    output_expected, rate, audio, audio_ratio = expected, info.frame_rate, options.audio, 1.0
    for stage in stages:
        if isinstance(stage, FrameGenerationStage):
            fg = stage.options
            if output_expected is not None:
                output_expected = max(0, output_expected - 1) * fg.factor + int(output_expected > 0)
            if fg.mode == "fps":
                rate *= fg.factor
            audio = fg.audio
            if audio == "stretch":
                audio_ratio /= fg.factor
    decode = [tool, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(source), *options.decode_args]
    if options.start_frame:
        decode += ["-vf", f"select=gte(n\\,{options.start_frame})", "-fps_mode", "passthrough"]
    if options.frame_limit is not None:
        decode += ["-frames:v", str(options.frame_limit)]
    decode += ["-an", "-f", "rawvideo", "-pix_fmt", options.pixel_format, "-"]
    encode = [tool, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", options.pixel_format,
              "-s", f"{info.width}x{info.height}", "-r", str(rate), "-i", "-"]
    if audio != "none" and info.has_audio:
        if options.start_frame:
            encode += ["-ss", str(options.start_frame / info.fps)]
        encode += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0"]
        if audio == "stretch":
            from .framegen_video import atempo_chain
            encode += ["-filter:a", atempo_chain(audio_ratio), "-c:a", "aac", "-b:a", "192k"]
        else:
            encode += ["-c:a", "copy"]
        if options.start_frame or options.frame_limit is not None:
            encode += ["-shortest"]
    else:
        encode += ["-map", "0:v:0"]
    encode += list(DEFAULT_ENCODE_ARGS if options.encode_args is None else options.encode_args) + [str(destination)]
    decoder = encoder = None
    iterators = []
    frames_in = frames_out = 0
    cancelled = False
    started = last_status = time.perf_counter()
    with tempfile.TemporaryFile() as decode_error, tempfile.TemporaryFile() as encode_error:
        def abort_decode():
            if decoder is not None and decoder.poll() is None:
                decoder.kill()
        def read_frames():
            nonlocal frames_in, cancelled
            while True:
                if should_stop is not None and should_stop():
                    cancelled = True
                    break
                chunk = decoder.stdout.read(info.width * info.height * bytes_per_pixel)
                if not chunk:
                    break
                if len(chunk) != info.width * info.height * bytes_per_pixel:
                    raise VideoToolError("ffmpeg returned a truncated RGB frame")
                frames_in += 1
                if progress_input and progress is not None:
                    progress(frames_in, expected)
                yield np.frombuffer(chunk, dtype=dtype).reshape(info.height, info.width, 3)
        try:
            decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=decode_error)
            encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=encode_error)
            frames = read_frames()
            iterators.append(frames)
            for stage in stages:
                frames = stage.apply(frames, info.width, info.height, prefetch=True, cancel=abort_decode)
                iterators.append(frames)
            for frame in frames:
                if not progress_input and should_stop is not None and should_stop():
                    cancelled = True
                    break
                if frame.dtype != np.dtype(dtype):
                    frame = (np.clip(_float_frame(frame), 0, 1) * np.float32(scale) + 0.5).astype(dtype)
                encoder.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))
                frames_out += 1
                if not progress_input and progress is not None:
                    progress(frames_out, output_expected)
                now = time.perf_counter()
                if now - last_status >= options.status_interval:
                    log(f"STATUS frames {frames_out}/{output_expected if output_expected is not None else '?'} {frames_out / (now - started):.2f} fps")
                    last_status = now
            if cancelled:
                abort_decode()
            for iterator in reversed(iterators):
                _close(iterator)
            encoder.stdin.close()
            decoder.wait(); encoder.wait()
            for process, error, name in ((decoder, decode_error, "decode"), (encoder, encode_error, "encode")):
                if process.returncode and not (process is decoder and cancelled):
                    error.seek(0)
                    raise VideoToolError(f"ffmpeg {name} failed: {error.read().decode(errors='replace')[-2000:]}")
        finally:
            abort_decode()
            try:
                for iterator in reversed(iterators):
                    _close(iterator)
            finally:
                for process in (decoder, encoder):
                    if process is None:
                        continue
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    for pipe in (process.stdin, process.stdout):
                        if pipe is not None:
                            try:
                                pipe.close()
                            except OSError:
                                pass
    seconds = time.perf_counter() - started
    log(f"DONE {frames_in} -> {frames_out} frames in {seconds:.1f} s -> {destination}")
    return info, frames_in, frames_out, seconds, rate
