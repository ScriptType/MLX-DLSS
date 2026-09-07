"""Local paired temporal-quality probe using the original neural-rendering weights.

Private inputs and rendered artifacts stay in the caller-selected output directory.
Scores are computed before encoding; sampled resident memory is not total GPU memory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from mlxdlss.mlxdlss_stream import MLXDLSSStreamSession
from mlxdlss.temporal import FlowMotionEstimator
from mlxdlss.video import probe


def warped_effect_error(current, previous, source, previous_source, backward_uv, forward_uv):
    """Independent bilinear evaluator with a fixed input-derived visibility mask."""
    height, width = source.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    mx = xx + backward_uv[..., 0] * width
    my = yy + backward_uv[..., 1] * height
    reverse = cv2.remap(forward_uv, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = (backward_uv + reverse) * np.array([width, height], np.float32)
    warped_source = cv2.remap(previous_source, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask = ((mx >= 8) & (mx < width - 8) & (my >= 8) & (my < height - 8)
            & (np.square(fb).sum(-1) < 1) & (np.abs(source - warped_source).mean(-1) < 0.05))
    previous_effect = previous - previous_source
    warped = cv2.remap(previous_effect, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    error = np.abs(current - source - warped)
    return (float(error[mask].mean()) if mask.any() else None), float(mask.mean())


def aggregate_warp_metrics(errors, coverages):
    """Summarize optional per-frame scores without emitting non-standard NaN JSON."""
    valid = [float(error) for error in errors if error is not None and np.isfinite(error)]
    finite_coverage = [float(coverage) for coverage in coverages if np.isfinite(coverage)]
    return {
        "warped_effect_error": float(np.mean(valid)) if valid else None,
        "coverage": float(np.mean(finite_coverage)) if finite_coverage else 0.0,
        "valid_frames": len(valid),
    }


def encode_preview(frames, output, frame_rate, *, ffmpeg="ffmpeg"):
    """Encode a review copy at the exact source rate and publish it atomically."""
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 4 or values.shape[-1] != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("preview frames must be a non-empty finite NHWC RGB sequence")
    height, width = values.shape[1:3]
    raw = np.clip(values * 255 + 0.5, 0, 255).astype(np.uint8).tobytes()
    output = Path(output)
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    partial.unlink(missing_ok=True)
    try:
        subprocess.run(
            [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(frame_rate), "-i", "-", "-an",
             "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(partial)],
            input=raw,
            capture_output=True,
            check=True,
        )
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)


def write_report(path, report):
    """Write strict JSON atomically so interrupted runs leave no plausible report."""
    path = Path(path)
    partial = path.with_name(f"{path.stem}.partial{path.suffix}")
    partial.unlink(missing_ok=True)
    try:
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        partial.write_text(payload)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def run_case(frames, model, binary, *, temporal, motion, robust_motion, scale, detail, output, frame_rate=60):
    height, width = frames[0].shape[:2]
    session = MLXDLSSStreamSession(model, width, height, mlxdlss=binary, temporal=temporal,
                                  motion=motion, robust_motion=robust_motion, processing_scale=scale,
                                  detail_strength=detail)
    rendered = []
    peak_rss_kib = 0
    started = time.perf_counter()
    try:
        for frame in frames:
            rendered.append(session.process_frame(frame))
            memory = subprocess.run(["ps", "-o", "rss=", "-p", str(session.process.pid)], capture_output=True, text=True)
            if memory.returncode == 0 and memory.stdout.strip():
                peak_rss_kib = max(peak_rss_kib, int(memory.stdout.strip()))
        summary = session.close()
    finally:
        session.abort()
    seconds = time.perf_counter() - started
    values = np.stack(rendered)
    changes = values - np.asarray(frames)
    magnitude = float(np.abs(changes).mean())
    highpass = float(np.mean([np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5)).mean() for c in changes]))
    encode_preview(values, output, frame_rate)
    return values, {"frames": len(frames), "seconds_with_flow_and_rss_sampling": seconds,
                    "sampled_renderer_peak_rss_kib": peak_rss_kib, "scene_cuts": session.scene_cuts,
                    "mean_abs_effect": magnitude, "highpass_effect": highpass, "stream": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mlxdlss", required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--freeze-frames", type=int, default=32)
    parser.add_argument("--processing-scale", type=float, default=1)
    parser.add_argument("--detail-strength", type=float, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 2 or args.freeze_frames < 2:
        parser.error("at least two moving and frozen frames are required")
    info = probe(args.input)
    result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.input), "-frames:v", str(args.frames), "-an",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    frames = np.frombuffer(result.stdout, np.uint8).reshape(-1, info.height, info.width, 3).astype(np.float32) / 255
    if len(frames) < 2:
        parser.error("input contains fewer than two frames")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    estimator = FlowMotionEstimator()
    evaluation_flows = [(estimator(frames[i], frames[i - 1]), estimator(frames[i - 1], frames[i])) for i in range(1, len(frames))]
    report = {"extent": [info.width, info.height], "source_fps": info.fps,
              "source_frame_rate": str(info.frame_rate), "preview_fps": info.fps,
              "preview_frame_rate": str(info.frame_rate),
              "processing_scale": args.processing_scale, "detail_strength": args.detail_strength, "cases": {}}
    cases = [("independent", False, "zero", False), ("history-zero", True, "zero", False),
             ("history-flow", True, "flow", False), ("history-guarded", True, "flow", True)]
    for name, temporal, motion, robust in cases:
        outputs, stats = run_case(frames, args.model, args.mlxdlss, temporal=temporal, motion=motion, robust_motion=robust,
                                  scale=args.processing_scale, detail=args.detail_strength, output=args.output_dir / f"{name}.mp4",
                                  frame_rate=info.frame_rate)
        errors, coverage = [], []
        for i, (b, f) in enumerate(evaluation_flows, 1):
            error, covered = warped_effect_error(outputs[i], outputs[i - 1], frames[i], frames[i - 1], b, f)
            errors.append(error); coverage.append(covered)
        stats.update(aggregate_warp_metrics(errors, coverage), per_frame_error=errors)
        report["cases"][name] = stats
        print(json.dumps({"case": name, **{k: v for k, v in stats.items() if k != "per_frame_error"}}, allow_nan=False), flush=True)
    frozen = [frames[0]] * args.freeze_frames
    for name, temporal, motion, robust in (cases[0], cases[2], cases[3]):
        outputs, stats = run_case(frozen, args.model, args.mlxdlss, temporal=temporal, motion=motion, robust_motion=robust,
                                  scale=args.processing_scale, detail=args.detail_strength, output=args.output_dir / f"frozen-{name}.mp4",
                                  frame_rate=info.frame_rate)
        stats.update(consecutive_error=float(np.abs(np.diff(outputs, axis=0)).mean()),
                     last_vs_first=float(np.abs(outputs[-1] - outputs[0]).mean()))
        report["cases"][f"frozen-{name}"] = stats
        print(json.dumps({"case": f"frozen-{name}", **stats}, allow_nan=False), flush=True)
    write_report(args.output_dir / "metrics.json", report)


if __name__ == "__main__":
    main()
