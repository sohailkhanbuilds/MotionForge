"""Command-line entry point for MotionForge."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from motionforge.pose import extract_video
from motionforge.smoothing import smooth_landmarks
from motionforge.bvh import export_bvh


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a human video into an experimental BVH animation.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-dir", default="output", help="Directory for BVH, CSV and metadata")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0, help="MediaPipe pose model (0 is lightest)")
    parser.add_argument("--smooth-window", type=int, default=5, help="Odd moving-average window; use 1 to disable")
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[MotionForge] Processing: {args.input}")
    timestamps, frames, fps, size = extract_video(args.input, args.model_complexity)
    frames = smooth_landmarks(frames, max(1, args.smooth_window))
    bvh_path = out / "motionforge.bvh"
    csv_path = out / "landmarks.csv"
    export_bvh(frames, fps, bvh_path)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["time_seconds"] + [f"lm{i}_{axis}" for i in range(33) for axis in ("x", "y", "z", "visibility")]
        writer.writerow(header)
        for t, frame in zip(timestamps, frames):
            writer.writerow([f"{t:.6f}"] + frame.reshape(-1).tolist())

    metadata = {
        "input": str(Path(args.input).name), "fps": fps, "frames": len(frames),
        "duration_seconds": round(len(frames) / fps, 3), "source_width": size[0],
        "source_height": size[1], "model_complexity": args.model_complexity,
        "smooth_window": args.smooth_window,
        "outputs": [bvh_path.name, csv_path.name],
        "notes": ["Single-camera approximate 3D pose estimation.", "BVH hierarchy and joint rotations are experimental; inspect and retarget in Blender."],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[MotionForge] Done. Frames: {len(frames)} | FPS: {fps:.2f}")
    print(f"[MotionForge] BVH: {bvh_path.resolve()}")
    print(f"[MotionForge] CSV: {csv_path.resolve()}")
    return 0
