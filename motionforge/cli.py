"""Command-line entry point for MotionForge."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from motionforge.pose import extract_video
from motionforge.smoothing import smooth_landmarks
from motionforge.bvh import export_bvh
from motionforge.diagnostics import tracking_quality
from motionforge.preview import render_pose_preview
from motionforge.object_tracking import track_object, set_pose_frames

def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a human video into BVH motion capture data.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output-dir", default="output", help="Directory for BVH, CSV, preview and metadata")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=0, help="MediaPipe pose model (0 is lightest)")
    parser.add_argument("--smooth-window", type=int, default=5, help="Odd smoothing window; use 1 to disable")
    parser.add_argument("--no-preview", action="store_true", help="Skip annotated MP4 preview")
    parser.add_argument("--object-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Track one object selected in the first frame (pixel coordinates)")
    parser.add_argument("--object-type", choices=("ball", "box", "cube", "cylinder", "bottle", "cone"), default="ball", help="Primitive used in Blender object export")
    parser.add_argument("--object-name", default="TrackedObject", help="Name for the generated Blender object")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[MotionForge] Processing: {args.input}")
    timestamps, raw_frames, fps, size = extract_video(args.input, args.model_complexity)
    frames = smooth_landmarks(raw_frames, max(1, args.smooth_window))
    bvh_path, csv_path = out / "motionforge.bvh", out / "landmarks.csv"
    export_bvh(frames, fps, bvh_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        axes = ("world_x", "world_y", "world_z", "visibility", "image_x", "image_y")
        writer.writerow(["time_seconds"] + [f"lm{i}_{axis}" for i in range(33) for axis in axes])
        for t, frame in zip(timestamps, frames):
            writer.writerow([f"{t:.6f}"] + frame.reshape(-1).tolist())
    preview_path = None
    if not args.no_preview:
        preview_path = out / "pose_preview.mp4"
        render_pose_preview(args.input, frames, fps, preview_path)
    object_outputs = []
    if args.object_roi:
        set_pose_frames(frames)
        object_json, object_script = track_object(args.input, args.object_roi, fps, out, args.object_type, args.object_name)
        object_outputs = [object_json.name, object_script.name]
        print(f"[MotionForge] Object motion: {object_json.resolve()}")
        print(f"[MotionForge] Blender object script: {object_script.resolve()}")
    metadata = {
        "input": Path(args.input).name, "fps": fps, "frames": len(frames),
        "duration_seconds": round(len(frames) / fps, 3), "source_width": size[0],
        "source_height": size[1], "model_complexity": args.model_complexity,
        "smooth_window": args.smooth_window,
        "tracking_quality": tracking_quality(raw_frames),
        "outputs": [bvh_path.name, csv_path.name] + ([preview_path.name] if preview_path else []) + object_outputs,
        "notes": [
            "Single-camera approximate 3D pose estimation.",
            "MediaPipe world landmarks are hip-centred; root translation is estimated from 2D image position.",
            "True depth, occluded joints, foot contact and custom-rig retargeting are not solved in this MVP.",
            "Tracking visibility metrics are not a measure of 3D animation accuracy.",
            "BVH joint rotations are experimental and require validation in Blender.",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[MotionForge] Done. Frames: {len(frames)} | FPS: {fps:.2f}")
    print(f"[MotionForge] BVH: {bvh_path.resolve()}")
    print(f"[MotionForge] CSV: {csv_path.resolve()}")
    if preview_path:
        print(f"[MotionForge] Pose preview: {preview_path.resolve()}")
    return 0
