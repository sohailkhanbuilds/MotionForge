"""Single-object 2D tracking and Blender primitive animation export.

This MVP tracks a user-selected bounding box with OpenCV CSRT. It estimates
screen-plane motion only; it does not reconstruct object geometry or true depth.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


def _make_tracker():
    factories = []
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        factories.append(legacy.TrackerCSRT_create)
    if hasattr(cv2, "TrackerCSRT_create"):
        factories.append(cv2.TrackerCSRT_create)
    if hasattr(cv2, "TrackerKCF_create"):
        factories.append(cv2.TrackerKCF_create)
    if not factories:
        raise RuntimeError("No OpenCV CSRT/KCF tracker is available. Install opencv-contrib-python.")
    return factories[0]()


def track_object(video_path, roi, fps, output_dir, object_type="ball", object_name="TrackedObject", attach_threshold=0.08):
    """Track an ROI and export normalized 2D trajectory + Blender script.

    roi is (x, y, width, height) in pixels on the first video frame.
    """
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError("Object ROI width and height must be positive.")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video for object tracking: {video_path}")
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read the first frame for object tracking.")
    height, width = first.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        cap.release()
        raise ValueError(f"ROI {roi} is outside the first frame ({width}x{height}).")
    tracker = _make_tracker()
    if not tracker.init(first, (x, y, w, h)):
        cap.release()
        raise RuntimeError("Could not initialize object tracker. Try a tighter ROI with a clear object boundary.")

    frames = []
    index = 0
    current_box = (x, y, w, h)
    try:
        while True:
            if index == 0:
                ok_frame, frame = True, first
            else:
                ok_frame, frame = cap.read()
            if not ok_frame:
                break
            if index > 0:
                success, box = tracker.update(frame)
                if success:
                    current_box = tuple(float(v) for v in box)
                    confidence = 1.0
                else:
                    confidence = 0.0
            else:
                confidence = 1.0
            bx, by, bw, bh = current_box
            frames.append({
                "frame": index,
                "time": round(index / max(float(fps), 1.0), 6),
                "center_x": float(np.clip((bx + bw / 2) / width, 0, 1)),
                "center_y": float(np.clip((by + bh / 2) / height, 0, 1)),
                "width": float(bw / width),
                "height": float(bh / height),
                "tracked": bool(confidence),
            })
            index += 1
    finally:
        cap.release()

    # Proximity-based interaction hints: compare object center to wrist image landmarks.
    interaction = []
    for i, item in enumerate(frames):
        if i >= len(_POSE_FRAMES):
            break
        pose = _POSE_FRAMES[i]
        for side, wrist_idx in (("left", 15), ("right", 16)):
            lm = pose[wrist_idx]
            if lm[3] >= 0.35:
                distance = float(np.hypot(item["center_x"] - lm[4], item["center_y"] - lm[5]))
                if distance <= attach_threshold:
                    interaction.append({"frame": i, "time": item["time"], "hand": side, "distance": round(distance, 5)})
                    break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "object_name": object_name, "object_type": object_type,
        "source_width": width, "source_height": height, "fps": float(fps),
        "coordinate_note": "Normalized 2D image coordinates; depth and physical scale are not reconstructed.",
        "interaction_note": "Hand proximity is a heuristic hint, not verified grasp/contact.",
        "frames": frames, "hand_proximity_events": interaction,
    }
    json_path = output_dir / "object_motion.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    script_path = output_dir / "object_animation.py"
    script_path.write_text(_blender_script(payload), encoding="utf-8")
    return json_path, script_path


# Set by CLI immediately before calling track_object. Kept module-local to avoid
# coupling object tracking to MediaPipe's runtime object model.
_POSE_FRAMES = []


def set_pose_frames(frames):
    global _POSE_FRAMES
    _POSE_FRAMES = frames


def _blender_script(payload):
    data = json.dumps(payload)
    return '''"""Run this script inside Blender's Scripting workspace."""
import bpy, json
DATA = json.loads(r''' + repr(data) + ''')
kind = DATA["object_type"].lower()
name = DATA["object_name"]
if kind in ("ball", "sphere"):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16)
elif kind in ("box", "cube"):
    bpy.ops.mesh.primitive_cube_add()
elif kind in ("cylinder", "bottle"):
    bpy.ops.mesh.primitive_cylinder_add(vertices=24)
elif kind in ("cone",):
    bpy.ops.mesh.primitive_cone_add(vertices=24)
else:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16)
obj = bpy.context.object
obj.name = name
obj["motionforge_source"] = "approximate 2D tracked path"
# Map image X/Y to Blender X/Z. Y depth is intentionally held at zero.
for item in DATA["frames"]:
    if not item["tracked"]:
        continue
    frame = int(item["frame"]) + 1
    x = (item["center_x"] - 0.5) * 4.0
    z = (0.5 - item["center_y"]) * 4.0
    obj.location = (x, 0.0, z)
    obj.keyframe_insert(data_path="location", frame=frame)
scene = bpy.context.scene
scene.render.fps = max(1, int(round(DATA["fps"])))
scene.frame_end = max(1, len(DATA["frames"]))
print("MotionForge object animation created:", obj.name)
print("Note: 2D screen-plane animation only; no true depth or verified grasp parenting.")
''' 
