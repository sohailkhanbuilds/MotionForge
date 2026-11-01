"""Experimental BVH writer driven by MediaPipe pose landmarks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HIERARCHY = [
    ("Hips", None, 23),
    ("Spine", "Hips", 11), ("Neck", "Spine", 0), ("Head", "Neck", None),
    ("LeftShoulder", "Spine", 13), ("LeftElbow", "LeftShoulder", 15), ("LeftWrist", "LeftElbow", None),
    ("RightShoulder", "Spine", 14), ("RightElbow", "RightShoulder", 16), ("RightWrist", "RightElbow", None),
    ("LeftUpLeg", "Hips", 25), ("LeftLeg", "LeftUpLeg", 27), ("LeftFoot", "LeftLeg", 31),
    ("RightUpLeg", "Hips", 26), ("RightLeg", "RightUpLeg", 28), ("RightFoot", "RightLeg", 32),
]
PARENTS = {name: parent for name, parent, _ in HIERARCHY}
LANDMARK_INDEX = {name: idx for name, _, idx in HIERARCHY if idx is not None}
CHILDREN = {name: [] for name, _, _ in HIERARCHY}
for name, parent, _ in HIERARCHY:
    if parent:
        CHILDREN[parent].append(name)


def _point(frame, joint):
    shoulders = (frame[11, :3] + frame[12, :3]) * 0.5
    if joint == "Hips":
        return (frame[23, :3] + frame[24, :3]) * 0.5
    if joint == "Spine":
        return shoulders
    if joint == "Neck":
        return shoulders * 0.65 + frame[0, :3] * 0.35
    if joint == "LeftShoulder":
        return frame[11, :3]
    if joint == "RightShoulder":
        return frame[12, :3]
    idx = LANDMARK_INDEX.get(joint)
    return frame[idx, :3] if idx is not None else None


def _joint_positions(frame):
    return {name: _point(frame, name) for name, _, _ in HIERARCHY}


def _rotation_for_joint(name, rest, current):
    a_vectors, b_vectors = [], []
    for child in CHILDREN[name]:
        a = rest[child] - rest[name]
        b = current[child] - current[name]
        if np.linalg.norm(a) > 1e-5 and np.linalg.norm(b) > 1e-5:
            a_vectors.append(a / np.linalg.norm(a))
            b_vectors.append(b / np.linalg.norm(b))
    if not a_vectors:
        return Rotation.identity()
    try:
        return Rotation.align_vectors(np.asarray(b_vectors), np.asarray(a_vectors))[0]
    except Exception:
        return Rotation.identity()


def export_bvh(frames: np.ndarray, fps: float, output_path: str | Path):
    """Write a generic humanoid BVH. Root screen-plane translation is approximate."""
    valid = [f for f in frames if float(f[:, 3].max()) > 0.2]
    if not valid:
        raise ValueError("No valid pose frames available for BVH export.")
    rest = _joint_positions(valid[0])
    for name, value in rest.items():
        if value is None or not np.all(np.isfinite(value)):
            rest[name] = np.zeros(3, dtype=float)
    offsets = {
        name: np.zeros(3) if parent is None else (rest[name] - rest[parent]) * 100.0
        for name, parent, _ in HIERARCHY
    }
    lines = ["HIERARCHY"]

    def emit_joint(name, depth):
        pad = "  " * depth
        lines.extend([f"{pad}{'ROOT' if PARENTS[name] is None else 'JOINT'} {name}", f"{pad}{{"])
        off = offsets[name]
        lines.append(f"{pad}  OFFSET {off[0]:.5f} {off[1]:.5f} {off[2]:.5f}")
        if PARENTS[name] is None:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
        for child in CHILDREN[name]:
            emit_joint(child, depth + 1)
        if not CHILDREN[name]:
            lines.extend([f"{pad}  End Site", f"{pad}  {{", f"{pad}    OFFSET 0.00000 2.00000 0.00000", f"{pad}  }}"])
        lines.append(f"{pad}}}")

    emit_joint("Hips", 0)
    frame_time = 1.0 / max(float(fps), 1.0)
    lines.extend(["MOTION", f"Frames: {len(frames)}", f"Frame Time: {frame_time:.8f}"])
    rest_hip_screen = (valid[0][23, 4:6] + valid[0][24, 4:6]) * 0.5 if frames.shape[2] >= 6 else None

    for frame in frames:
        current = _joint_positions(frame)
        for name in current:
            if current[name] is None or not np.all(np.isfinite(current[name])):
                current[name] = rest[name]
        world_rot = {name: _rotation_for_joint(name, rest, current) for name, _, _ in HIERARCHY}
        local_rot = {}
        for name, parent, _ in HIERARCHY:
            local_rot[name] = world_rot[name] if parent is None else world_rot[parent].inv() * world_rot[name]

        root_delta = (current["Hips"] - rest["Hips"]) * 100.0
        # MediaPipe world landmarks are hip-centred, so estimate root translation
        # from normalized image-space hip position. This is only a 2D screen-plane
        # approximation (170 cm per image-height); it cannot recover true depth.
        if rest_hip_screen is not None:
            hip_screen = (frame[23, 4:6] + frame[24, 4:6]) * 0.5
            if np.all(np.isfinite(hip_screen)) and float(frame[[23, 24], 3].min()) > 0.2:
                delta = hip_screen - rest_hip_screen
                root_delta = np.array([delta[0] * 170.0, -delta[1] * 170.0, 0.0])
        values = root_delta.tolist()
        for name, _, _ in HIERARCHY:
            values.extend(local_rot[name].as_euler("ZXY", degrees=True).tolist())
        lines.append(" ".join(f"{v:.5f}" for v in values))
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
