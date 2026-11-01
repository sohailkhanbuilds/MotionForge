"""Experimental BVH writer driven by MediaPipe world landmarks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

# BVH hierarchy. Offsets are inferred from the first detected frame.
# Each joint's world orientation is estimated by aligning its rest child
# vectors with the current child vectors; local rotations are parent-relative.
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
    if joint == "Hips":
        return (frame[23, :3] + frame[24, :3]) * 0.5
    if joint == "Spine":
        return (frame[11, :3] + frame[12, :3]) * 0.5
    if joint == "LeftShoulder":
        return frame[11, :3]
    if joint == "RightShoulder":
        return frame[12, :3]
    idx = LANDMARK_INDEX.get(joint)
    return frame[idx, :3] if idx is not None else None


def _joint_positions(frame):
    return {name: _point(frame, name) for name, _, _ in HIERARCHY}


def _rotation_for_joint(name, rest, current):
    children = CHILDREN[name]
    pairs_a, pairs_b = [], []
    for child in children:
        a = rest[child] - rest[name]
        b = current[child] - current[name]
        if np.linalg.norm(a) > 1e-5 and np.linalg.norm(b) > 1e-5:
            pairs_a.append(a / np.linalg.norm(a))
            pairs_b.append(b / np.linalg.norm(b))
    if not pairs_a:
        return Rotation.identity()
    try:
        return Rotation.align_vectors(np.asarray(pairs_b), np.asarray(pairs_a))[0]
    except Exception:
        return Rotation.identity()


def export_bvh(frames: np.ndarray, fps: float, output_path: str | Path):
    """Write an approximate humanoid BVH file. Coordinates are converted to cm."""
    valid_frames = [f for f in frames if float(f[:, 3].max()) > 0.2]
    if not valid_frames:
        raise ValueError("No valid pose frames available for BVH export.")
    rest = _joint_positions(valid_frames[0])
    # Fill any missing joint positions using the rest pose to keep hierarchy valid.
    for key, value in rest.items():
        if value is None or not np.all(np.isfinite(value)):
            rest[key] = np.zeros(3, dtype=float)
    offsets = {}
    for name, parent, _ in HIERARCHY:
        offsets[name] = np.zeros(3) if parent is None else (rest[name] - rest[parent]) * 100.0

    lines = ["HIERARCHY"]
    def emit_joint(name, depth):
        pad = "  " * depth
        lines.append(f"{pad}{'ROOT' if PARENTS[name] is None else 'JOINT'} {name}")
        lines.append(f"{pad}{{")
        off = offsets[name]
        lines.append(f"{pad}  OFFSET {off[0]:.5f} {off[1]:.5f} {off[2]:.5f}")
        if PARENTS[name] is None:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
        for child in CHILDREN[name]:
            emit_joint(child, depth + 1)
        if not CHILDREN[name]:
            lines.append(f"{pad}  End Site")
            lines.append(f"{pad}  {{")
            lines.append(f"{pad}    OFFSET 0.00000 2.00000 0.00000")
            lines.append(f"{pad}  }}")
        lines.append(f"{pad}}}")
    emit_joint("Hips", 0)

    lines += ["MOTION", f"Frames: {len(frames)}", f"Frame Time: {1.0 / max(float(fps), 1.0):.8f}"]
    rest_rot = {name: _rotation_for_joint(name, rest, rest) for name, _, _ in HIERARCHY}
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
        values = [root_delta[0], root_delta[1], root_delta[2]]
        for name, _, _ in HIERARCHY:
            # BVH channel order is Z, X, Y.
            zxy = local_rot[name].as_euler("ZXY", degrees=True)
            if name == "Hips":
                values.extend(zxy.tolist())
            else:
                values.extend(zxy.tolist())
        lines.append(" ".join(f"{v:.5f}" for v in values))
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
