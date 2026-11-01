"""Confidence-aware experimental BVH writer driven by MediaPipe landmarks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HIERARCHY = [
    ("Hips", None, 23),
    ("Spine", "Hips", 11), ("Neck", "Spine", 0), ("Head", "Neck", 0),
    ("LeftShoulder", "Spine", 11), ("LeftElbow", "LeftShoulder", 13), ("LeftWrist", "LeftElbow", 15),
    ("RightShoulder", "Spine", 12), ("RightElbow", "RightShoulder", 14), ("RightWrist", "RightElbow", 16),
    ("LeftUpLeg", "Hips", 25), ("LeftLeg", "LeftUpLeg", 27), ("LeftFoot", "LeftLeg", 31),
    ("RightUpLeg", "Hips", 26), ("RightLeg", "RightUpLeg", 28), ("RightFoot", "RightLeg", 32),
]
PARENTS = {name: parent for name, parent, _ in HIERARCHY}
LANDMARK_INDEX = {name: idx for name, _, idx in HIERARCHY if idx is not None}
CHILDREN = {name: [] for name, _, _ in HIERARCHY}
for name, parent, _ in HIERARCHY:
    if parent:
        CHILDREN[parent].append(name)


def _point(frame: np.ndarray, joint: str):
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


def _joint_confidence(frame: np.ndarray, joint: str) -> float:
    """Return conservative confidence for joints synthesized from landmarks."""
    if joint == "Hips":
        indices = (23, 24)
    elif joint == "Spine":
        indices = (11, 12)
    elif joint == "Neck":
        indices = (11, 12, 0)
    elif joint == "LeftShoulder":
        indices = (11,)
    elif joint == "RightShoulder":
        indices = (12,)
    else:
        indices = (LANDMARK_INDEX[joint],)
    return float(np.min(frame[list(indices), 3]))


def _joint_positions(frame: np.ndarray):
    return {name: _point(frame, name) for name, _, _ in HIERARCHY}


def _rotation_for_joint(name, rest, current):
    rest_vectors, current_vectors = [], []
    for child in CHILDREN[name]:
        a = rest[child] - rest[name]
        b = current[child] - current[name]
        if np.linalg.norm(a) > 1e-5 and np.linalg.norm(b) > 1e-5:
            rest_vectors.append(a / np.linalg.norm(a))
            current_vectors.append(b / np.linalg.norm(b))
    if not rest_vectors:
        return Rotation.identity()
    try:
        # Align all visible child directions together where possible. This is
        # more stable for the pelvis and chest than choosing a single limb.
        return Rotation.align_vectors(
            np.asarray(current_vectors), np.asarray(rest_vectors)
        )[0]
    except (ValueError, np.linalg.LinAlgError):
        return Rotation.identity()


def _clean_positions(frame: np.ndarray, rest: dict[str, np.ndarray], threshold: float = 0.35):
    current = _joint_positions(frame)
    for name in current:
        value = current[name]
        if (
            value is None
            or not np.all(np.isfinite(value))
            or _joint_confidence(frame, name) < threshold
        ):
            current[name] = rest[name]
    return current


def export_bvh(frames: np.ndarray, fps: float, output_path: str | Path):
    """Write a generic humanoid BVH with confidence-gated joints.

    Root screen-plane translation is approximate; true camera-space depth is
    not reconstructed by this single-camera MVP.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[1] < 33 or frames.shape[2] < 6:
        raise ValueError("frames must have shape (frames, 33, 6)")
    valid = [f for f in frames if float(f[:, 3].max()) > 0.2]
    if not valid:
        raise ValueError("No valid pose frames available for BVH export.")

    # A short median reference is less sensitive to a noisy first detected
    # frame, which otherwise produces visibly distorted bone offsets.
    reference_frames = valid[: min(len(valid), max(3, int(round(fps * 0.5))))]
    reference = np.median(np.stack(reference_frames), axis=0)
    rest = _joint_positions(reference)
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
        kind = "ROOT" if PARENTS[name] is None else "JOINT"
        lines.extend([f"{pad}{kind} {name}", f"{pad}{{"])
        off = offsets[name]
        lines.append(f"{pad}  OFFSET {off[0]:.5f} {off[1]:.5f} {off[2]:.5f}")
        if PARENTS[name] is None:
            lines.append(f"{pad}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{pad}  CHANNELS 3 Zrotation Xrotation Yrotation")
        for child in CHILDREN[name]:
            emit_joint(child, depth + 1)
        if not CHILDREN[name]:
            lines.extend([
                f"{pad}  End Site", f"{pad}  {{",
                f"{pad}    OFFSET 0.00000 2.00000 0.00000", f"{pad}  }}",
            ])
        lines.append(f"{pad}}}")

    emit_joint("Hips", 0)
    frame_time = 1.0 / max(float(fps), 1.0)
    lines.extend(["MOTION", f"Frames: {len(frames)}", f"Frame Time: {frame_time:.8f}"])

    hip_screen = np.stack([
        (f[23, 4:6] + f[24, 4:6]) * 0.5
        for f in reference_frames
        if float(f[[23, 24], 3].min()) > 0.2
    ])
    rest_hip_screen = np.median(hip_screen, axis=0) if len(hip_screen) else None

    for frame in frames:
        current = _clean_positions(frame, rest)
        world_rot = {
            name: _rotation_for_joint(name, rest, current)
            for name, _, _ in HIERARCHY
        }
        local_rot = {}
        for name, parent, _ in HIERARCHY:
            local_rot[name] = (
                world_rot[name] if parent is None
                else world_rot[parent].inv() * world_rot[name]
            )

        root_delta = (current["Hips"] - rest["Hips"]) * 100.0
        # MediaPipe world landmarks are hip-centred. Estimate only screen-plane
        # travel from normalized image coordinates; depth remains unavailable.
        if rest_hip_screen is not None and _joint_confidence(frame, "Hips") > 0.2:
            screen_hip = (frame[23, 4:6] + frame[24, 4:6]) * 0.5
            if np.all(np.isfinite(screen_hip)):
                delta = screen_hip - rest_hip_screen
                root_delta = np.array([delta[0] * 170.0, -delta[1] * 170.0, 0.0])
        values = root_delta.tolist()
        for name, _, _ in HIERARCHY:
            values.extend(local_rot[name].as_euler("ZXY", degrees=True).tolist())
        lines.append(" ".join(f"{v:.5f}" for v in values))

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
