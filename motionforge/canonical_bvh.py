"""Canonical retargeted-joint export to the experimental BVH writer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .bvh import export_bvh
from .skeleton import JOINTS, SkeletonError

JOINT_INDEX = {joint.name: index for index, joint in enumerate(JOINTS)}


def canonical_to_landmarks(frames: np.ndarray) -> np.ndarray:
    """Adapt (N, 16, 3) canonical joints into the legacy 33x6 BVH input.

    Unused MediaPipe landmarks are zero-confidence. This adapter preserves the
    canonical joints represented by the current BVH hierarchy; it does not
    invent hand/finger joints or camera depth.
    """
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(JOINTS), 3):
        raise SkeletonError(
            f"frames must have shape (N, {len(JOINTS)}, 3) in canonical joint order."
        )
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise SkeletonError("frames must be non-empty and finite.")
    result = np.zeros((len(values), 33, 6), dtype=np.float64)
    result[:, :, 3] = 0.0
    # Explicit landmark mappings for the joints consumed by bvh._point().
    mapping = {
        11: "LeftShoulder", 12: "RightShoulder",
        13: "LeftElbow", 14: "RightElbow",
        15: "LeftWrist", 16: "RightWrist",
        23: "Hips", 24: "Hips",
        25: "LeftUpLeg", 26: "RightUpLeg",
        27: "LeftLeg", 28: "RightLeg",
        31: "LeftFoot", 32: "RightFoot",
    }
    for landmark, joint_name in mapping.items():
        result[:, landmark, :3] = values[:, JOINT_INDEX[joint_name], :]
        result[:, landmark, 3] = 1.0
    # Neck = 0.65 * shoulder midpoint + 0.35 * landmark 0.
    shoulder_mid = (
        values[:, JOINT_INDEX["LeftShoulder"], :]
        + values[:, JOINT_INDEX["RightShoulder"], :]
    ) * 0.5
    result[:, 0, :3] = (
        values[:, JOINT_INDEX["Neck"], :] - 0.65 * shoulder_mid
    ) / 0.35
    result[:, 0, 3] = 1.0
    # Map the canonical head centre to both ear landmarks so the BVH writer
    # can reconstruct a distinct head point above the neck.
    for landmark in (7, 8):
        result[:, landmark, :3] = values[:, JOINT_INDEX["Head"], :]
        result[:, landmark, 3] = 1.0
    return result


def export_canonical_bvh(
    frames: np.ndarray, fps: float, output_path: str | Path
) -> Path:
    """Export retargeted canonical frames through the existing BVH writer."""
    adapted = canonical_to_landmarks(frames)
    export_bvh(adapted, fps, output_path)
    return Path(output_path)
