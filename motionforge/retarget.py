"""Length-aware retargeting for the canonical MotionForge humanoid skeleton.

This first retargeter preserves observed bone directions while fitting them to
target bone lengths. It is a geometric baseline, not a full IK/constraint solver.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from motionforge.skeleton import JOINTS, SkeletonError


def retarget_frame(
    source_positions: Mapping[str, np.ndarray],
    target_bone_lengths: Mapping[str, float],
    root_position: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit one source pose to target lengths, returning target joint positions.

    target_bone_lengths is keyed by child joint name (e.g. LeftElbow).
    The root is placed at root_position or at the source Hips position.
    Missing/degenerate source bone directions raise an error rather than
    silently producing NaNs or fabricated joint orientations.
    """
    root_value = source_positions.get("Hips") if root_position is None else root_position
    root = np.asarray(root_value, dtype=np.float64)
    if root.shape != (3,) or not np.all(np.isfinite(root)):
        raise SkeletonError("root_position must be a finite 3D vector.")
    output: dict[str, np.ndarray] = {"Hips": root.copy()}

    for joint in JOINTS:
        if joint.parent is None:
            continue
        if joint.name not in source_positions or joint.parent not in source_positions:
            raise SkeletonError(f"Missing source position for {joint.name!r} or its parent.")
        start = np.asarray(source_positions[joint.parent], dtype=np.float64)
        end = np.asarray(source_positions[joint.name], dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,) or not (
            np.all(np.isfinite(start)) and np.all(np.isfinite(end))
        ):
            raise SkeletonError(f"Invalid source coordinates for {joint.name!r}.")
        vector = end - start
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= 1e-8:
            raise SkeletonError(f"Cannot infer direction for zero-length source bone {joint.name!r}.")
        length = target_bone_lengths.get(joint.name)
        if length is None or not np.isfinite(length) or length <= 0:
            raise SkeletonError(f"Positive target length required for {joint.name!r}.")
        target_start = output[joint.parent]
        output[joint.name] = target_start + (vector / magnitude) * float(length)

    return output
