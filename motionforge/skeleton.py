"""Canonical humanoid skeleton contract shared by motion and rigging modules.

Landmark indices follow MediaPipe Pose's 33-point topology. Coordinates are
passed through in MediaPipe world space (metres, hip-centred); this module does
not claim calibrated camera-space reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str | None
    landmark_indices: tuple[int, ...]
    role: str


# Stable names are intended to become the interchange contract for retargeting.
JOINTS: tuple[Joint, ...] = (
    Joint("Hips", None, (23, 24), "root"),
    Joint("Spine", "Hips", (11, 12), "torso"),
    Joint("Neck", "Spine", (0, 11, 12), "torso"),
    Joint("Head", "Neck", (0,), "head"),
    Joint("LeftShoulder", "Spine", (11,), "arm"),
    Joint("LeftElbow", "LeftShoulder", (13,), "arm"),
    Joint("LeftWrist", "LeftElbow", (15,), "hand"),
    Joint("RightShoulder", "Spine", (12,), "arm"),
    Joint("RightElbow", "RightShoulder", (14,), "arm"),
    Joint("RightWrist", "RightElbow", (16,), "hand"),
    Joint("LeftUpLeg", "Hips", (25,), "leg"),
    Joint("LeftLeg", "LeftUpLeg", (27,), "leg"),
    Joint("LeftFoot", "LeftLeg", (31,), "foot"),
    Joint("RightUpLeg", "Hips", (26,), "leg"),
    Joint("RightLeg", "RightUpLeg", (28,), "leg"),
    Joint("RightFoot", "RightLeg", (32,), "foot"),
)

JOINT_BY_NAME = {joint.name: joint for joint in JOINTS}
CHILDREN: dict[str, tuple[str, ...]] = {
    joint.name: tuple(child.name for child in JOINTS if child.parent == joint.name)
    for joint in JOINTS
}
COORDINATE_CONVENTION = "mediapipe_world_metres_hip_centred"


class SkeletonError(ValueError):
    """Raised when skeleton data violates the canonical contract."""


def validate_skeleton(joints: Iterable[Joint] = JOINTS) -> None:
    """Validate unique names, valid parents, landmark indices and acyclic graph."""
    items = tuple(joints)
    names = [joint.name for joint in items]
    if len(names) != len(set(names)):
        raise SkeletonError("Joint names must be unique.")
    known = set(names)
    roots = [joint for joint in items if joint.parent is None]
    if len(roots) != 1:
        raise SkeletonError("A humanoid skeleton must have exactly one root.")
    for joint in items:
        if joint.parent is not None and joint.parent not in known:
            raise SkeletonError(f"Unknown parent {joint.parent!r} for {joint.name!r}.")
        if not joint.landmark_indices or any(
            not isinstance(index, int) or index < 0 or index > 32
            for index in joint.landmark_indices
        ):
            raise SkeletonError(f"Invalid landmark mapping for {joint.name!r}.")

    parents = {joint.name: joint.parent for joint in items}
    for name in names:
        visited: set[str] = set()
        cursor: str | None = name
        while cursor is not None:
            if cursor in visited:
                raise SkeletonError(f"Cycle detected at joint {cursor!r}.")
            visited.add(cursor)
            cursor = parents[cursor]


def joint_positions(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Map one (33, >=4) MediaPipe frame to canonical joint positions.

    Multi-landmark joints use the arithmetic mean of their source landmarks.
    Invalid or low-confidence values are returned as NaN; callers decide how
    to interpolate or gate them.
    """
    values = np.asarray(frame, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 33 or values.shape[1] < 4:
        raise SkeletonError("frame must have shape (33, >=4).")
    result: dict[str, np.ndarray] = {}
    for joint in JOINTS:
        points = values[list(joint.landmark_indices), :3]
        confidence = values[list(joint.landmark_indices), 3]
        if (
            not np.all(np.isfinite(points))
            or not np.all(np.isfinite(confidence))
            or float(np.min(confidence)) < 0.2
        ):
            result[joint.name] = np.full(3, np.nan, dtype=np.float64)
        else:
            result[joint.name] = np.mean(points, axis=0)
    return result


validate_skeleton()
