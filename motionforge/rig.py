"""Portable target-rig profiles for MotionForge retargeting.

Profiles store measured bone lengths in metres. They do not contain mesh,
skin weights, bind-pose rotations or a production humanoid rig.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .skeleton import JOINTS, joint_positions

SCHEMA_VERSION = 1
BONE_NAMES = tuple(joint.name for joint in JOINTS if joint.parent is not None)


class RigProfileError(ValueError):
    """Raised when a rig profile is incomplete or invalid."""


@dataclass(frozen=True)
class RigProfile:
    name: str
    bone_lengths: Mapping[str, float]
    units: str = "metres"
    skeleton_id: str = "motionforge.humanoid.v1"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise RigProfileError("Rig profile name cannot be empty.")
        if self.units != "metres":
            raise RigProfileError("Rig profile lengths must be stored in metres.")
        if self.skeleton_id != "motionforge.humanoid.v1":
            raise RigProfileError(f"Unsupported skeleton_id: {self.skeleton_id}")
        keys = set(self.bone_lengths)
        expected = set(BONE_NAMES)
        if keys != expected:
            missing, extra = sorted(expected - keys), sorted(keys - expected)
            raise RigProfileError(f"Bone lengths mismatch; missing={missing}, extra={extra}.")
        cleaned = {}
        for name in BONE_NAMES:
            value = float(self.bone_lengths[name])
            if not np.isfinite(value) or value <= 1e-5:
                raise RigProfileError(f"Bone length for {name} must be finite and positive.")
            cleaned[name] = value
        object.__setattr__(self, "bone_lengths", cleaned)

    @classmethod
    def from_pose(cls, name: str, frame: np.ndarray) -> "RigProfile":
        """Estimate canonical bone lengths from one confident pose frame."""
        positions = joint_positions(frame)
        lengths = {}
        for joint in JOINTS:
            if joint.parent is None:
                continue
            start, end = positions[joint.parent], positions[joint.name]
            if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
                raise RigProfileError(f"Cannot measure {joint.name}: source landmarks are unreliable.")
            length = float(np.linalg.norm(end - start))
            if length <= 1e-5:
                raise RigProfileError(f"Cannot measure degenerate bone: {joint.name}.")
            lengths[joint.name] = length
        return cls(name=name, bone_lengths=lengths)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "skeleton_id": self.skeleton_id,
            "units": self.units,
            "bone_lengths": dict(self.bone_lengths),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RigProfile":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RigProfileError(f"Unable to read rig profile: {exc}") from exc
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise RigProfileError(f"Unsupported rig profile schema: {payload.get('schema_version')!r}")
        try:
            return cls(
                name=payload["name"],
                bone_lengths=payload["bone_lengths"],
                units=payload["units"],
                skeleton_id=payload["skeleton_id"],
            )
        except (KeyError, TypeError) as exc:
            raise RigProfileError(f"Malformed rig profile: {exc}") from exc
