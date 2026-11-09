"""Structural and numeric validation for MotionForge BVH exports."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class BVHValidationError(ValueError):
    """Raised when a BVH file is malformed or contains invalid motion data."""


@dataclass(frozen=True)
class BVHReport:
    joint_count: int
    channel_count: int
    frame_count: int
    frame_time: float
    motion_values: int


def validate_bvh(path: str | Path) -> BVHReport:
    """Validate hierarchy declarations and every numeric motion row.

    This is a strict validator for MotionForge's current BVH output profile,
    not a general-purpose parser for every BVH dialect.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise BVHValidationError(f"Cannot read BVH file: {exc}") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    try:
        motion_index = lines.index("MOTION")
    except ValueError as exc:
        raise BVHValidationError("Missing MOTION section.") from exc
    hierarchy = lines[:motion_index]
    if not hierarchy or hierarchy[0] != "HIERARCHY":
        raise BVHValidationError("Missing HIERARCHY header.")
    joint_count = sum(
        line.startswith("ROOT ") or line.startswith("JOINT ")
        for line in hierarchy
    )
    if joint_count < 1 or not any(line.startswith("ROOT ") for line in hierarchy):
        raise BVHValidationError("Hierarchy must contain at least one ROOT.")
    channel_count = 0
    for line in hierarchy:
        if line.startswith("CHANNELS "):
            parts = line.split()
            if len(parts) < 3:
                raise BVHValidationError(f"Malformed channel declaration: {line}")
            try:
                declared = int(parts[1])
            except ValueError as exc:
                raise BVHValidationError(f"Invalid channel count: {line}") from exc
            if declared != len(parts) - 2:
                raise BVHValidationError(f"Channel count mismatch: {line}")
            channel_count += declared
    if channel_count < 1:
        raise BVHValidationError("No motion channels declared.")
    motion = lines[motion_index + 1:]
    if len(motion) < 2 or not motion[0].startswith("Frames:") or not motion[1].startswith("Frame Time:"):
        raise BVHValidationError("MOTION must declare Frames and Frame Time.")
    try:
        frame_count = int(motion[0].split(":", 1)[1].strip())
        frame_time = float(motion[1].split(":", 1)[1].strip())
    except (ValueError, IndexError) as exc:
        raise BVHValidationError("Invalid frame count or frame time.") from exc
    if frame_count < 1 or not np.isfinite(frame_time) or frame_time <= 0:
        raise BVHValidationError("Frame count and frame time must be positive.")
    rows = motion[2:]
    if len(rows) != frame_count:
        raise BVHValidationError(
            f"Declared {frame_count} frames but found {len(rows)} motion rows."
        )
    total_values = 0
    for number, row in enumerate(rows, start=1):
        try:
            values = np.asarray([float(value) for value in row.split()], dtype=float)
        except ValueError as exc:
            raise BVHValidationError(f"Non-numeric value in motion row {number}.") from exc
        if len(values) != channel_count:
            raise BVHValidationError(
                f"Motion row {number} has {len(values)} values; expected {channel_count}."
            )
        if not np.all(np.isfinite(values)):
            raise BVHValidationError(f"Non-finite value in motion row {number}.")
        total_values += len(values)
    return BVHReport(joint_count, channel_count, frame_count, frame_time, total_values)
