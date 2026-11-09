"""Validated in-memory motion sequence contract.

This is the boundary between capture providers, cleanup, retargeting and export.
It deliberately stores landmark observations rather than pretending they are a
fully reconstructed 3D skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from motionforge.skeleton import COORDINATE_CONVENTION


class MotionDataError(ValueError):
    """Raised when a motion sequence is malformed."""


@dataclass(frozen=True)
class MotionSequence:
    timestamps: np.ndarray
    frames: np.ndarray
    fps: float
    frame_size: tuple[int, int]
    coordinate_convention: str = COORDINATE_CONVENTION
    source: str | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        frames = np.asarray(self.frames, dtype=np.float32)
        if timestamps.ndim != 1 or len(timestamps) == 0:
            raise MotionDataError("timestamps must be a non-empty one-dimensional array.")
        if frames.ndim != 3 or frames.shape[0] != len(timestamps):
            raise MotionDataError("frames must have shape (N, 33, 6) matching timestamps.")
        if frames.shape[1:] != (33, 6):
            raise MotionDataError("each frame must have shape (33, 6).")
        if not np.all(np.isfinite(timestamps)):
            raise MotionDataError("timestamps must contain only finite values.")
        if np.any(np.diff(timestamps) <= 0):
            raise MotionDataError("timestamps must be strictly increasing.")
        if not np.isfinite(self.fps) or self.fps <= 0 or self.fps > 240:
            raise MotionDataError("fps must be finite and in the range (0, 240].")
        if (
            len(self.frame_size) != 2
            or any(not isinstance(value, int) or value <= 0 for value in self.frame_size)
        ):
            raise MotionDataError("frame_size must contain positive integer width and height.")
        if self.coordinate_convention != COORDINATE_CONVENTION:
            raise MotionDataError(
                f"Unsupported coordinate convention: {self.coordinate_convention!r}"
            )
        # Freeze normalized arrays on the instance so downstream stages get
        # consistent dtypes regardless of the capture backend.
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "frame_size", tuple(self.frame_size))

    @property
    def duration(self) -> float:
        """Elapsed time from first to last sample (not container duration)."""
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def frame_count(self) -> int:
        return int(self.frames.shape[0])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "fps": float(self.fps),
            "duration_seconds": self.duration,
            "frame_size": list(self.frame_size),
            "coordinate_convention": self.coordinate_convention,
            "source": self.source,
            "landmark_count": 33,
            "landmark_fields": ["world_x", "world_y", "world_z", "visibility", "image_x", "image_y"],
            "depth_note": "Single-camera landmark depth is not calibrated absolute scene depth.",
        }
