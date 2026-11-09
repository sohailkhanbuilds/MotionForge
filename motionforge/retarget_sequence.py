"""Multi-frame retargeting and lightweight motion quality diagnostics."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from motionforge.retarget import retarget_frame
from motionforge.skeleton import JOINTS, SkeletonError


def retarget_sequence(
    source_frames: Sequence[Mapping[str, np.ndarray]],
    target_bone_lengths: Mapping[str, float],
    root_positions: np.ndarray | None = None,
) -> np.ndarray:
    """Retarget a sequence to fixed target bone lengths.

    Returns an array shaped (frame_count, joint_count, 3), ordered exactly as
    motionforge.skeleton.JOINTS. Root positions may be supplied per frame.
    """
    if not source_frames:
        raise SkeletonError("At least one source frame is required.")
    roots = None if root_positions is None else np.asarray(root_positions, dtype=np.float64)
    if roots is not None and roots.shape != (len(source_frames), 3):
        raise SkeletonError("root_positions must have shape (frame_count, 3).")
    output = []
    for index, frame in enumerate(source_frames):
        root = None if roots is None else roots[index]
        fitted = retarget_frame(frame, target_bone_lengths, root)
        output.append(np.stack([fitted[joint.name] for joint in JOINTS]))
    return np.stack(output)


def measure_bone_lengths(
    frames: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Report mean, standard deviation and max deviation for each bone length."""
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(JOINTS), 3):
        raise SkeletonError(
            f"frames must have shape (N, {len(JOINTS)}, 3) in canonical joint order."
        )
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise SkeletonError("frames must be non-empty and contain only finite coordinates.")
    index = {joint.name: i for i, joint in enumerate(JOINTS)}
    report: dict[str, dict[str, float]] = {}
    for joint in JOINTS:
        if joint.parent is None:
            continue
        lengths = np.linalg.norm(
            values[:, index[joint.name], :] - values[:, index[joint.parent], :],
            axis=1,
        )
        mean = float(np.mean(lengths))
        report[joint.name] = {
            "mean": mean,
            "std": float(np.std(lengths)),
            "max_deviation": float(np.max(np.abs(lengths - mean))),
        }
    return report


def measure_temporal_smoothness(frames: np.ndarray, fps: float) -> dict[str, float]:
    """Measure average joint speed and acceleration; these are diagnostics, not scores."""
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(JOINTS), 3):
        raise SkeletonError(
            f"frames must have shape (N, {len(JOINTS)}, 3) in canonical joint order."
        )
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise SkeletonError("At least two finite frames are required.")
    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        raise SkeletonError("fps must be in the range (0, 240].")
    velocity = np.diff(values, axis=0) * float(fps)
    speed = np.linalg.norm(velocity, axis=2)
    result = {"mean_joint_speed": float(np.mean(speed)), "max_joint_speed": float(np.max(speed))}
    if len(values) >= 3:
        acceleration = np.diff(velocity, axis=0) * float(fps)
        magnitude = np.linalg.norm(acceleration, axis=2)
        result["mean_joint_acceleration"] = float(np.mean(magnitude))
        result["max_joint_acceleration"] = float(np.max(magnitude))
    else:
        result["mean_joint_acceleration"] = 0.0
        result["max_joint_acceleration"] = 0.0
    return result
