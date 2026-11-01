"""Confidence-aware temporal smoothing for 3D and image-space landmarks."""
from __future__ import annotations

import numpy as np


def smooth_landmarks(frames: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth landmark coordinates with a visibility-weighted Gaussian window.

    Compared with a box moving average, Gaussian weights reduce jitter without
    flattening short movements as aggressively. Low-confidence samples have
    less influence; visibility values themselves are never modified.
    """
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[2] < 4:
        raise ValueError("frames must have shape (frames, landmarks, >=4)")
    if window <= 1 or len(frames) < 3:
        return frames.copy()

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    sigma = max(1.0, window / 3.0)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    out = frames.copy()
    coordinate_channels = [0, 1, 2] + ([4, 5] if frames.shape[2] >= 6 else [])
    ids = np.arange(len(frames))
    for landmark in range(frames.shape[1]):
        visibility = np.clip(frames[:, landmark, 3].astype(np.float64), 0.0, 1.0)
        valid = visibility > 0.05
        if not np.any(valid):
            continue

        # Interpolate missing samples first, then use visibility as a weight so
        # uncertain detections do not pull the filtered trajectory off course.
        for axis in coordinate_channels:
            values = frames[:, landmark, axis].astype(np.float64)
            filled = values.copy()
            filled[~valid] = np.interp(ids[~valid], ids[valid], values[valid])
            padded_values = np.pad(filled, (radius, radius), mode="edge")
            padded_weights = np.pad(visibility, (radius, radius), mode="edge")
            numerator = np.convolve(padded_values * padded_weights, kernel, mode="valid")
            denominator = np.convolve(padded_weights, kernel, mode="valid")
            filtered = np.divide(
                numerator, denominator,
                out=filled.copy(), where=denominator > 1e-8,
            )
            out[:, landmark, axis] = filtered.astype(out.dtype, copy=False)

    return out
