"""Temporal smoothing for 3D and normalized image-space landmarks."""
import numpy as np


def smooth_landmarks(frames: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smooth coordinate channels; keep visibility unchanged."""
    if window <= 1 or len(frames) < 3:
        return frames.copy()
    if window % 2 == 0:
        window += 1
    out = frames.copy()
    radius = window // 2
    coordinate_channels = [0, 1, 2] + ([4, 5] if frames.shape[2] >= 6 else [])
    for landmark in range(frames.shape[1]):
        valid = frames[:, landmark, 3] > 0.2
        if not np.any(valid):
            continue
        ids = np.arange(len(frames))
        for axis in coordinate_channels:
            values = frames[:, landmark, axis]
            filled = values.copy()
            filled[~valid] = np.interp(ids[~valid], ids[valid], values[valid])
            padded = np.pad(filled, (radius, radius), mode="edge")
            kernel = np.ones(window, dtype=np.float32) / window
            out[:, landmark, axis] = np.convolve(padded, kernel, mode="valid")
    return out
