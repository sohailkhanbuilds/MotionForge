"""Small dependency-light temporal smoothing helpers."""
import numpy as np


def smooth_landmarks(frames: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing over time; preserves visibility channel."""
    if window <= 1 or len(frames) < 3:
        return frames.copy()
    if window % 2 == 0:
        window += 1
    out = frames.copy()
    radius = window // 2
    for landmark in range(frames.shape[1]):
        for axis in range(3):
            values = frames[:, landmark, axis]
            # Do not let undetected zero frames drag valid coordinates to origin.
            valid = frames[:, landmark, 3] > 0.2
            if not np.any(valid):
                continue
            filled = values.copy()
            ids = np.arange(len(values))
            filled[~valid] = np.interp(ids[~valid], ids[valid], values[valid])
            padded = np.pad(filled, (radius, radius), mode="edge")
            kernel = np.ones(window, dtype=np.float32) / window
            out[:, landmark, axis] = np.convolve(padded, kernel, mode="valid")
    return out
