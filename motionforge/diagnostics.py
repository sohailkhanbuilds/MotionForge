"""Tracking quality metrics for MotionForge output."""
from __future__ import annotations
import numpy as np

def tracking_quality(frames: np.ndarray) -> dict:
    """Summarize pose visibility; this is not a measure of 3D accuracy."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[1] < 33 or frames.shape[2] < 4:
        raise ValueError("frames must have shape (frames, 33, >=4)")
    visibility = np.clip(np.nan_to_num(frames[:, :, 3], nan=0.0), 0.0, 1.0)
    frame_scores = visibility.mean(axis=1)
    detected = frame_scores >= 0.35
    return {
        "mean_visibility": round(float(visibility.mean()), 4),
        "detected_frame_percent": round(float(detected.mean() * 100.0), 2),
        "low_confidence_frame_percent": round(float((~detected).mean() * 100.0), 2),
        "landmark_visibility_percent": {
            "head": round(float(visibility[:, [0, 7, 8]].mean() * 100.0), 2),
            "arms": round(float(visibility[:, 11:17].mean() * 100.0), 2),
            "legs": round(float(visibility[:, 23:33].mean() * 100.0), 2),
        },
        "warning": "Visibility is detector confidence, not measured motion accuracy.",
    }
