"""Render a lightweight 2D pose overlay video for motion-capture QA."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    (0, 11), (0, 12),
)

def render_pose_preview(video_path: str | Path, frames: np.ndarray, fps: float, output_path: str | Path):
    """Write source video with detected pose landmarks overlaid."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not reopen input video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(float(fps), 1.0), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create pose preview MP4 (mp4v codec unavailable).")
    index = 0
    try:
        while index < len(frames):
            ok, image = cap.read()
            if not ok:
                break
            landmarks = frames[index]
            points = {}
            for i in range(min(33, len(landmarks))):
                x, y = landmarks[i, 4:6]
                if float(landmarks[i, 3]) >= 0.25 and np.isfinite(x) and np.isfinite(y):
                    points[i] = (int(np.clip(x, 0, 1) * (width - 1)),
                                 int(np.clip(y, 0, 1) * (height - 1)))
            for a, b in CONNECTIONS:
                if a in points and b in points:
                    cv2.line(image, points[a], points[b], (40, 220, 80), 2, cv2.LINE_AA)
            for point in points.values():
                cv2.circle(image, point, 4, (40, 80, 255), -1, cv2.LINE_AA)
            cv2.putText(image, f"MotionForge | frame {index + 1}/{len(frames)}",
                        (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(image)
            index += 1
    finally:
        cap.release()
        writer.release()
    if index == 0:
        raise ValueError("No frames were written to the pose preview.")
    return Path(output_path)
