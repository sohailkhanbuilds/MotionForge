"""Video pose extraction using MediaPipe world and image landmarks."""
from __future__ import annotations
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np


def extract_video(video_path: str | Path, model_complexity: int = 0):
    """Return timestamps, frames, fps, frame_size.

    Each frame has shape (33, 6): world x/y/z, visibility, normalized image x/y.
    World coordinates are hip-centred; image coordinates provide approximate
    screen-plane root motion. Neither is calibrated absolute world tracking.
    """
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.1 or fps > 240:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    timestamps, frames = [], []
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=int(model_complexity),
        smooth_landmarks=True, enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    frame_index = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            result = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            arr = np.zeros((33, 6), dtype=np.float32)
            if result.pose_world_landmarks:
                for i, lm in enumerate(result.pose_world_landmarks.landmark):
                    arr[i, :4] = (lm.x, lm.y, lm.z, lm.visibility)
            if result.pose_landmarks:
                for i, lm in enumerate(result.pose_landmarks.landmark):
                    arr[i, 4:6] = (lm.x, lm.y)
            frames.append(arr)
            timestamps.append(frame_index / fps)
            frame_index += 1
    finally:
        pose.close()
        cap.release()
    if not frames:
        raise ValueError("The video contains no readable frames.")
    if not any(float(f[:, 3].max()) > 0.1 for f in frames):
        raise ValueError("No person was detected. Try a brighter video with the full body visible.")
    return np.asarray(timestamps), np.stack(frames), fps, (width, height)
