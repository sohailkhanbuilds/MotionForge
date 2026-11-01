"""Video pose extraction using MediaPipe's 3D world landmarks."""
from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

# Stable, human-readable subset of MediaPipe Pose landmark indices.
LANDMARKS = {
    "hips": (23, 24),
    "spine": (11, 12),
    "neck": (11, 12),
    "head": (0,),
    "left_shoulder": (11,), "left_elbow": (13,), "left_wrist": (15,),
    "right_shoulder": (12,), "right_elbow": (14,), "right_wrist": (16,),
    "left_hip": (23,), "left_knee": (25,), "left_ankle": (27,),
    "left_foot": (31,),
    "right_hip": (24,), "right_knee": (26,), "right_ankle": (28,),
    "right_foot": (32,),
}


def extract_video(video_path: str | Path, model_complexity: int = 0):
    """Return (timestamps, landmark_frames, fps, frame_size).

    Each landmark frame is a (33, 4) float array: world x/y/z in metres-ish
    and visibility. MediaPipe world coordinates are hip-centred estimates.
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
        static_image_mode=False,
        model_complexity=int(model_complexity),
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_index = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            arr = np.zeros((33, 4), dtype=np.float32)
            if result.pose_world_landmarks:
                for i, lm in enumerate(result.pose_world_landmarks.landmark):
                    arr[i] = (lm.x, lm.y, lm.z, lm.visibility)
            else:
                arr[:, 3] = 0.0
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
