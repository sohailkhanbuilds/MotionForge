"""High-level video capture API returning a validated MotionSequence."""
from __future__ import annotations

from pathlib import Path

from motionforge.motion_data import MotionSequence
from motionforge.pose import extract_video


def capture_video(
    video_path: str | Path,
    model_complexity: int = 0,
) -> MotionSequence:
    """Extract MediaPipe landmarks and package them in MotionForge's data contract."""
    timestamps, frames, fps, frame_size = extract_video(video_path, model_complexity)
    return MotionSequence(
        timestamps=timestamps,
        frames=frames,
        fps=fps,
        frame_size=frame_size,
        source=Path(video_path).name,
    )
