import numpy as np
import pytest

from motionforge.motion_data import MotionDataError, MotionSequence


def sample():
    return MotionSequence(
        timestamps=np.array([0.0, 0.04, 0.08]),
        frames=np.zeros((3, 33, 6), dtype=np.float32),
        fps=25.0,
        frame_size=(1280, 720),
        source="walk.mp4",
    )


def test_motion_sequence_normalizes_and_reports_metadata():
    sequence = sample()
    assert sequence.frame_count == 3
    assert sequence.duration == pytest.approx(0.08)
    metadata = sequence.to_metadata()
    assert metadata["landmark_count"] == 33
    assert "not calibrated" in metadata["depth_note"]


@pytest.mark.parametrize("timestamps", [
    np.array([]), np.array([0.0, 0.0]), np.array([0.0, np.nan]),
])
def test_rejects_invalid_timestamps(timestamps):
    frames = np.zeros((len(timestamps), 33, 6), dtype=np.float32)
    with pytest.raises(MotionDataError):
        MotionSequence(timestamps, frames, 30.0, (640, 480))


def test_rejects_invalid_frame_shape_and_fps():
    with pytest.raises(MotionDataError):
        MotionSequence(np.array([0.0]), np.zeros((1, 32, 6)), 30.0, (640, 480))
    with pytest.raises(MotionDataError):
        MotionSequence(np.array([0.0]), np.zeros((1, 33, 6)), 0.0, (640, 480))
