import numpy as np
from motionforge.smoothing import smooth_landmarks


def test_smoothing_keeps_shape_and_visibility():
    frames = np.zeros((9, 33, 4), dtype=np.float32)
    frames[:, :, 0] = np.arange(9)[:, None]
    frames[:, :, 3] = 1.0
    result = smooth_landmarks(frames, 3)
    assert result.shape == frames.shape
    assert np.all(result[:, :, 3] == 1.0)
    assert result[4, 0, 0] < frames[4, 0, 0] + 1e-6


def test_window_one_is_noop():
    frames = np.random.default_rng(2).random((4, 33, 4)).astype(np.float32)
    np.testing.assert_array_equal(smooth_landmarks(frames, 1), frames)


def test_low_visibility_outlier_has_less_influence():
    frames = np.zeros((7, 33, 4), dtype=np.float32)
    frames[:, :, 3] = 1.0
    frames[3, 0, 0] = 100.0
    frames[3, 0, 3] = 0.1
    result = smooth_landmarks(frames, 5)
    assert result[3, 0, 0] < 20.0
