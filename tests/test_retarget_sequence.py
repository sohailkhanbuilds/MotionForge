import numpy as np
import pytest

from motionforge.retarget_sequence import (
    measure_bone_lengths, measure_temporal_smoothness, retarget_sequence,
)
from motionforge.skeleton import JOINTS, SkeletonError


def source_frame(offset=0.0):
    return {
        joint.name: np.array([float(i) + offset, float(i % 3), 0.0])
        for i, joint in enumerate(JOINTS)
    }


def target_lengths():
    return {joint.name: 0.75 for joint in JOINTS if joint.parent is not None}


def test_sequence_retargeting_preserves_root_trajectory_and_bone_lengths():
    frames = [source_frame(0.0), source_frame(1.0), source_frame(2.0)]
    roots = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    result = retarget_sequence(frames, target_lengths(), roots)
    assert result.shape == (3, len(JOINTS), 3)
    np.testing.assert_allclose(result[:, 0, :], roots)
    report = measure_bone_lengths(result)
    assert all(item["mean"] == pytest.approx(0.75) for item in report.values())
    assert all(item["std"] == pytest.approx(0.0) for item in report.values())


def test_retarget_sequence_rejects_bad_root_array():
    with pytest.raises(SkeletonError, match="root_positions"):
        retarget_sequence([source_frame()], target_lengths(), np.zeros((2, 3)))


def test_temporal_smoothness_reports_zero_for_static_pose():
    frame = np.stack([source_frame()[joint.name] for joint in JOINTS])
    metrics = measure_temporal_smoothness(np.stack([frame, frame, frame]), 30.0)
    assert metrics["mean_joint_speed"] == 0.0
    assert metrics["mean_joint_acceleration"] == 0.0


def test_quality_metrics_reject_invalid_arrays():
    with pytest.raises(SkeletonError):
        measure_bone_lengths(np.zeros((0, len(JOINTS), 3)))
    with pytest.raises(SkeletonError):
        measure_temporal_smoothness(np.zeros((1, len(JOINTS), 3)), 30.0)
