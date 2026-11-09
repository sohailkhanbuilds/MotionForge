import numpy as np
import pytest

from motionforge.skeleton import (
    CHILDREN, COORDINATE_CONVENTION, JOINTS, SkeletonError,
    joint_positions, validate_skeleton,
)


def test_canonical_skeleton_is_a_single_connected_tree():
    validate_skeleton()
    assert len(JOINTS) == 16
    assert sum(j.parent is None for j in JOINTS) == 1
    assert CHILDREN["Hips"] == ("Spine", "LeftUpLeg", "RightUpLeg")
    assert COORDINATE_CONVENTION == "mediapipe_world_metres_hip_centred"


def test_rejects_cycles_and_unknown_parents():
    from motionforge.skeleton import Joint

    with pytest.raises(SkeletonError, match="Cycle"):
        validate_skeleton((Joint("A", "B", (0,), "x"), Joint("B", "A", (1,), "x")))
    with pytest.raises(SkeletonError, match="Unknown parent"):
        validate_skeleton((Joint("A", "Missing", (0,), "x"),))


def test_joint_positions_average_synthesized_joints():
    frame = np.zeros((33, 6), dtype=np.float32)
    frame[:, 3] = 1.0
    frame[23, :3] = (1, 2, 3)
    frame[24, :3] = (3, 4, 5)
    frame[15, 3] = 0.0
    positions = joint_positions(frame)
    np.testing.assert_allclose(positions["Hips"], (2, 3, 4))
    assert np.isnan(positions["LeftWrist"]).all()


def test_low_confidence_joint_is_nan():
    frame = np.zeros((33, 6), dtype=np.float32)
    frame[:, 3] = 1.0
    frame[15, 3] = 0.1
    assert np.isnan(joint_positions(frame)["LeftWrist"]).all()
