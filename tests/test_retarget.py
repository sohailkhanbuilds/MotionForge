import numpy as np
import pytest

from motionforge.retarget import retarget_frame
from motionforge.skeleton import JOINTS, SkeletonError


def make_pose():
    positions = {}
    for index, joint in enumerate(JOINTS):
        positions[joint.name] = np.array([float(index), 0.0, 0.0])
    return positions


def lengths(value=2.0):
    return {joint.name: value for joint in JOINTS if joint.parent is not None}


def test_retargeter_fits_target_bone_lengths():
    source = make_pose()
    target = retarget_frame(source, lengths(), root_position=np.array([10.0, 1.0, 0.0]))
    np.testing.assert_allclose(target["Hips"], [10, 1, 0])
    for joint in JOINTS:
        if joint.parent is not None:
            distance = np.linalg.norm(target[joint.name] - target[joint.parent])
            assert distance == pytest.approx(2.0)


def test_retargeter_rejects_missing_lengths_and_degenerate_bones():
    source = make_pose()
    missing = lengths()
    missing.pop("LeftElbow")
    with pytest.raises(SkeletonError, match="target length"):
        retarget_frame(source, missing)
    source["LeftElbow"] = source["LeftShoulder"].copy()
    with pytest.raises(SkeletonError, match="zero-length"):
        retarget_frame(source, lengths())
