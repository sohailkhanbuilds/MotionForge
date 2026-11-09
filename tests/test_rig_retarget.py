import numpy as np

from motionforge.rig import RigProfile
from motionforge.rig_retarget import retarget_to_profile, sequence_to_joint_maps
from motionforge.skeleton import JOINTS


def make_frame():
    # A valid, non-degenerate canonical humanoid pose.
    points = {
        "Hips": (0, 0, 0), "Spine": (0, 0.5, 0), "Neck": (0, 0.8, 0),
        "Head": (0, 1.0, 0),
        "LeftShoulder": (-0.2, 0.5, 0), "LeftElbow": (-0.4, 0.3, 0),
        "LeftWrist": (-0.6, 0.1, 0),
        "RightShoulder": (0.2, 0.5, 0), "RightElbow": (0.4, 0.3, 0),
        "RightWrist": (0.6, 0.1, 0),
        "LeftUpLeg": (-0.15, -0.4, 0), "LeftLeg": (-0.15, -0.8, 0),
        "LeftFoot": (-0.15, -0.9, 0.1),
        "RightUpLeg": (0.15, -0.4, 0), "RightLeg": (0.15, -0.8, 0),
        "RightFoot": (0.15, -0.9, 0.1),
    }
    return {name: np.array(value, dtype=float) for name, value in points.items()}


def test_saved_rig_profile_drives_sequence_retargeting():
    source = make_frame()
    # A second frame translated in space.
    second = {name: point + np.array([0.1, 0.0, 0.0]) for name, point in source.items()}
    lengths = {
        joint.name: float(np.linalg.norm(source[joint.name] - source[joint.parent]))
        for joint in JOINTS if joint.parent is not None
    }
    profile = RigProfile("Target", lengths)
    result = retarget_to_profile([source, second], profile)
    assert result.shape == (2, len(JOINTS), 3)
    assert np.all(np.isfinite(result))


def test_sequence_to_joint_maps_round_trip():
    values = np.stack([[make_frame()[joint.name] for joint in JOINTS]])
    maps = sequence_to_joint_maps(values)
    assert len(maps) == 1
    for index, joint in enumerate(JOINTS):
        np.testing.assert_allclose(maps[0][joint.name], values[0, index])
