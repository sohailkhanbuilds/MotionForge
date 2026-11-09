import numpy as np
import pytest

from motionforge.canonical_bvh import canonical_to_landmarks, export_canonical_bvh
from motionforge.skeleton import JOINTS


def canonical_frames(count=3):
    values = np.zeros((count, len(JOINTS), 3), dtype=float)
    positions = {
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
    for index, joint in enumerate(JOINTS):
        values[:, index, :] = positions[joint.name]
    values[1:, :, 0] += 0.02
    return values


def test_canonical_adapter_maps_supported_joints():
    canonical = canonical_frames(2)
    landmarks = canonical_to_landmarks(canonical)
    assert landmarks.shape == (2, 33, 6)
    np.testing.assert_allclose(landmarks[:, 15, :3], canonical[:, 6, :])
    assert np.all(landmarks[:, 15, 3] == 1.0)
    assert np.all(landmarks[:, 1, 3] == 0.0)


def test_canonical_bvh_export_creates_motion_file(tmp_path):
    output = tmp_path / "retargeted.bvh"
    export_canonical_bvh(canonical_frames(), 30, output)
    data = output.read_text(encoding="utf-8")
    assert "ROOT Hips" in data
    assert "Frames: 3" in data
    rows = data.split("MOTION", 1)[1].strip().splitlines()[2:]
    assert len(rows) == 3
    assert all(len(row.split()) == 51 for row in rows)


@pytest.mark.parametrize("bad", [np.zeros((2, 10, 3)), np.zeros((0, len(JOINTS), 3))])
def test_adapter_rejects_invalid_shapes_and_empty_sequences(bad):
    with pytest.raises(ValueError):
        canonical_to_landmarks(bad)
