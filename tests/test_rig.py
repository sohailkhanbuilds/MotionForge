import json

import numpy as np
import pytest

from motionforge.rig import BONE_NAMES, RigProfile, RigProfileError


def pose_frame():
    frame = np.zeros((33, 6), dtype=np.float32)
    frame[:, 3] = 1.0
    points = {
        0: (0.0, 0.7, 0.0), 11: (-0.2, 0.5, 0.0), 12: (0.2, 0.5, 0.0),
        13: (-0.4, 0.3, 0.0), 14: (0.4, 0.3, 0.0),
        15: (-0.6, 0.1, 0.0), 16: (0.6, 0.1, 0.0),
        23: (-0.15, 0.0, 0.0), 24: (0.15, 0.0, 0.0),
        25: (-0.15, -0.4, 0.0), 26: (0.15, -0.4, 0.0),
        27: (-0.15, -0.8, 0.0), 28: (0.15, -0.8, 0.0),
        31: (-0.15, -0.9, 0.1), 32: (0.15, -0.9, 0.1),
    }
    for index, point in points.items():
        frame[index, :3] = point
    return frame


def test_profile_can_be_measured_and_round_tripped(tmp_path):
    profile = RigProfile.from_pose("Hero", pose_frame())
    assert set(profile.bone_lengths) == set(BONE_NAMES)
    assert all(value > 0 for value in profile.bone_lengths.values())
    path = profile.save(tmp_path / "hero.rig.json")
    loaded = RigProfile.load(path)
    assert loaded.to_dict() == profile.to_dict()


def test_profile_rejects_missing_or_invalid_lengths():
    with pytest.raises(RigProfileError, match="mismatch"):
        RigProfile("Broken", {})
    values = {name: 0.2 for name in BONE_NAMES}
    values["LeftWrist"] = float("nan")
    with pytest.raises(RigProfileError, match="finite and positive"):
        RigProfile("Broken", values)


def test_profile_rejects_low_confidence_measurement():
    frame = pose_frame()
    frame[15, 3] = 0.0
    with pytest.raises(RigProfileError, match="unreliable"):
        RigProfile.from_pose("Hero", frame)


def test_load_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(RigProfileError, match="Unsupported"):
        RigProfile.load(path)
