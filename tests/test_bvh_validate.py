import numpy as np
import pytest

from motionforge.bvh import export_bvh
from motionforge.bvh_validate import BVHValidationError, validate_bvh


def make_frames():
    frames = np.zeros((2, 33, 6), dtype=float)
    frames[:, :, 3] = 1.0
    points = {
        0: (0, 0.7, 0), 11: (-0.2, 0.5, 0), 12: (0.2, 0.5, 0),
        13: (-0.4, 0.3, 0), 14: (0.4, 0.3, 0),
        15: (-0.6, 0.1, 0), 16: (0.6, 0.1, 0),
        23: (-0.15, 0, 0), 24: (0.15, 0, 0),
        25: (-0.15, -0.4, 0), 26: (0.15, -0.4, 0),
        27: (-0.15, -0.8, 0), 28: (0.15, -0.8, 0),
        31: (-0.15, -0.9, 0.1), 32: (0.15, -0.9, 0.1),
    }
    for index, point in points.items():
        frames[:, index, :3] = point
    return frames


def test_validate_generated_bvh(tmp_path):
    path = tmp_path / "valid.bvh"
    export_bvh(make_frames(), 24, path)
    report = validate_bvh(path)
    assert report.joint_count == 16
    assert report.channel_count == 51
    assert report.frame_count == 2
    assert report.frame_time == pytest.approx(1 / 24, abs=1e-7)
    assert report.motion_values == 102


def test_validator_rejects_wrong_motion_row_width(tmp_path):
    path = tmp_path / "broken.bvh"
    export_bvh(make_frames(), 24, path)
    text = path.read_text(encoding="utf-8")
    rows = text.splitlines()
    motion_row = next(i for i, line in enumerate(rows) if line.startswith("0.00000"))
    rows[motion_row] = "0 1 2"
    path.write_text("\n".join(rows), encoding="utf-8")
    with pytest.raises(BVHValidationError, match="expected 51"):
        validate_bvh(path)


def test_validator_rejects_non_finite_motion(tmp_path):
    path = tmp_path / "broken.bvh"
    export_bvh(make_frames(), 24, path)
    text = path.read_text(encoding="utf-8").replace("0.00000", "nan", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(BVHValidationError):
        validate_bvh(path)
