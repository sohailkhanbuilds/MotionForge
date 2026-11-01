import numpy as np
from motionforge.bvh import export_bvh


def test_bvh_export_writes_hierarchy_and_motion(tmp_path):
    frames = np.zeros((3, 33, 6), dtype=np.float32)
    frames[:, :, 3] = 1.0
    # Give the main landmarks a non-degenerate rest pose.
    points = {
        0: (0.0, 0.7, 0.0), 11: (-0.2, 0.5, 0.0), 12: (0.2, 0.5, 0.0),
        13: (-0.4, 0.3, 0.0), 14: (0.4, 0.3, 0.0),
        15: (-0.6, 0.1, 0.0), 16: (0.6, 0.1, 0.0),
        23: (-0.15, 0.0, 0.0), 24: (0.15, 0.0, 0.0),
        25: (-0.15, -0.4, 0.0), 26: (0.15, -0.4, 0.0),
        27: (-0.15, -0.8, 0.0), 28: (0.15, -0.8, 0.0),
        31: (-0.15, -0.9, 0.1), 32: (0.15, -0.9, 0.1),
    }
    for idx, xyz in points.items():
        frames[:, idx, :3] = xyz
    frames[1:, :, 4] = 0.01
    output = tmp_path / "test.bvh"
    export_bvh(frames, 30.0, output)
    text = output.read_text(encoding="utf-8")
    assert "HIERARCHY" in text
    assert "ROOT Hips" in text
    assert "JOINT LeftWrist" in text
    assert "Frames: 3" in text
    motion = text.split("MOTION", 1)[1].strip().splitlines()
    assert len(motion) == 5  # frame count, frame time, then three frames
    assert len(motion[2].split()) == 54  # 6 root channels + 16 joints * 3
