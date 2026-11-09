import numpy as np
from motionforge.diagnostics import tracking_quality

def test_tracking_quality_reports_visibility_without_claiming_accuracy():
    frames = np.zeros((4, 33, 6), dtype=np.float32)
    frames[:2, :, 3] = 1.0
    result = tracking_quality(frames)
    assert result["detected_frame_percent"] == 50.0
    assert result["low_confidence_frame_percent"] == 50.0
    assert "not measured motion accuracy" in result["warning"]
