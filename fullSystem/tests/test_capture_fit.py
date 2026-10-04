"""A screen capture reaches the pipeline at the calibrated frame size."""
import numpy as np
import pytest

from capture_fit import fit_frame, parse_size


def test_parse_size_reads_width_by_height():
    assert parse_size("3840x2160") == (3840, 2160)


@pytest.mark.parametrize("bad", ["3840", "3840x", "x2160", "0x2160", "3840x-1", "big"])
def test_parse_size_rejects_what_is_not_a_size(bad):
    with pytest.raises(ValueError):
        parse_size(bad)


def test_a_half_size_capture_is_brought_up_to_the_calibration_size():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[500:540, 900:960] = 255                      # an object at (930, 520)
    out = fit_frame(frame, (3840, 2160))
    assert out.shape == (2160, 3840, 3)
    ys, xs = np.nonzero(out[:, :, 0] > 127)
    assert xs.mean() == pytest.approx(2 * 930, abs=2)   # same place, 4K pixels
    assert ys.mean() == pytest.approx(2 * 520, abs=2)


def test_no_target_size_leaves_the_frame_alone():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert fit_frame(frame, None) is frame


def test_a_frame_already_at_the_target_size_is_not_resampled():
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
    assert fit_frame(frame, (3840, 2160)) is frame
