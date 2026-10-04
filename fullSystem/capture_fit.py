"""Bring a screen capture to the frame size the scene was calibrated at."""
import cv2


def parse_size(text: str) -> tuple[int, int]:
    """'3840x2160' -> (3840, 2160)."""
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"expected WIDTHxHEIGHT, got {text!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"size must be positive, got {text!r}")
    return w, h


def fit_frame(frame, size):
    """Resize a frame to size (width, height); None is returned as is."""
    if size is None or (frame.shape[1], frame.shape[0]) == tuple(size):
        return frame
    return cv2.resize(frame, tuple(size), interpolation=cv2.INTER_LINEAR)
