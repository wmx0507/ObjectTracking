"""
Video I/O, visualization, and trajectory drawing utilities.
Uses OpenCV only for reading/writing video frames (not for algorithm logic).
"""

import cv2
import numpy as np
import os


def imread_unicode(path: str) -> np.ndarray:
    """
    Read an image from a path that may contain Unicode (e.g. Chinese) characters.
    OpenCV's cv2.imread fails on such paths on Windows.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return img


def read_video(path: str):
    """Generator that yields frames from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info = {'fps': fps, 'total_frames': total_frames, 'width': width, 'height': height}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def create_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a video writer for output."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def draw_box(frame: np.ndarray, x: int, y: int, w: int, h: int,
             color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw bounding box."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    return frame


def draw_trajectory(frame: np.ndarray, positions: list[tuple[int, int]],
                    color: tuple = (0, 255, 255), thickness: int = 2) -> np.ndarray:
    """Draw trajectory line connecting all recorded positions."""
    if len(positions) < 2:
        return frame
    for i in range(1, len(positions)):
        cv2.line(frame, positions[i - 1], positions[i], color, thickness)
    return frame


def draw_circle(frame: np.ndarray, x: int, y: int,
                radius: int = 5, color: tuple = (0, 0, 255), thickness: int = -1) -> np.ndarray:
    """Draw center point marker."""
    cv2.circle(frame, (x, y), radius, color, thickness)
    return frame


def put_text(frame: np.ndarray, text: str, x: int = 10, y: int = 30,
             color: tuple = (255, 255, 255), size: float = 0.8) -> np.ndarray:
    """Overlay status text on the frame."""
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)
    return frame


def crop_template_from_target(image_path: str, center: tuple[int, int],
                               half_size: int = 40) -> np.ndarray:
    """
    Crop a template from a target image around a given center point.
    Used to extract the vehicle/cyclist template from the target images.
    """
    img = imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    cx, cy = center
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(img.shape[1], cx + half_size)
    y2 = min(img.shape[0], cy + half_size)
    return img[y1:y2, x1:x2]
