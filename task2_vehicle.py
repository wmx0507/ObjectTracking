"""
Task 2: Track the designated vehicle in 大疆无人机航拍视频.mp4.
Template extracted from 大疆无人机航拍视频目标.png (blue circle).
Must NOT misidentify other vehicles.
"""

import os
import cv2
import numpy as np
from tracker import ObjectTracker
from utils import read_video, create_writer, draw_box, draw_trajectory, draw_circle, put_text, imread_unicode

DATA_DIR = os.path.join(os.path.dirname(__file__), '图像处理作业1')
VIDEO_PATH = os.path.join(DATA_DIR, '大疆无人机航拍视频.mp4')
TARGET_IMAGE_PATH = os.path.join(DATA_DIR, '大疆无人机航拍视频目标.png')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output_task2.mp4')


def detect_blue_circle_center(image_path: str) -> tuple[int, int]:
    """Detect the center of the blue circle marker in the target image."""
    img = imread_unicode(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Blue color range in HSV
    lower_blue = np.array([100, 80, 80])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours, pick the largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No blue circle detected in target image")

    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print(f"Detected blue circle center: ({cx}, {cy})")
    return cx, cy


def extract_template_from_target(image_path: str, center: tuple[int, int],
                                  template_size: int = 50) -> np.ndarray:
    """Extract a square template around the given center."""
    img = imread_unicode(image_path)
    cx, cy = center
    half = template_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img.shape[1], cx + half)
    y2 = min(img.shape[0], cy + half)
    template = img[y1:y2, x1:x2]
    print(f"Extracted template: {template.shape}")
    return template


def main():
    # Detect circle and extract template
    cx_blue, cy_blue = detect_blue_circle_center(TARGET_IMAGE_PATH)
    template_size = 60
    template = extract_template_from_target(TARGET_IMAGE_PATH, (cx_blue, cy_blue), template_size)

    templates = [template]
    print(f"Template size: {template.shape}")

    # Initialize tracker with stricter threshold to avoid false matches
    tracker = ObjectTracker(
        templates=templates,
        corr_threshold=0.68,
        search_margin=120,
        template_update_rate=0.10,
        lost_max_frames=12,
    )

    # Video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")

    writer = create_writer(OUTPUT_PATH, fps, width, height)

    frame_idx = 0
    all_positions = []

    for frame in read_video(VIDEO_PATH):
        result = {'corr_value': 0.0, 'center': None, 'lost': False}

        if frame_idx == 0:
            best_pos, corr_val, _ = tracker.init_first_position(frame)
            result = {'corr_value': corr_val, 'center': tracker.current_pos, 'lost': False}
            print(f"Frame {frame_idx}: Initial position={best_pos}, corr={corr_val:.3f}")

            # Extract a fresh template from the video frame at the detected position.
            # This video-based template matches the video perfectly, giving high NCC scores.
            bx, by = best_pos
            bw, bh = tracker.template_w, tracker.template_h
            video_template = frame[by:by+bh, bx:bx+bw].astype(np.float64)
            tracker.templates.append(video_template)
            print(f"  Added video-extracted template: {video_template.shape}")
        else:
            result = tracker.track_frame(frame)

        if result['center'] is not None:
            all_positions.append(result['center'])
        if frame_idx % 50 == 0:
                status = 'LOST' if tracker.target_lost else 'OK'
                print(f"Frame {frame_idx}/{total_frames}: corr={result['corr_value']:.3f}, {status}")

        # Draw
        if tracker.current_pos is not None and not tracker.target_lost:
            cx, cy = tracker.current_pos
            x = cx - tracker.template_w // 2
            y = cy - tracker.template_h // 2
            draw_box(frame, x, y, tracker.template_w, tracker.template_h, color=(255, 0, 0))
            draw_circle(frame, cx, cy, radius=5, color=(0, 0, 255))

        pts = [(int(p[0]), int(p[1])) for p in all_positions]
        draw_trajectory(frame, pts, color=(0, 255, 255))

        status_text = f"Vehicle Track | Frame: {frame_idx} | Corr: {result.get('corr_value', 0):.3f}"
        if tracker.target_lost:
            status_text += " | LOST"
        put_text(frame, status_text)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"\nDone. Output: {OUTPUT_PATH}")
    print(f"Frames: {frame_idx}, Trajectory points: {len(all_positions)}")

    # Save trajectory
    traj_path = os.path.join(os.path.dirname(__file__), 'trajectory_task2.txt')
    with open(traj_path, 'w') as f:
        f.write("# frame cx cy\n")
        for i, (cx, cy) in enumerate(all_positions):
            f.write(f"{i} {cx:.1f} {cy:.1f}\n")
    print(f"Trajectory saved to: {traj_path}")


if __name__ == '__main__':
    main()
