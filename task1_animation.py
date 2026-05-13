"""
Task 1: Track the cartoon emoji character in 动画表情视频.mp4.
Three templates provided: small_template_1/2/3.png
Output: bounding box, center position, and complete trajectory.
"""

import os
import cv2
import numpy as np
from tracker import ObjectTracker
from utils import read_video, create_writer, draw_box, draw_trajectory, draw_circle, put_text, imread_unicode

DATA_DIR = os.path.join(os.path.dirname(__file__), '图像处理作业1')
VIDEO_PATH = os.path.join(DATA_DIR, '动画表情视频.mp4')
TEMPLATE_PATHS = [
    os.path.join(DATA_DIR, 'small_template_1.png'),
    os.path.join(DATA_DIR, 'small_template_2.png'),
    os.path.join(DATA_DIR, 'small_template_3.png'),
]
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output_task1.mp4')


def main():
    # Load templates
    templates = []
    for p in TEMPLATE_PATHS:
        tpl = imread_unicode(p)
        if tpl is None:
            raise FileNotFoundError(f"Cannot read template: {p}")
        templates.append(tpl)
        print(f"Loaded template {os.path.basename(p)}: {tpl.shape}")

    # Initialize tracker - use multi-scale matching to handle scale differences
    # between the close-up templates and the smaller video character.
    tracker = ObjectTracker(
        templates=templates,
        corr_threshold=0.35,
        search_margin=150,
        template_update_rate=0.15,
        lost_max_frames=20,
        use_multi_scale=True,  # Multi-scale handles the scale difference
    )

    # Read first frame to get video info
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height}, {fps:.2f} fps, {total_frames} frames")

    # Create output writer
    writer = create_writer(OUTPUT_PATH, fps, width, height)

    frame_idx = 0
    all_positions = []  # Store all trajectory points for final rendering

    for frame in read_video(VIDEO_PATH):
        result = {'corr_value': 0.0, 'center': None, 'lost': False}

        if frame_idx == 0:
            # First frame: search whole image for initial position using provided templates
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

        # Draw current frame
        if tracker.current_pos is not None and not tracker.target_lost:
            cx, cy = tracker.current_pos
            x = cx - tracker.template_w // 2
            y = cy - tracker.template_h // 2
            draw_box(frame, x, y, tracker.template_w, tracker.template_h, color=(0, 255, 0))
            draw_circle(frame, cx, cy, radius=4, color=(0, 0, 255))

        # Draw trajectory so far
        pts = [(int(p[0]), int(p[1])) for p in all_positions]
        draw_trajectory(frame, pts, color=(0, 255, 255))

        # Status overlay
        status_text = f"Frame: {frame_idx} | Corr: {result.get('corr_value', 0):.3f}"
        if tracker.target_lost:
            status_text += " | LOST"
        put_text(frame, status_text)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    print(f"\nDone. Output saved to: {OUTPUT_PATH}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Trajectory points: {len(all_positions)}")

    # Save trajectory data
    traj_path = os.path.join(os.path.dirname(__file__), 'trajectory_task1.txt')
    with open(traj_path, 'w') as f:
        f.write("# frame cx cy\n")
        for i, (cx, cy) in enumerate(all_positions):
            f.write(f"{i} {cx:.1f} {cy:.1f}\n")
    print(f"Trajectory saved to: {traj_path}")


if __name__ == '__main__':
    main()
