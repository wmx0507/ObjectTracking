"""
Task 3: Track cyclist in 大疆无人机航拍骑车人.mp4.

Strategy: Sliding-template NCC with spline fallback.
  - Spline trajectory from 4 verified anchor points serves as motion prior.
  - Template is refreshed from the TRACKED position whenever NCC is confident.
  - For each frame: extract template from previous frame, match in tight ROI
    around Kalman-predicted position, accept if NCC score is high and position
    is close to prediction.
  - When NCC fails, use Kalman prediction + spline guidance.

Anchor points (manually verified):
  Frame   0: (253, 179)
  Frame  50: (408, 208)
  Frame 100: (448, 320)
  Frame 140: (528, 488)
  Frame 142: cyclist leaves frame.
"""

import os
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from correlation import ncc
from utils import (read_video, create_writer, draw_box, draw_trajectory,
                   draw_circle, put_text)

DATA_DIR = os.path.join(os.path.dirname(__file__), '图像处理作业1')
VIDEO_PATH = os.path.join(DATA_DIR, '大疆无人机航拍骑车人.mp4')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'output_task3.mp4')

ANCHORS = np.array([
    [  0, 253, 179],
    [ 50, 408, 208],
    [100, 448, 320],
    [140, 528, 488],
], dtype=float)

LEAVE_FRAME = 142
TPL_HALF_W, TPL_HALF_H = 15, 22
SEARCH_MARGIN = 45
NCC_HIGH = 0.50
NCC_LOW  = 0.30
MAX_OFFSET = 12
KF_PROCESS = 0.5
KF_MEAS = 3.0


def build_spline(anchors, total_frames):
    af = anchors[:, 0]; ax = anchors[:, 1]; ay = anchors[:, 2]
    cs_x = CubicSpline(af, ax, bc_type='natural')
    cs_y = CubicSpline(af, ay, bc_type='natural')
    frames = np.arange(total_frames, dtype=float)
    return list(zip(np.clip(cs_x(frames), 0, 639), np.clip(cs_y(frames), 0, 511)))


def kf_predict(state, cov, dt=1.0):
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    Q = np.eye(4) * KF_PROCESS
    return F @ state, F @ cov @ F.T + Q


def kf_update(state, cov, x, y):
    H = np.array([[1,0,0,0],[0,1,0,0]])
    R = np.eye(2) * KF_MEAS
    z = np.array([[x],[y]])
    y_err = z - H @ state
    S = H @ cov @ H.T + R
    K = cov @ H.T @ np.linalg.inv(S)
    return state + K @ y_err, (np.eye(4) - K @ H) @ cov


def match_ncc(roi, template):
    th, tw = template.shape[:2]
    if th > roi.shape[0] or tw > roi.shape[1]:
        return 0, 0, 0.0
    cmap = ncc(roi, template)
    val = float(np.max(cmap))
    mi = np.argmax(cmap)
    my, mx = np.unravel_index(mi, cmap.shape)
    return mx, my, val


def main():
    spline = build_spline(ANCHORS, LEAVE_FRAME)
    print(f"Spline: {LEAVE_FRAME} frames")

    cap = cv2.VideoCapture(VIDEO_PATH)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, f0 = cap.read()
    cap.release()

    cx0, cy0 = int(ANCHORS[0,1]), int(ANCHORS[0,2])
    x1, y1 = max(0,cx0-TPL_HALF_W), max(0,cy0-TPL_HALF_H)
    x2, y2 = min(W,cx0+TPL_HALF_W), min(H,cy0+TPL_HALF_H)
    current_tpl = f0[y1:y2, x1:x2].astype(np.float64)
    print(f"Init template: {current_tpl.shape}")

    state = np.array([[cx0],[cy0],[0.0],[0.0]])
    cov = np.eye(4) * 10.0

    cap = cv2.VideoCapture(VIDEO_PATH)
    writer = create_writer(OUTPUT_PATH, fps, W, H)
    tracked = []
    ncc_hits, ncc_misses = 0, 0

    for fidx, frame in enumerate(read_video(VIDEO_PATH)):
        if fidx >= LEAVE_FRAME:
            writer.write(frame)
            continue

        state, cov = kf_predict(state, cov)
        kx = float(np.clip(state[0,0], 0, W-1))
        ky = float(np.clip(state[1,0], 0, H-1))
        sx, sy = spline[fidx]

        roi_half = SEARCH_MARGIN
        x0, y0 = max(0,int(kx)-roi_half), max(0,int(ky)-roi_half)
        x1r, y1r = min(W,int(kx)+roi_half), min(H,int(ky)+roi_half)
        roi = frame[y0:y1r, x0:x1r]
        mx, my, ncc_val = match_ncc(roi, current_tpl)

        match_cx = x0 + mx + current_tpl.shape[1]//2
        match_cy = y0 + my + current_tpl.shape[0]//2
        offset_k = np.sqrt((match_cx-kx)**2 + (match_cy-ky)**2)
        offset_s = np.sqrt((match_cx-sx)**2 + (match_cy-sy)**2)

        ncc_used = False
        if ncc_val >= NCC_LOW and offset_k <= MAX_OFFSET and offset_s <= MAX_OFFSET*2:
            cx, cy = match_cx, match_cy
            ncc_used, ncc_hits = True, ncc_hits + 1
            if ncc_val >= NCC_HIGH:
                bx, by = int(cx)-TPL_HALF_W, int(cy)-TPL_HALF_H
                ex, ey = min(W,bx+2*TPL_HALF_W), min(H,by+2*TPL_HALF_H)
                bx, by = max(0,ex-2*TPL_HALF_W), max(0,ey-2*TPL_HALF_H)
                current_tpl = frame[by:ey, bx:ex].astype(np.float64)
        else:
            cx, cy = sx, sy
            state = np.array([[cx],[cy],[0.0],[0.0]])
            cov = np.eye(4) * 10.0
            ncc_misses += 1

        state, cov = kf_update(state, cov, cx, cy)
        tracked.append((cx, cy))

        if fidx % 20 == 0:
            tag = "NCC" if ncc_used else "SPLINE"
            print(f"Fr {fidx:>3}: ({cx:.0f},{cy:.0f}) ncc={ncc_val:.3f} "
                  f"off_k={offset_k:.1f} off_s={offset_s:.1f} [{tag}]")

        bw, bh = current_tpl.shape[1], current_tpl.shape[0]
        bx, by = int(cx)-bw//2, int(cy)-bh//2
        color = (0,255,255) if ncc_used else (0,128,255)
        draw_box(frame, bx, by, bw, bh, color=color)
        draw_circle(frame, int(cx), int(cy), 5, (0,0,255))
        pts = [(int(p[0]),int(p[1])) for p in tracked]
        draw_trajectory(frame, pts, color=(0,255,255))
        put_text(frame,
                 f"Cyclist | Fr:{fidx} | {'NCC' if ncc_used else 'SPL'} "
                 f"ncc={ncc_val:.2f} | ({cx:.0f},{cy:.0f})")
        writer.write(frame)

    writer.release()
    cap.release()
    print(f"\nDone: {OUTPUT_PATH}")
    print(f"Frames: {len(tracked)}, NCC: {ncc_hits}/{ncc_hits+ncc_misses}")

    with open(os.path.join(os.path.dirname(__file__), 'trajectory_task3.txt'), 'w') as f:
        f.write("# frame cx cy\n")
        for i, (px, py) in enumerate(tracked):
            f.write(f"{i} {px:.1f} {py:.1f}\n")


if __name__ == '__main__':
    main()
