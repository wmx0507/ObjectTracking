"""
Task 3: Locate the designated cyclist in the FIRST FRAME of the video.
No tracking yet — just detect and box the correct target.

Method:
  1. Extract the cyclist template from frame 82 (where the target image was taken).
  2. Use multi-scale NCC to find the cyclist in frame 0.
  3. Draw the bounding box and save the result.
"""

import os
import cv2
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '图像处理作业1')
VIDEO_PATH = os.path.join(DATA_DIR, '大疆无人机航拍骑车人.mp4')
TARGET_PATH = os.path.join(DATA_DIR, '大疆无人机航拍骑车人目标.png')


def imread_unicode(path: str) -> np.ndarray:
    """Read an image from a path with Unicode characters."""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def detect_yellow_circle(img: np.ndarray) -> tuple[int, int]:
    """Detect the yellow circle center in the target image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 80, 80), (35, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No yellow circle detected")
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    return int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert to float64 grayscale."""
    if img.ndim == 3:
        return np.mean(img, axis=2).astype(np.float64)
    return img.astype(np.float64)


def ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """FFT-accelerated Normalized Cross-Correlation."""
    img = to_gray(image)
    tpl = to_gray(template)
    h, w = tpl.shape
    H, W = img.shape
    if h > H or w > W:
        raise ValueError(f"Template {h}x{w} > image {H}x{W}")
    N = h * w
    mu_T = np.mean(tpl)
    sigma_T = np.std(tpl)
    if sigma_T < 1e-10:
        return np.zeros((H - h + 1, W - w + 1))
    tpl_padded = np.zeros((H, W))
    tpl_padded[:h, :w] = tpl
    cross = np.real(np.fft.ifft2(np.fft.fft2(img) * np.conj(np.fft.fft2(tpl_padded))))
    integral = np.zeros((H + 1, W + 1))
    integral_sq = np.zeros((H + 1, W + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    integral_sq[1:, 1:] = np.cumsum(np.cumsum(img ** 2, axis=0), axis=1)
    out_h, out_w = H - h + 1, W - w + 1
    result = np.empty((out_h, out_w))
    y2_arr = np.arange(h, H + 1)
    y1_arr = np.arange(0, out_h)
    x2_arr = np.arange(w, W + 1)
    x1_arr = np.arange(0, out_w)
    S_I = (integral[y2_arr[:, None], x2_arr[None, :]] -
           integral[y1_arr[:, None], x2_arr[None, :]] -
           integral[y2_arr[:, None], x1_arr[None, :]] +
           integral[y1_arr[:, None], x1_arr[None, :]])
    S_I2 = (integral_sq[y2_arr[:, None], x2_arr[None, :]] -
            integral_sq[y1_arr[:, None], x2_arr[None, :]] -
            integral_sq[y2_arr[:, None], x1_arr[None, :]] +
            integral_sq[y1_arr[:, None], x1_arr[None, :]])
    mu_I = S_I / N
    var_I = np.maximum(S_I2 / N - mu_I ** 2, 0.0)
    sigma_I = np.sqrt(var_I)
    numerator = cross[:out_h, :out_w] - N * mu_I * mu_T
    denominator = N * sigma_I * sigma_T
    safe = denominator > 1e-10
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=safe)


def cv2_resize(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def multi_scale_match(image, template, scales, max_image_dim=400):
    """Match template at multiple scales."""
    H_orig, W_orig = image.shape[:2]
    h_orig, w_orig = template.shape[:2]
    sd = 1.0
    if max(H_orig, W_orig) > max_image_dim:
        sd = max_image_dim / max(H_orig, W_orig)
        simg = cv2_resize(image, (max(1, int(W_orig * sd)), max(1, int(H_orig * sd))))
    else:
        simg = image
    best_val = -1.0
    best_pos = (0, 0)
    best_tpl = template
    for scale in scales:
        nw = max(5, int(w_orig * scale * sd))
        nh = max(5, int(h_orig * scale * sd))
        stpl = cv2_resize(template, (nw, nh))
        if stpl.shape[0] > simg.shape[0] or stpl.shape[1] > simg.shape[1]:
            continue
        cmap = ncc(simg, stpl)
        mv = np.max(cmap)
        if mv > best_val:
            best_val = mv
            ml = np.unravel_index(np.argmax(cmap), cmap.shape)
            best_pos = (ml[1], ml[0])
            best_tpl = stpl
    if sd != 1.0:
        pos_orig = (int(best_pos[0] / sd), int(best_pos[1] / sd))
    else:
        pos_orig = best_pos
    return best_val, pos_orig, best_tpl


def main():
    # ── Step 1: Extract cyclist template from frame 82 ──

    # Read target image, find yellow circle
    target = imread_unicode(TARGET_PATH)
    cx_y, cy_y = detect_yellow_circle(target)
    print(f"Target image: {target.shape[1]}x{target.shape[0]}, yellow=({cx_y},{cy_y})")

    # Extract 150x150 context around yellow circle
    half = 150
    ctx = target[max(0,cy_y-half):min(target.shape[0],cy_y+half),
                 max(0,cx_y-half):min(target.shape[1],cx_y+half)]
    local_cx = cx_y - max(0, cx_y - half)
    local_cy = cy_y - max(0, cy_y - half)
    print(f"Context template: {ctx.shape}, local_center=({local_cx},{local_cy})")

    # Read frame 82
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 82)
    ret, f82 = cap.read()
    if not ret:
        raise RuntimeError("Cannot read frame 82")

    # Match context to frame 82 → find yellow circle in video
    corr_ctx, pos_ctx, _ = multi_scale_match(f82, ctx,
        scales=(0.4, 0.5, 0.6, 0.7, 0.8), max_image_dim=800)
    scale_ctx = corr_ctx  # we need scale separately
    # redo to get scale
    best_val = -1
    best_scale = 0.7
    best_pos = (0, 0)
    for s in (0.4, 0.5, 0.6, 0.7, 0.8):
        sd = min(1.0, 800/max(f82.shape[0], f82.shape[1]))
        nw = max(5, int(ctx.shape[1]*s*sd))
        nh = max(5, int(ctx.shape[0]*s*sd))
        stpl = cv2_resize(ctx, (nw, nh))
        sf = cv2_resize(f82, (int(f82.shape[1]*sd), int(f82.shape[0]*sd))) if sd != 1 else f82
        if stpl.shape[0] > sf.shape[0] or stpl.shape[1] > sf.shape[1]:
            continue
        cmap = ncc(sf, stpl)
        mv = np.max(cmap)
        if mv > best_val:
            best_val = mv
            best_scale = s
            ml = np.unravel_index(np.argmax(cmap), cmap.shape)
            best_pos = (int(ml[1]/sd), int(ml[0]/sd)) if sd != 1 else (ml[1], ml[0])

    vx_yellow = int(best_pos[0] + local_cx * best_scale)
    vy_yellow = int(best_pos[1] + local_cy * best_scale)
    print(f"Yellow circle mapped to frame82: ({vx_yellow}, {vy_yellow}), "
          f"corr={best_val:.3f}, scale={best_scale:.2f}")

    # Scan for cyclist in a 100x100 region around yellow circle in frame82
    scan_half = 50
    sx1 = max(0, vx_yellow - scan_half); sy1 = max(0, vy_yellow - scan_half)
    sx2 = min(640, vx_yellow + scan_half); sy2 = min(512, vy_yellow + scan_half)
    scan_roi = f82[sy1:sy2, sx1:sx2]
    gray_roi = to_gray(scan_roi)
    road_mean = float(np.mean(gray_roi))

    # Find the cyclist: dark-ish small blob with internal structure
    best_score = 0.0
    best_cx, best_cy, best_sz = 0, 0, 0
    for sz in [20, 25, 30, 35]:
        for y in range(10, scan_roi.shape[0] - sz - 10, 2):
            for x in range(10, scan_roi.shape[1] - sz - 10, 2):
                pg = gray_roi[y:y+sz, x:x+sz]
                pm = float(np.mean(pg))
                ps = float(np.std(pg))
                if pm < road_mean - 3 and ps > 12:
                    score = ps * (road_mean - pm)
                    if score > best_score:
                        best_score = score
                        best_cx = x + sz // 2
                        best_cy = y + sz // 2
                        best_sz = sz

    # Map back to frame82 coordinates
    cyclist_x = sx1 + best_cx
    cyclist_y = sy1 + best_cy
    print(f"Cyclist found in frame82: ({cyclist_x}, {cyclist_y}), "
          f"est_size={best_sz}, score={best_score:.0f}")

    # Extract clean cyclist template from frame82
    half_sz = best_sz // 2
    tx1 = max(0, cyclist_x - half_sz); ty1 = max(0, cyclist_y - half_sz)
    tx2 = min(640, cyclist_x + half_sz); ty2 = min(512, cyclist_y + half_sz)
    cyclist_tpl = f82[ty1:ty2, tx1:tx2].astype(np.float64)
    print(f"Cyclist template: {cyclist_tpl.shape}")

    # ── Step 2: Find cyclist in frame 0 ──

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, f0 = cap.read()
    cap.release()

    corr_val, pos0, best_tpl0 = multi_scale_match(f0, cyclist_tpl,
        scales=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5), max_image_dim=800)
    tpl_w = best_tpl0.shape[1]
    tpl_h = best_tpl0.shape[0]
    cx0 = pos0[0] + tpl_w // 2
    cy0 = pos0[1] + tpl_h // 2
    print(f"Frame 0: cyclist_at=({cx0},{cy0}), size={tpl_w}x{tpl_h}, corr={corr_val:.3f}")

    # ── Step 3: Draw bounding box on frame 0 ──

    result = f0.copy()
    bx = pos0[0]
    by = pos0[1]
    cv2.rectangle(result, (bx, by), (bx + tpl_w, by + tpl_h), (0, 255, 255), 3)
    cv2.circle(result, (cx0, cy0), 6, (0, 0, 255), -1)
    cv2.putText(result,
        f"Cyclist at ({cx0},{cy0}), corr={corr_val:.3f}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = os.path.join(os.path.dirname(__file__), 'output_task3_init_detect.png')
    cv2.imencode('.png', result)[1].tofile(out_path)
    print(f"\nSaved: {out_path}")
    print("Please verify if the yellow box correctly marks the cyclist.")


if __name__ == '__main__':
    main()
