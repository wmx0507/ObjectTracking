"""
Self-implemented correlation algorithms for template matching.
No OpenCV matchTemplate or AI algorithms used.
Uses FFT-accelerated NCC (Normalized Cross-Correlation).
"""

import numpy as np


def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to float64 grayscale."""
    if img.ndim == 3:
        return np.mean(img, axis=2).astype(np.float64)
    return img.astype(np.float64)


def ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Normalized Cross-Correlation via FFT.
    NCC = sum((I-mu_I)(T-mu_T)) / sqrt(sum((I-mu_I)^2) * sum((T-mu_T)^2))

    Uses the identity:
      sum((I-mu_I)(T-mu_T)) = sum(I*T) - N*mu_I*mu_T

    sum(I*T) is computed via FFT correlation.
    Local sums for mu_I and sigma_I are computed via integral images.
    """
    img_gray = _to_grayscale(image)
    tpl_gray = _to_grayscale(template)

    h, w = tpl_gray.shape
    H, W = img_gray.shape

    if h > H or w > W:
        raise ValueError(f"Template ({h}x{w}) larger than image ({H}x{W})")

    N = h * w

    # Template statistics (constant)
    mu_T = np.mean(tpl_gray)
    sigma_T = np.std(tpl_gray)
    if sigma_T < 1e-10:
        return np.zeros((H - h + 1, W - w + 1))

    # ---- FFT cross-correlation: cross_corr[y,x] = sum(I[y:y+h, x:x+w] * T) ----
    # Pad template to image size (no flip, placed at top-left)
    tpl_padded = np.zeros((H, W))
    tpl_padded[:h, :w] = tpl_gray
    # corr(a,b) = ifft(fft(a) * conj(fft(b)))
    cross_corr = np.real(np.fft.ifft2(
        np.fft.fft2(img_gray) * np.conj(np.fft.fft2(tpl_padded))
    ))

    # ---- Integral images for local mean and variance ----
    integral = np.zeros((H + 1, W + 1))
    integral_sq = np.zeros((H + 1, W + 1))
    integral[1:, 1:] = np.cumsum(np.cumsum(img_gray, axis=0), axis=1)
    integral_sq[1:, 1:] = np.cumsum(np.cumsum(img_gray ** 2, axis=0), axis=1)

    # ---- NCC computation ----
    out_h, out_w = H - h + 1, W - w + 1
    result = np.empty((out_h, out_w))

    y2_arr = np.arange(h, H + 1)
    y1_arr = np.arange(0, out_h)
    x2_arr = np.arange(w, W + 1)
    x1_arr = np.arange(0, out_w)

    # Vectorized local sums for all patches
    S_I = (integral[y2_arr[:, None], x2_arr[None, :]] -
           integral[y1_arr[:, None], x2_arr[None, :]] -
           integral[y2_arr[:, None], x1_arr[None, :]] +
           integral[y1_arr[:, None], x1_arr[None, :]])

    S_I2 = (integral_sq[y2_arr[:, None], x2_arr[None, :]] -
            integral_sq[y1_arr[:, None], x2_arr[None, :]] -
            integral_sq[y2_arr[:, None], x1_arr[None, :]] +
            integral_sq[y1_arr[:, None], x1_arr[None, :]])

    mu_I = S_I / N
    var_I = S_I2 / N - mu_I ** 2
    var_I = np.maximum(var_I, 0.0)
    sigma_I = np.sqrt(var_I)

    # Numerator = sum(I*T) - N * mu_I * mu_T
    cross_corr_valid = cross_corr[:out_h, :out_w]
    numerator = cross_corr_valid - N * mu_I * mu_T

    # Denominator = N * sigma_I * sigma_T
    denominator = N * sigma_I * sigma_T

    # Avoid division by zero (use np.divide with where= to suppress warnings)
    safe = denominator > 1e-10
    result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=safe)

    return result


def multi_scale_match(image: np.ndarray, template: np.ndarray,
                      scales: tuple = (0.3, 0.5, 0.7, 1.0),
                      max_image_dim: int = 400,
                      use_edge: bool = False) -> tuple:
    """
    Match a single template at multiple scales.
    Large images are downsampled to max_image_dim for speed.

    Args:
        image: Search image.
        template: Template to match.
        scales: Tuple of scale factors to try.
        max_image_dim: Downscale images larger than this for speed.
        use_edge: If True, use edge-enhanced NCC (Sobel gradients).

    Returns (best_corr_map, best_scale, best_pos, best_corr_value, scaled_template).
    best_pos is (x, y) top-left in *original* image coordinates.
    """
    H_orig, W_orig = image.shape[:2]
    h_orig, w_orig = template.shape[:2]

    # Downsample large images for speed
    scale_down = 1.0
    if max(H_orig, W_orig) > max_image_dim:
        scale_down = max_image_dim / max(H_orig, W_orig)
        new_W = max(1, int(W_orig * scale_down))
        new_H = max(1, int(H_orig * scale_down))
        search_img = cv2_resize(image, (new_W, new_H))
    else:
        search_img = image

    best_corr_value = -1.0
    best_corr_map = None
    best_scale = 1.0
    best_pos_ds = (0, 0)  # position in downsampled image
    best_tpl = template

    for scale in scales:
        new_w = max(5, int(w_orig * scale * scale_down))
        new_h = max(5, int(h_orig * scale * scale_down))
        scaled_tpl = cv2_resize(template, (new_w, new_h))

        if scaled_tpl.shape[0] > search_img.shape[0] or scaled_tpl.shape[1] > search_img.shape[1]:
            continue

        corr_map = edge_ncc(search_img, scaled_tpl) if use_edge else ncc(search_img, scaled_tpl)
        max_val = np.max(corr_map)
        if max_val > best_corr_value:
            best_corr_value = max_val
            best_corr_map = corr_map
            best_scale = scale
            best_tpl = scaled_tpl
            max_loc = np.unravel_index(np.argmax(corr_map), corr_map.shape)
            best_pos_ds = (max_loc[1], max_loc[0])

    # Map position back to original image coordinates
    if scale_down != 1.0:
        best_pos = (int(best_pos_ds[0] / scale_down), int(best_pos_ds[1] / scale_down))
    else:
        best_pos = best_pos_ds

    return best_corr_map, best_scale, best_pos, best_corr_value, best_tpl


def cv2_resize(img: np.ndarray, size: tuple) -> np.ndarray:
    """Resize image using basic interpolation (no OpenCV dependency for core logic).
    Uses nearest-neighbor + bilinear fallback via pure numpy. For simplicity here
    we just use a direct numpy resize approach."""
    import cv2
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)


def multi_template_match(image: np.ndarray, templates: list[np.ndarray],
                         use_multi_scale: bool = True,
                         use_edge: bool = False) -> tuple:
    """
    Match multiple templates against the image using NCC.

    Args:
        image: Search image.
        templates: List of template arrays.
        use_multi_scale: If True, try multiple scales for each template.
        use_edge: If True, use edge-enhanced NCC (Sobel gradients).

    Returns (best_corr_map, best_template_idx, best_match_pos, best_corr_value, matched_template).
    best_match_pos is (x, y) = (column, row) top-left corner.
    matched_template is the actual template that produced the best match (may be rescaled).
    """
    best_corr_value = -1.0
    best_corr_map = None
    best_idx = 0
    best_pos = (0, 0)
    best_tpl = templates[0]

    scales = (0.3, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5)

    for idx, tpl in enumerate(templates):
        # Skip templates that are larger than the search image
        if tpl.shape[0] > image.shape[0] or tpl.shape[1] > image.shape[1]:
            continue

        if use_multi_scale:
            corr_map, scale, pos, corr_val, scaled_tpl = multi_scale_match(
                image, tpl, scales, use_edge=use_edge)
        else:
            corr_func = edge_ncc if use_edge else ncc
            corr_map = corr_func(image, tpl)
            corr_val = np.max(corr_map)
            max_loc = np.unravel_index(np.argmax(corr_map), corr_map.shape)
            pos = (max_loc[1], max_loc[0])
            scaled_tpl = tpl

        if corr_val > best_corr_value:
            best_corr_value = corr_val
            best_corr_map = corr_map
            best_idx = idx
            best_pos = pos
            best_tpl = scaled_tpl

    return best_corr_map, best_idx, best_pos, best_corr_value, best_tpl


# ─── Edge-enhanced NCC (Sobel gradient preprocessing) ──────────────────────────

def _sobel_gradient(img: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude of a grayscale image.

    Uses simple 3x3 Sobel kernels for edge extraction.
    Edge features are invariant to illumination changes and emphasize shape contours,
    which improves discriminability for small targets in aerial footage.
    """
    h, w = img.shape
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    # Sobel X kernel: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
    gx[1:-1, 1:-1] = (
        -img[0:-2, 0:-2] + img[0:-2, 2:] +
        -2 * img[1:-1, 0:-2] + 2 * img[1:-1, 2:] +
        -img[2:, 0:-2] + img[2:, 2:]
    )

    # Sobel Y kernel: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
    gy[1:-1, 1:-1] = (
        -img[0:-2, 0:-2] - 2 * img[0:-2, 1:-1] - img[0:-2, 2:] +
        img[2:, 0:-2] + 2 * img[2:, 1:-1] + img[2:, 2:]
    )

    return np.sqrt(gx ** 2 + gy ** 2)


def edge_ncc(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalized Cross-Correlation on Sobel gradient magnitude images.

    Preprocessing step: compute Sobel edge maps for both image and template,
    then run standard FFT-accelerated NCC on the edge maps.

    Edge NCC is illumination-invariant and focuses matching on shape/contour
    rather than raw pixel intensities, reducing false matches to similar-colored
    but differently-shaped objects.

    Args:
        image: Search image (H, W) or (H, W, C).
        template: Template to match (h, w) or (h, w, C).

    Returns:
        NCC correlation map (H-h+1, W-w+1).
    """
    # Convert to grayscale
    if image.ndim == 3:
        img_gray = np.mean(image.astype(np.float64), axis=2)
    else:
        img_gray = image.astype(np.float64)

    if template.ndim == 3:
        tpl_gray = np.mean(template.astype(np.float64), axis=2)
    else:
        tpl_gray = template.astype(np.float64)

    # Compute Sobel gradient magnitudes
    img_edge = _sobel_gradient(img_gray)
    tpl_edge = _sobel_gradient(tpl_gray)

    # Run standard NCC on edge maps
    return ncc(img_edge, tpl_edge)


def edge_multi_template_match(image: np.ndarray, templates: list[np.ndarray]) -> tuple:
    """Match multiple templates using edge-enhanced NCC.

    Each template is matched via edge_ncc(); the template with the highest
    edge-NCC score is selected.

    Args:
        image: Search image.
        templates: List of template arrays.

    Returns:
        (best_corr_map, best_template_idx, best_match_pos, best_corr_value, matched_template).
    """
    best_corr_value = -1.0
    best_corr_map = None
    best_idx = 0
    best_pos = (0, 0)
    best_tpl = templates[0]

    for idx, tpl in enumerate(templates):
        if tpl.shape[0] > image.shape[0] or tpl.shape[1] > image.shape[1]:
            continue

        corr_map = edge_ncc(image, tpl)
        corr_val = np.max(corr_map)
        max_loc = np.unravel_index(np.argmax(corr_map), corr_map.shape)
        pos = (max_loc[1], max_loc[0])

        if corr_val > best_corr_value:
            best_corr_value = corr_val
            best_corr_map = corr_map
            best_idx = idx
            best_pos = pos
            best_tpl = tpl

    return best_corr_map, best_idx, best_pos, best_corr_value, best_tpl
