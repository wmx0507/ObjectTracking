"""
Main tracking logic combining correlation matching, Kalman prediction,
multi-template strategy, multi-scale matching, template update, and lost-target handling.
"""

import numpy as np
from correlation import multi_template_match
from kalman_tracker import KalmanTracker


class ObjectTracker:
    """Tracks a single object across video frames."""

    def __init__(self,
                 templates: list[np.ndarray],
                 corr_threshold: float = 0.55,
                 search_margin: int = 120,
                 template_update_rate: float = 0.12,
                 lost_max_frames: int = 10,
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1,
                 use_multi_scale: bool = True,
                 use_edge_ncc: bool = False):

        self.templates = [t.astype(np.float64) for t in templates]
        self.use_multi_scale = use_multi_scale
        self.use_edge_ncc = use_edge_ncc
        self.corr_threshold = corr_threshold
        self.search_margin = search_margin
        self.update_rate = template_update_rate
        self.lost_max_frames = lost_max_frames

        self.kalman = KalmanTracker(dt=1.0,
                                     process_noise=process_noise,
                                     measurement_noise=measurement_noise)
        self.kalman_initialized = False

        self.lost_frames = 0
        self.target_lost = False

        self.current_pos = None
        self.trajectory = []

        # Current template dimensions (updated per-frame with multi-scale)
        self.template_h = templates[0].shape[0]
        self.template_w = templates[0].shape[1]

    def init_first_position(self, frame: np.ndarray,
                            init_pos: tuple[int, int] = None) -> tuple:
        """
        Find initial target position. Returns (best_pos, corr_val, corr_map).
        best_pos is top-left corner in original image coordinates.
        """
        if init_pos is not None:
            x0 = max(0, init_pos[0] - self.search_margin)
            y0 = max(0, init_pos[1] - self.search_margin)
            x1 = min(frame.shape[1], init_pos[0] + self.search_margin)
            y1 = min(frame.shape[0], init_pos[1] + self.search_margin)
            roi = frame[y0:y1, x0:x1]
            search_region = roi
            corr_map, best_idx, best_pos_local, corr_val, best_tpl = multi_template_match(
                roi, self.templates, use_multi_scale=self.use_multi_scale,
                use_edge=self.use_edge_ncc)
            best_pos = (best_pos_local[0] + x0, best_pos_local[1] + y0)
        else:
            search_region = frame
            corr_map, best_idx, best_pos, corr_val, best_tpl = multi_template_match(
                frame, self.templates, use_multi_scale=self.use_multi_scale,
                use_edge=self.use_edge_ncc)

        # multi_scale_match downsamples images with max_dim > 400 for speed.
        # When multi-scale is used, best_tpl.shape is in the (possibly downsampled)
        # search image coordinates; correct to original scale.
        if self.use_multi_scale:
            max_dim = 400
            downscale = min(1.0, max_dim / max(search_region.shape[0], search_region.shape[1]))
            self.template_h = max(1, int(best_tpl.shape[0] / downscale))
            self.template_w = max(1, int(best_tpl.shape[1] / downscale))
        else:
            self.template_h = best_tpl.shape[0]
            self.template_w = best_tpl.shape[1]

        cx = best_pos[0] + self.template_w // 2
        cy = best_pos[1] + self.template_h // 2
        self.current_pos = (cx, cy)
        self.trajectory.append((cx, cy))
        self.kalman.init(cx, cy)
        self.kalman_initialized = True
        return best_pos, corr_val, corr_map

    def track_frame(self, frame: np.ndarray) -> dict:
        """
        Process one frame. Returns a dict with tracking results.
        """
        result = {
            'pos': None,
            'center': None,
            'corr_value': 0.0,
            'lost': False,
            'predicted': False,
        }

        H, W = frame.shape[:2]

        if self.kalman_initialized:
            pred_x, pred_y = self.kalman.predict()
        else:
            return result

        # Use max template dims for ROI to ensure any template fits
        max_tw = max(t.shape[1] for t in self.templates)
        max_th = max(t.shape[0] for t in self.templates)
        x0 = max(0, int(pred_x) - self.search_margin - max_tw // 2)
        y0 = max(0, int(pred_y) - self.search_margin - max_th // 2)
        x1 = min(W, int(pred_x) + self.search_margin + max_tw // 2)
        y1 = min(H, int(pred_y) + self.search_margin + max_th // 2)

        # Ensure ROI is at least as large as the largest template,
        # even when the prediction is near image edges.
        min_roi_w = max_tw + 20
        min_roi_h = max_th + 20
        if x1 - x0 < min_roi_w:
            cx_roi = (x0 + x1) // 2
            x0 = max(0, cx_roi - min_roi_w // 2)
            x1 = min(W, x0 + min_roi_w)
            x0 = max(0, x1 - min_roi_w)
        if y1 - y0 < min_roi_h:
            cy_roi = (y0 + y1) // 2
            y0 = max(0, cy_roi - min_roi_h // 2)
            y1 = min(H, y0 + min_roi_h)
            y0 = max(0, y1 - min_roi_h)

        roi = frame[y0:y1, x0:x1]

        # Multi-template matching in search ROI (single-scale for speed)
        corr_map, best_idx, best_pos_local, corr_val, best_tpl = multi_template_match(
            roi, self.templates, use_multi_scale=False,
            use_edge=self.use_edge_ncc)

        best_x = best_pos_local[0] + x0
        best_y = best_pos_local[1] + y0

        # Use the matched template's actual dimensions
        matched_w = best_tpl.shape[1]
        matched_h = best_tpl.shape[0]
        cx = best_x + matched_w // 2
        cy = best_y + matched_h // 2

        if corr_val >= self.corr_threshold:
            self.kalman.update(cx, cy)
            self.lost_frames = 0
            self.target_lost = False

            # Update template dimensions
            self.template_w = matched_w
            self.template_h = matched_h

            # Template update via EMA (only if dimensions match)
            if matched_w == self.templates[best_idx].shape[1] and matched_h == self.templates[best_idx].shape[0]:
                patch = frame[best_y:best_y + matched_h,
                              best_x:best_x + matched_w].astype(np.float64)
                self.templates[best_idx] = (
                    self.update_rate * patch +
                    (1 - self.update_rate) * self.templates[best_idx]
                )
            else:
                self.templates[best_idx] = best_tpl.astype(np.float64)

            result['pos'] = (best_x, best_y)
            result['center'] = (cx, cy)
            result['corr_value'] = corr_val
        else:
            self.lost_frames += 1
            if self.lost_frames >= self.lost_max_frames:
                self.target_lost = True

            result['pos'] = (int(pred_x - self.template_w // 2),
                             int(pred_y - self.template_h // 2))
            result['center'] = (int(pred_x), int(pred_y))
            result['corr_value'] = corr_val
            result['predicted'] = True

        result['lost'] = self.target_lost
        self.current_pos = result['center']
        if not self.target_lost:
            self.trajectory.append(result['center'])

        return result
