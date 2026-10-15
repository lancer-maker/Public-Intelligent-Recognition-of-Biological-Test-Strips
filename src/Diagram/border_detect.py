
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

Rect = Tuple[float, float, float, float]


def detect_red_vertical_lines(warped: np.ndarray) -> List[int]:

    b, g, r = cv2.split(warped.astype(np.int16))
    red = ((r - g > 40) & (r - b > 40) & (r > 110)).astype(np.uint8)
    h = warped.shape[0]
    colsum = red.sum(axis=0)
    xs = [x for x in range(warped.shape[1]) if colsum[x] > config.RED_LINE_COVER_FRAC * h]
    groups: List[List[int]] = []
    for x in xs:
        if groups and x - groups[-1][-1] <= 5:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [int(np.mean(g)) for g in groups]


def detect_red_boxes(image: np.ndarray) -> Optional[dict]:

    h, w = image.shape[:2]
    b, g, r = cv2.split(image.astype(np.int16))
    mask = ((r - g > config.RED_DELTA) & (r - b > config.RED_DELTA)
            & (r > config.RED_MIN)).astype(np.uint8) * 255
    if int((mask > 0).sum()) < 200:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    min_area = config.RED_BOX_MIN_AREA_FRAC * w * h
    a_lo, a_hi = config.RED_BOX_ASPECT_RANGE
    boxes: List[Tuple[int, int, int, int]] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh <= 0 or bw * bh < min_area:
            continue
        if not (a_lo <= bw / bh <= a_hi):
            continue
        boxes.append((x, y, x + bw, y + bh))
    if len(boxes) < config.RED_BOX_MIN_COUNT:
        return None

    top_frac, bot_frac = config.RED_BOX_ROW_SPLIT
    top: List[Tuple[int, int, int, int]] = []
    bottom: List[Tuple[int, int, int, int]] = []
    middle: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        cy = (box[1] + box[3]) / 2.0 / h
        if cy < top_frac:
            top.append(box)
        elif cy > bot_frac:
            bottom.append(box)
        else:
            middle.append(box)
    if len(top) != 4 or len(bottom) != 4:
        return None

    top.sort(key=lambda t: (t[0] + t[2]) / 2.0)
    bottom.sort(key=lambda t: (t[0] + t[2]) / 2.0)
    test: Dict[str, Tuple[int, int, int, int]] = {}
    for tag, box in zip(("1", "3", "5", "7"), top):
        test[tag] = box
    for tag, box in zip(("2", "4", "6", "8"), bottom):
        test[tag] = box

    if middle:
        qc = max(middle, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))
    else:
        qc = (max(box[2] for box in top[:2]), min(box[1] for box in top),
              min(box[0] for box in top[2:]), max(box[3] for box in bottom))
    return {"test": test, "qc": qc, "size": (int(w), int(h))}


def red_boxes_to_rois(detected: dict) -> Dict[str, Rect]:

    w, h = detected["size"]
    rois: Dict[str, Rect] = {}
    for tag in config.TEST_TAGS:
        box = detected["test"].get(tag)
        if box is not None:
            rois[tag] = (box[0] / w, box[1] / h, box[2] / w, box[3] / h)
    qc = detected.get("qc")
    if qc is not None:
        rois[config.QC_TAG] = (qc[0] / w, qc[1] / h, qc[2] / w, qc[3] / h)
    return rois


def _column_valleys(warped: np.ndarray) -> List[Tuple[int, float]]:

    if not _HAS_SCIPY:
        return []
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    band = gray[int(0.10 * h):int(0.90 * h), :]
    profile = gaussian_filter1d(band.mean(axis=0), 4)
    distance = max(5, int(config.COL_PEAK_DISTANCE_FRAC * w))
    peaks, props = find_peaks(-profile, distance=distance,
                              prominence=config.MIN_WALL_PROMINENCE * 0.5)
    return [(int(p), float(pr)) for p, pr in zip(peaks, props["prominences"])]


def _select_six_boundaries(valleys: List[Tuple[int, float]], width: int) -> Optional[List[int]]:

    if len(valleys) < 6:
        return None
    min_gap = 0.10 * width
    chosen: List[Tuple[int, float]] = []
    for x, prom in sorted(valleys, key=lambda t: -t[1]):
        if all(abs(x - cx) >= min_gap for cx, _ in chosen):
            chosen.append((x, prom))
        if len(chosen) == 6:
            break
    if len(chosen) < 6:
        return None
    xs = sorted(x for x, _ in chosen)
    gaps = np.diff(xs)
    lo, hi = config.COL_WIDTH_RANGE
    if np.any(gaps < lo * width) or np.any(gaps > hi * width):
        return None
    if float(np.std(gaps)) > 0.35 * float(np.mean(gaps)):
        return None
    return [int(x) for x in xs]


def _columns_from_boundaries(boundaries: Sequence[int], width: int) -> List[Tuple[float, float]]:

    v = list(boundaries)
    wall = config.WALL_WIDTH_FRAC * width
    return [
        (float(v[0]), float(v[1] - wall)),
        (float(v[1]), float(v[2] - wall)),
        (float(v[2]), float(v[3] - wall)),
        (float(v[3]), float(v[4] - wall)),
        (float(v[4]), float(v[5])),
    ]


def _row_edges(warped: np.ndarray, columns: Sequence[Tuple[float, float]]
               ) -> Tuple[float, float, float]:

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    mask = np.zeros(w, dtype=bool)
    for x0, x1 in columns:
        mask[max(0, int(x0)):min(w, int(x1))] = True
    if not mask.any():
        return 0.03 * h, 0.50 * h, 0.97 * h

    profile = gaussian_filter1d(gray[:, mask].mean(axis=1), 4)
    fallback = (0.03 * h, 0.50 * h, 0.97 * h)
    if not _HAS_SCIPY:
        return fallback

    peaks, props = find_peaks(profile, distance=max(5, int(0.06 * h)),
                              prominence=config.ROW_PEAK_PROMINENCE)
    if len(peaks) == 0:
        return fallback

    def _best_in(lo_frac: float, hi_frac: float) -> Optional[int]:
        sel = [(int(p), float(pr)) for p, pr in zip(peaks, props["prominences"])
               if lo_frac * h <= p <= hi_frac * h]
        return max(sel, key=lambda t: t[1])[0] if sel else None

    top = _best_in(0.0, config.ROW_EDGE_SEARCH) or fallback[0]
    bot = _best_in(1.0 - config.ROW_EDGE_SEARCH, 1.0) or fallback[2]
    lo, hi = config.ROW_DIVIDER_SEARCH
    div = _best_in(lo, hi) or fallback[1]
    if not (top < div < bot):
        top, div, bot = fallback
    return float(top), float(div), float(bot)


def detect_grid(warped: np.ndarray, mode: str = None) -> Optional[dict]:

    mode = (mode or config.ROI_DETECT_MODE).lower()
    h, w = warped.shape[:2]

    columns: Optional[List[Tuple[float, float]]] = None
    method = ""

    if mode in ("auto", "red"):
        red = detect_red_vertical_lines(warped)
        if len(red) >= 10:
            red = sorted(red)[:10]
            columns = [(float(red[i]), float(red[i + 1])) for i in range(0, 10, 2)]
            method = "red-lines"
        elif mode == "red":
            return None

    if columns is None and mode in ("auto", "profile"):
        valleys = _column_valleys(warped)
        boundaries = _select_six_boundaries(valleys, w)
        if boundaries is not None:
            columns = _columns_from_boundaries(boundaries, w)
            method = "profile-valleys"

    if columns is None:
        return None

    lo, hi = config.COL_WIDTH_RANGE
    for x0, x1 in columns:
        if not (lo * w <= (x1 - x0) <= hi * w):
            return None

    top, div, bot = _row_edges(warped, columns)
    return {
        "columns": columns,
        "rows": (top, div, bot),
        "size": (int(w), int(h)),
        "method": method,
    }


def grid_to_rois(grid: dict) -> Dict[str, Rect]:

    w, h = grid["size"]
    columns = grid["columns"]
    top, div, bot = grid["rows"]

    x_inset = config.ROI_X_INSET_FRAC
    y_inset = config.ROI_Y_INSET_FRAC
    rois: Dict[str, Rect] = {}

    def _rect(x0: float, y0: float, x1: float, y1: float) -> Rect:

        dx = (x1 - x0) * x_inset
        dy = (y1 - y0) * y_inset
        return ((x0 + dx) / w, (y0 + dy) / h, (x1 - dx) / w, (y1 - dy) / h)

    for col_idx, (top_tag, bot_tag) in config.COLUMN_TAG_MAP.items():
        if col_idx >= len(columns):
            continue
        cx0, cx1 = columns[col_idx]
        rois[top_tag] = _rect(cx0, top, cx1, div)
        rois[bot_tag] = _rect(cx0, div, cx1, bot)

    qc_idx = config.QC_COLUMN_INDEX
    if qc_idx < len(columns):
        qx0, qx1 = columns[qc_idx]
        qdy = (bot - top) * config.QC_Y_INSET_FRAC
        rois[config.QC_TAG] = ((qx0 + (qx1 - qx0) * x_inset) / w, (top + qdy) / h,
                               (qx1 - (qx1 - qx0) * x_inset) / w, (bot - qdy) / h)
    return rois


def draw_grid(image: np.ndarray, grid: dict, color: Tuple[int, int, int] = (255, 0, 255)
              ) -> np.ndarray:

    vis = image.copy()
    h, w = vis.shape[:2]
    for x0, x1 in grid["columns"]:
        cv2.line(vis, (int(x0), 0), (int(x0), h), color, 1)
        cv2.line(vis, (int(x1), 0), (int(x1), h), color, 1)
    for y in grid["rows"]:
        cv2.line(vis, (0, int(y)), (w, int(y)), color, 1)
    return vis
