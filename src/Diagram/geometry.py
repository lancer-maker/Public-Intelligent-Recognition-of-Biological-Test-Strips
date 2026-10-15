
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

Rect = Tuple[float, float, float, float]
Size = Tuple[int, int]


def order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"order_quad 需要 4 个点, 实际收到 {pts.shape[0]} 个")
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[int(np.argmin(s))]
    ordered[2] = pts[int(np.argmax(s))]
    ordered[1] = pts[int(np.argmax(d))]
    ordered[3] = pts[int(np.argmin(d))]
    return ordered


def compute_homography(src_quad: Sequence[Sequence[float]], size: Size) -> np.ndarray:

    w, h = int(size[0]), int(size[1])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(order_quad(src_quad), dst)


def warp_to_standard(image: np.ndarray, matrix: np.ndarray, size: Size) -> np.ndarray:

    w, h = int(size[0]), int(size[1])
    return cv2.warpPerspective(image, matrix, (w, h))


def invert(matrix: np.ndarray) -> np.ndarray:

    return np.linalg.inv(np.asarray(matrix, dtype=np.float64))


def norm_rect_to_pixels(rect: Rect, size: Size) -> Tuple[int, int, int, int]:

    x0, y0, x1, y1 = rect
    w, h = int(size[0]), int(size[1])
    px0 = int(round(x0 * w))
    py0 = int(round(y0 * h))
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    return px0, py0, px1, py1


def crop_rect(image: np.ndarray, rect_pixels: Tuple[int, int, int, int]) -> np.ndarray:

    x0, y0, x1, y1 = rect_pixels
    h, w = image.shape[:2]
    x0c, x1c = max(0, x0), min(w, x1)
    y0c, y1c = max(0, y0), min(h, y1)
    patch = image[y0c:y1c, x0c:x1c]
    if patch.size == 0:
        return np.zeros((max(1, y1 - y0), max(1, x1 - x0), image.shape[2] if image.ndim == 3 else 1),
                        dtype=image.dtype)
    if (x1c - x0c, y1c - y0c) != (x1 - x0, y1 - y0):
        patch = cv2.copyMakeBorder(
            patch, y0c - y0, y1 - y1c, x0c - x0, x1 - x1c,
            cv2.BORDER_CONSTANT, value=(0, 0, 0) if image.ndim == 3 else 0,
        )
    return patch.copy()


def crop_normalized(image: np.ndarray, rect: Rect, size: Size) -> np.ndarray:

    return crop_rect(image, norm_rect_to_pixels(rect, size))


def project_rect_back(rect: Rect, inv_matrix: np.ndarray, size: Size) -> np.ndarray:

    px = norm_rect_to_pixels(rect, size)
    pts = np.float32([[[px[0], px[1]]], [[px[2], px[1]]], [[px[2], px[3]]], [[px[0], px[3]]]])
    return cv2.perspectiveTransform(pts, inv_matrix).reshape(-1, 2)


def resize(image: np.ndarray, size: Size) -> np.ndarray:

    w, h = int(size[0]), int(size[1])
    src_h, src_w = image.shape[:2]
    interp = cv2.INTER_AREA if (src_w > w or src_h > h) else cv2.INTER_CUBIC
    return cv2.resize(image, (w, h), interpolation=interp)


def quad_area(points: np.ndarray) -> float:

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
