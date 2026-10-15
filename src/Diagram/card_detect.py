
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from . import config
from .calibration import detect_blue_quad, get_homography
from .geometry import compute_homography, order_quad, quad_area, warp_to_standard

_reference_cache: dict = {}


def detect_card_quad(image: np.ndarray) -> Tuple[Optional[np.ndarray], str]:

    if config.CARD_DETECT_BLUE_FIRST:
        quad = detect_blue_quad(image)
        if quad is not None:
            return quad, "blue"

    quad = _detect_quad_by_edges(image)
    if quad is not None:
        return quad, "canny-quad"
    return None, "none"


def _detect_quad_by_edges(image: np.ndarray) -> Optional[np.ndarray]:

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = config.CARD_GAUSSIAN_BLUR
    if k > 1:
        k = k if k % 2 == 1 else k + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    edges = cv2.Canny(gray, config.CARD_CANNY_LOW, config.CARD_CANNY_HIGH)
    kernel = np.ones((config.CARD_MORPH_KERNEL, config.CARD_MORPH_KERNEL), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=config.CARD_MORPH_ITERS)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = config.CARD_MIN_AREA_FRAC * h * w
    best: Optional[np.ndarray] = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            break
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, config.CARD_APPROX_EPS_FRAC * peri, True)
        if len(approx) != 4:
            continue
        quad = approx.reshape(-1, 2).astype(np.float32)
        if not cv2.isContourConvex(approx):
            continue
        if area > best_area:
            best, best_area = quad, area
    return order_quad(best) if best is not None else None


def _load_reference() -> Tuple[np.ndarray, np.ndarray]:

    if "image" not in _reference_cache:
        calib, matrix, _ = get_homography()
        image = cv2.imread(calib["reference_image"])
        if image is None:
            raise FileNotFoundError(f"参考图无法读取: {calib['reference_image']}")
        _reference_cache["image"] = image
        _reference_cache["matrix"] = np.asarray(matrix, dtype=np.float64)
    return _reference_cache["image"], _reference_cache["matrix"]


def register_by_reference(image: np.ndarray) -> Tuple[Optional[np.ndarray], int]:

    try:
        ref, ref_matrix = _load_reference()
    except FileNotFoundError:
        return None, 0

    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=int(config.ORB_FEATURES))
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    kp_img, des_img = orb.detectAndCompute(gray_img, None)
    if des_ref is None or des_img is None or len(kp_ref) < 10 or len(kp_img) < 10:
        return None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_img, des_ref, k=2)
    good = [m for m, n in knn if m.distance < config.ORB_RATIO * n.distance]
    if len(good) < 8:
        return None, 0

    src = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.ORB_RANSAC_THRESH)
    if homography is None or mask is None:
        return None, 0
    inliers = int(mask.sum())
    if inliers < config.ORB_MIN_INLIERS:
        return None, inliers
    return ref_matrix @ homography, inliers


def estimate_transform(image: np.ndarray, size: Tuple[int, int]
                       ) -> Tuple[np.ndarray, str]:

    if config.CARD_DETECT_BLUE_FIRST:
        quad = detect_blue_quad(image)
        if quad is not None:
            return compute_homography(quad, size), "blue"

    matrix, inliers = register_by_reference(image)
    if matrix is not None:
        return matrix, f"orb-reference(inliers={inliers})"

    quad = _detect_quad_by_edges(image)
    if quad is not None:
        return compute_homography(quad, size), "canny-quad"

    h, w = image.shape[:2]
    tw, th = int(size[0]), int(size[1])
    matrix = np.array([[tw / w, 0, 0], [0, th / h, 0], [0, 0, 1]], dtype=np.float64)
    return matrix, "resize-fallback"


def warp_card(image: np.ndarray, size: Tuple[int, int],
              quad: Optional[np.ndarray] = None,
              method: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, str]:

    if quad is not None:
        matrix = compute_homography(quad, size)
        return warp_to_standard(image, matrix, size), matrix, method or "provided"
    matrix, resolved = estimate_transform(image, size)
    return warp_to_standard(image, matrix, size), matrix, resolved


def describe_quad(quad: Optional[np.ndarray]) -> str:

    if quad is None:
        return "无"
    return f"面积 {quad_area(quad):.0f}px^2, 角点 {np.asarray(quad).astype(int).tolist()}"
