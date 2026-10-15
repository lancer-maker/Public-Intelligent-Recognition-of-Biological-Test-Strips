
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from . import config
from .geometry import compute_homography, order_quad, quad_area, warp_to_standard


def resolve_reference_path(force_refresh: bool = False) -> Path:

    local = Path(config.REFERENCE_IMAGE)
    sources = []
    if getattr(config, "REFERENCE_IMAGE_SOURCE", ""):
        sources.append(config.REFERENCE_IMAGE_SOURCE)
    sources.extend(config.REFERENCE_IMAGE_FALLBACKS)

    if force_refresh:
        for cand in sources:
            cand = Path(cand)
            if cand.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(cand), str(local))
                print(f"[calib] 已更新仓库内参考图: {local}  (来源: {cand})")
                return local
    if local.exists():
        return local
    for cand in sources:
        cand = Path(cand)
        if cand.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(cand), str(local))
            print(f"[calib] 已把参考图复制到仓库内: {local}  (来源: {cand})")
            return local
    raise FileNotFoundError(
        "未找到参考图 (用于标定卡盒几何与孔位)。请二选一:\n"
        f"  1) 把参考图放到: {local}\n"
        "  2) 在 config.REFERENCE_IMAGE_SOURCE / REFERENCE_IMAGE_FALLBACKS 中补充其路径。"
    )


def detect_blue_quad(image: np.ndarray) -> Optional[np.ndarray]:

    b, g, r = cv2.split(image.astype(np.int16))
    mask = ((b - r > config.BLUE_DELTA) & (b - g > config.BLUE_DELTA)
            & (b > config.BLUE_MIN)).astype(np.uint8) * 255
    if int((mask > 0).sum()) < 50:
        return None

    k = np.ones((config.BLUE_MORPH_KERNEL, config.BLUE_MORPH_KERNEL), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=config.BLUE_MORPH_ITERS)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.05 * image.shape[0] * image.shape[1]:
        return None

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, config.CARD_APPROX_EPS_FRAC * peri, True)
    quad = approx.reshape(-1, 2) if len(approx) == 4 else cv2.boxPoints(cv2.minAreaRect(contour))
    if not cv2.isContourConvex(np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2)):
        return None
    return order_quad(quad)


def _write_debug(image: np.ndarray, quad: np.ndarray, method: str, out_path: Path) -> None:

    vis = image.copy()
    cv2.polylines(vis, [quad.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    for idx, (x, y) in enumerate(quad):
        cv2.circle(vis, (int(x), int(y)), 8, (255, 0, 255), -1)
        cv2.putText(vis, str(idx), (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
    cv2.putText(vis, f"calib: {method}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def calibrate_reference(reference_path: Optional[str] = None,
                        force: bool = False) -> dict:

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if config.CALIB_JSON.exists() and not force:
        return load_calibration()

    ref_path = Path(reference_path) if reference_path else resolve_reference_path(force_refresh=force)
    image = cv2.imread(str(ref_path))
    if image is None:
        raise FileNotFoundError(
            f"参考图无法读取: {ref_path}\n请检查 config.REFERENCE_IMAGE 是否为有效图片路径。"
        )

    quad = detect_blue_quad(image)
    method = "blue-detect"
    if quad is None:
        quad = order_quad(config.REFERENCE_FALLBACK_QUAD)
        method = "fallback"
        print(f"[calib] 蓝框自动检测失败, 回退到固定坐标: {quad.tolist()}")

    size = (config.STANDARD_WIDTH, config.STANDARD_HEIGHT)
    matrix = compute_homography(quad, size)
    payload = {
        "reference_image": str(ref_path),
        "method": method,
        "source_quad": quad.tolist(),
        "standard_size": [int(size[0]), int(size[1])],
        "homography": matrix.tolist(),
        "inverse_homography": np.linalg.inv(matrix).tolist(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config.CALIB_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    _write_debug(image, quad, method, config.PREVIEW_DIR / "calib_reference_quad.png")
    warped = warp_to_standard(image, matrix, size)
    cv2.imwrite(str(config.PREVIEW_DIR / "calib_reference_standard.png"), warped)

    print(f"[calib] 标定完成 ({method}) -> {config.CALIB_JSON}")
    print(f"[calib] 源四角(TL,TR,BR,BL): {quad.tolist()}")
    print(f"[calib] 标准矩形: {size[0]}x{size[1]} | 面积占比参考: {quad_area(quad):.0f}px^2")
    return payload


def load_calibration() -> dict:

    if not config.CALIB_JSON.exists():
        return calibrate_reference()
    with open(config.CALIB_JSON, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    ref = calib.get("reference_image")
    if not ref or not Path(ref).exists():
        print(f"[calib] 标定文件记录的参考图不存在 ({ref}), 自动重新标定 ...")
        return calibrate_reference(force=True)
    return calib


def get_homography() -> tuple:

    calib = load_calibration()
    matrix = np.asarray(calib["homography"], dtype=np.float64)
    inv = np.asarray(calib["inverse_homography"], dtype=np.float64)
    return calib, matrix, inv
