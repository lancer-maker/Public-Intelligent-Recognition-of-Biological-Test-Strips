
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import config
from .border_detect import (detect_grid, detect_red_boxes, grid_to_rois,
                            red_boxes_to_rois)
from .calibration import get_homography
from .geometry import norm_rect_to_pixels, project_rect_back, warp_to_standard

Rect = Tuple[float, float, float, float]


def fixed_rois() -> Dict[str, Rect]:

    return {tag: (float(x0), float(y0), float(x1), float(y1))
            for tag, _name, x0, y0, x1, y1 in config.ROI_DEFINITIONS}


def detect_reference_rois() -> Tuple[Optional[Dict[str, Rect]], Optional[dict]]:

    calib, matrix, _ = get_homography()
    reference = cv2.imread(calib["reference_image"])
    if reference is None:
        return None, None
    size = (config.STANDARD_WIDTH, config.STANDARD_HEIGHT)
    warped = warp_to_standard(reference, matrix, size)

    detected = detect_red_boxes(warped)
    if detected is not None:
        rois = red_boxes_to_rois(detected)
        if all(tag in rois for tag in config.TEST_TAGS):
            return rois, {
                "method": "red-boxes",
                "size": detected["size"],
                "n_test": len(detected["test"]),
                "has_qc": detected.get("qc") is not None,
            }

    grid = detect_grid(warped)
    if grid is None:
        return None, None
    return grid_to_rois(grid), grid


def build_roi_config(rois: Optional[Dict[str, Rect]] = None, source: str = None) -> dict:

    name_map = {tag: name for tag, name, *_ in config.ROI_DEFINITIONS}
    if rois is None:
        rois = fixed_rois()
        source = source or "fixed"

    regions: Dict[str, dict] = {}
    for tag in [*config.TEST_TAGS, config.QC_TAG]:
        if tag not in rois:
            continue
        regions[tag] = {
            "name": name_map.get(tag, tag),
            "rect": [float(v) for v in rois[tag]],
            "type": "qc" if tag == config.QC_TAG else "test",
        }
    return {
        "standard_size": [int(config.STANDARD_WIDTH), int(config.STANDARD_HEIGHT)],
        "source": source or "auto-detect",
        "order": list(config.TEST_TAGS),
        "regions": regions,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_roi_config(force: bool = False) -> Path:

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if config.ROI_JSON.exists() and not force:
        return config.ROI_JSON

    rois, info = (None, None)
    if config.ROI_REDETECT:
        rois, info = detect_reference_rois()
    if rois is None:
        payload = build_roi_config(source="fixed")
        print("[roi] [warn] 孔位自动定位失败, 回退到 config.ROI_DEFINITIONS 固定坐标")
    else:
        payload = build_roi_config(rois, source=f"auto-detect({info['method']})")
        print(f"[roi] 孔位自动定位成功 (方法={info['method']}): 9 个区域")

    with open(config.ROI_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[roi] 已保存 9 个区域配置 -> {config.ROI_JSON}")
    return config.ROI_JSON


def load_roi_config(force_rebuild: bool = False) -> dict:

    if force_rebuild or not config.ROI_JSON.exists():
        save_roi_config(force=True)
    with open(config.ROI_JSON, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rois_from_warped(warped: np.ndarray) -> Tuple[dict, Optional[dict]]:

    if config.ROI_REDETECT_PER_IMAGE:
        grid = detect_grid(warped)
        if grid is not None:
            rois = grid_to_rois(grid)
            return build_roi_config(rois, source=f"auto-detect({grid['method']})"), grid
    return load_roi_config(), None


def draw_roi_boxes(image: np.ndarray, roi_cfg: dict, size: Tuple[int, int],
                   thickness: int = 2) -> np.ndarray:

    vis = image.copy()
    for tag, info in roi_cfg["regions"].items():
        x0, y0, x1, y1 = norm_rect_to_pixels(info["rect"], size)
        color = (0, 165, 255) if info["type"] == "qc" else (0, 0, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
        cv2.putText(vis, f"{tag} {info['name']}", (x0 + 4, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def draw_roi_preview(save: bool = True) -> Path:

    calib, matrix, inv = get_homography()
    roi_cfg = load_roi_config()
    size = tuple(roi_cfg["standard_size"])

    reference = cv2.imread(calib["reference_image"])
    if reference is None:
        raise FileNotFoundError(f"参考图无法读取: {calib['reference_image']}")

    warped = warp_to_standard(reference, matrix, size)
    preview_standard = draw_roi_boxes(warped, roi_cfg, size)

    preview_original = reference.copy()
    for tag, info in roi_cfg["regions"].items():
        back = project_rect_back(info["rect"], inv, size).astype(np.int32).reshape(-1, 1, 2)
        color = (0, 165, 255) if info["type"] == "qc" else (0, 0, 255)
        cv2.polylines(preview_original, [back], True, color, 2)

    if save:
        config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.PREVIEW_DIR / "roi_preview_standard.png"), preview_standard)
        cv2.imwrite(str(config.PREVIEW_DIR / "roi_preview_original.png"), preview_original)
        print(f"[roi] 预览已保存: {config.PREVIEW_DIR / 'roi_preview_standard.png'}")
        print(f"[roi] 预览已保存: {config.PREVIEW_DIR / 'roi_preview_original.png'}")
    return config.PREVIEW_DIR / "roi_preview_standard.png"
