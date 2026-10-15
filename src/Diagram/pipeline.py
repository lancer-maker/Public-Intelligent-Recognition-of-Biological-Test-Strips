
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from . import config
from .calibration import calibrate_reference, get_homography
from .card_detect import estimate_transform
from .geometry import crop_normalized, resize, warp_to_standard
from .gradcam import StripGradCAM, save_heatmap
from .predictor import EnsemblePredictor
from .qc import analyze_qc, save_qc_curve
from .roi import (draw_roi_boxes, draw_roi_preview, load_roi_config,
                  rois_from_warped, save_roi_config)
from .visualize import (build_summary, draw_qc_peaks, draw_results, draw_text_cn,
                        save_image)


def ensure_setup(force_calib: bool = False, force_roi: bool = False) -> dict:

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    calib = calibrate_reference(force=force_calib)
    save_roi_config(force=force_roi)
    roi_cfg = load_roi_config()
    return {"calib": calib, "roi": roi_cfg}


def run_preview(force: bool = False) -> Path:

    ensure_setup(force_calib=force, force_roi=force)
    return draw_roi_preview()


def run(image_path: str,
        predictor: Optional[EnsemblePredictor] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        use_gradcam: bool = True,
        verbose: bool = True) -> dict:

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    setup = ensure_setup()
    size = tuple(setup["roi"]["standard_size"])

    image_path = str(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取输入图像: {image_path}")

    stem = Path(image_path).stem
    out_dir = config.RESULTS_DIR / stem
    crops_dir = out_dir / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    matrix, method = estimate_transform(image, size)
    warped = warp_to_standard(image, matrix, size)
    inv_matrix = np.linalg.inv(matrix)
    log(f"[pipeline] 配准: {method}")

    roi_cfg, grid = rois_from_warped(warped)
    if grid is not None and grid.get("columns"):
        spans = [f"{x0:.0f}-{x1:.0f}" for x0, x1 in grid["columns"]]
        log(f"[pipeline] 孔位定位: {grid['method']} | 列范围 {spans}")
    else:
        log(f"[pipeline] 孔位来源: {roi_cfg.get('source', 'unknown')}")
    save_image(draw_roi_boxes(warped, roi_cfg, size), out_dir / "warped_standard.png")

    crop_paths: Dict[str, Path] = {}
    for tag, info in roi_cfg["regions"].items():
        crop = crop_normalized(warped, info["rect"], size)
        if info["type"] == "test":
            crop = resize(crop, config.MODEL_INPUT_SIZE)
        safe_name = str(info["name"]).replace(" ", "_")
        crop_path = crops_dir / f"{tag}_{safe_name}.png"
        save_image(crop, crop_path)
        crop_paths[tag] = crop_path

    qc_crop = cv2.imread(str(crop_paths[config.QC_TAG]))
    qc_result = analyze_qc(qc_crop)
    log(f"[pipeline] {qc_result.summary()}")
    save_image(draw_qc_peaks(qc_crop, qc_result), out_dir / "qc_annotated.png")
    save_qc_curve(qc_result, out_dir / "qc_curve.png")

    predictor = predictor or EnsemblePredictor(device=device, threshold_override=threshold)
    predictor.load()
    log(f"[pipeline] {predictor.describe()}")

    results: Dict[str, dict] = {}
    for tag in config.TEST_TAGS:
        pred = predictor.predict(crop_paths[tag])
        pred["name"] = config.TAG_TO_NAME.get(tag, "")
        results[tag] = pred
        log(f"  [{tag}] {pred['name']:<9} prob={pred['prob']:.4f} -> {pred['label']} "
            f"(成员: {', '.join(f'{p:.3f}' for p in pred['member_probs'])})")

    heatmap_paths = []
    if use_gradcam:
        log(f"[pipeline] Grad-CAM 使用最佳单模型: {predictor.best_weight_path}")
        cam_engine = StripGradCAM(predictor.best_model)
        for tag in config.TEST_TAGS:
            crop = cv2.imread(str(crop_paths[tag]))
            img_t, proj_t = predictor.build_input(crop_paths[tag])
            cam, cam_prob = cam_engine.generate(img_t, proj_t)
            out_path = config.HEATMAPS_DIR / stem / f"{tag}_{config.TAG_TO_NAME.get(tag, '')}_cam.png"
            title = f"{tag} {config.TAG_TO_NAME.get(tag, '')} p={results[tag]['prob']:.2f}"
            save_heatmap(crop, cam, out_path, title=title)
            heatmap_paths.append(str(out_path))
        log(f"[pipeline] 热力图已输出 {len(heatmap_paths)} 张 -> {config.HEATMAPS_DIR / stem}")

    annotated = draw_results(image, roi_cfg, results, qc_result,
                             inv_matrix=inv_matrix, size=size)
    annotated_std = draw_results(warped, roi_cfg, results, qc_result, inv_matrix=None, size=size)
    summary = build_summary(results, qc_result)

    annotated_path = save_image(annotated, out_dir / "diagram_result.png")
    save_image(annotated_std, out_dir / "diagram_result_standard.png")

    summary_txt = out_dir / "summary.txt"
    summary_txt.write_text(summary + "\n", encoding="utf-8")

    payload = {
        "image": image_path,
        "card_detect_method": method,
        "qc": {
            "valid": bool(qc_result.valid),
            "n_peaks": int(qc_result.n_peaks),
            "peaks": [int(p) for p in qc_result.peaks],
        },
        "threshold": {
            "primary": float(predictor.primary_threshold),
            "neg_max": float(predictor.neg_max),
            "pos_min": float(predictor.pos_min),
        },
        "results": {
            tag: {
                "name": results[tag]["name"],
                "prob": float(results[tag]["prob"]),
                "member_probs": [float(p) for p in results[tag]["member_probs"]],
                "label": results[tag]["label"],
            }
            for tag in config.TEST_TAGS
        },
        "summary": summary,
        "outputs": {
            "result_image": str(annotated_path),
            "result_standard": str(out_dir / "diagram_result_standard.png"),
            "summary_txt": str(summary_txt),
            "heatmaps": heatmap_paths,
        },
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log("\n" + "=" * 60)
    log(f"汇总: {summary}")
    log(f"结果图: {annotated_path}")
    log("=" * 60)
    return payload


def _tile(crop: np.ndarray, label: str) -> np.ndarray:

    h, w = crop.shape[:2]
    tile = np.full((h + 28, w, 3), 255, np.uint8)
    tile[28:28 + h, :w] = crop
    cv2.line(tile, (0, 27), (w, 27), (200, 200, 200), 1)
    return draw_text_cn(tile, label, (4, 4), (0, 0, 0), size=16)


def _contact_sheet(tiles, cols: int = 3, pad: int = 8) -> np.ndarray:

    rows = (len(tiles) + cols - 1) // cols
    th = max(t.shape[0] for t in tiles)
    tw = max(t.shape[1] for t in tiles)
    sheet = np.full((rows * (th + pad) + pad, cols * (tw + pad) + pad, 3), 245, np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        y, x = pad + r * (th + pad), pad + c * (tw + pad)
        sheet[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    return sheet


def preview_image(image_path: str, verbose: bool = True) -> dict:

    size = (config.STANDARD_WIDTH, config.STANDARD_HEIGHT)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取输入图像: {image_path}")

    stem = Path(image_path).stem
    out_dir = config.PREVIEW_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, method = estimate_transform(image, size)
    warped = warp_to_standard(image, matrix, size)
    inv = np.linalg.inv(matrix)
    roi_cfg, grid = rois_from_warped(warped)

    if verbose:
        print(f"[preview] 配准方法: {method}")
        print(f"[preview] 孔位来源: {roi_cfg.get('source', 'unknown')}")

    save_image(draw_roi_boxes(warped, roi_cfg, size),
               out_dir / f"{stem}_preview_standard.png")
    save_image(draw_results(image, roi_cfg, {}, None, inv_matrix=inv, size=size),
               out_dir / f"{stem}_preview_original.png")

    tiles = []
    for tag in config.TEST_TAGS:
        crop = resize(crop_normalized(warped, roi_cfg["regions"][tag]["rect"], size),
                      config.MODEL_INPUT_SIZE)
        tiles.append(_tile(crop, f"{tag} {config.TAG_TO_NAME.get(tag, '')}"))
    qc_info = roi_cfg["regions"][config.QC_TAG]
    tiles.append(_tile(resize(crop_normalized(warped, qc_info["rect"], size),
                              config.MODEL_INPUT_SIZE), f"QC {qc_info['name']}"))
    save_image(_contact_sheet(tiles, cols=3), out_dir / f"{stem}_crops.png")

    if verbose:
        print(f"[preview] 已输出到: {out_dir}")
    return {
        "standard": str(out_dir / f"{stem}_preview_standard.png"),
        "original": str(out_dir / f"{stem}_preview_original.png"),
        "crops": str(out_dir / f"{stem}_crops.png"),
        "method": method,
        "source": roi_cfg.get("source", "unknown"),
        "grid": grid,
    }
