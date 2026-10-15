
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config
from .geometry import project_rect_back

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
_font_cache: Dict[int, "ImageFont.FreeTypeFont"] = {}

LABEL_COLORS = {
    config.LABEL_POSITIVE: (0, 0, 255),
    config.LABEL_NEGATIVE: (0, 160, 0),
    config.LABEL_SUSPICIOUS: (0, 165, 255),
}


def _load_font(size: int):

    if not _HAS_PIL:
        return None
    if size in _font_cache:
        return _font_cache[size]
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    return None


def draw_text_cn(image: np.ndarray, text: str, org: Tuple[int, int],
                 color: Tuple[int, int, int], size: int = 18,
                 background: Optional[Tuple[int, int, int]] = None) -> np.ndarray:

    font = _load_font(size)
    if font is None:
        cv2.putText(image, text, (org[0], org[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30.0, color, 1)
        return image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    if background is not None:
        box = draw.textbbox(org, text, font=font)
        pad = 3
        draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
                       fill=(background[2], background[1], background[0]))
    draw.text(org, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def draw_results(base_image: np.ndarray,
                 roi_cfg: dict,
                 results: Dict[str, dict],
                 qc_result,
                 inv_matrix: Optional[np.ndarray] = None,
                 size: Tuple[int, int] = None) -> np.ndarray:

    from .geometry import norm_rect_to_pixels

    vis = base_image.copy()
    size = tuple(size or roi_cfg["standard_size"])

    for tag, info in roi_cfg["regions"].items():
        is_qc = info["type"] == "qc"
        if inv_matrix is not None:
            pts = project_rect_back(info["rect"], inv_matrix, size).astype(np.int32)
            p0, p1 = pts[0], pts[2]
        else:
            px = norm_rect_to_pixels(info["rect"], size)
            p0, p1 = np.array([px[0], px[1]]), np.array([px[2], px[3]])

        if is_qc:
            color = (0, 165, 255)
            label = f"QC {info['name']}"
            if qc_result is not None:
                label += f" {'有效' if qc_result.valid else '无效'}({qc_result.n_peaks})"
        else:
            res = results.get(tag, {})
            color = LABEL_COLORS.get(res.get("label", ""), (0, 0, 255))
            prob = res.get("prob")
            label = f"{tag} {info['name']}"
            if prob is not None:
                label += f" {res.get('label', '')} {prob:.2f}"

        if inv_matrix is not None:
            cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, color, 2)
            anchor = (int(pts[0][0]), int(pts[0][1]))
        else:
            cv2.rectangle(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), color, 2)
            anchor = (int(p0[0]), int(p0[1]))

        vis = draw_text_cn(vis, label, (anchor[0] + 4, max(2, anchor[1] + 4)),
                           color, size=18, background=(255, 255, 255))
    return vis


def build_summary(results: Dict[str, dict], qc_result) -> str:

    if qc_result is not None and not qc_result.valid:
        return "质控无效，本次试验结果不可用"

    parts = []
    for tag in config.TEST_TAGS:
        res = results.get(tag)
        name = config.TAG_TO_NAME.get(tag, "")
        if res is None:
            parts.append(f"{tag}({name}):未检测")
        else:
            parts.append(f"{tag}({name}):{res.get('label', '?')}")
    return " ".join(parts)


def draw_qc_peaks(crop_bgr: np.ndarray, qc_result) -> np.ndarray:

    vis = crop_bgr.copy()
    if qc_result is None or not qc_result.peaks:
        return vis
    h, w = vis.shape[:2]
    if qc_result.curve.size != h:
        scale = h / max(1, qc_result.curve.size)
        ys = [int(min(h - 1, max(0, p * scale))) for p in qc_result.peaks]
    else:
        ys = list(qc_result.peaks)
    for idx, y in enumerate(ys):
        cv2.line(vis, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(vis, str(idx + 1), (2, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return vis


def save_image(image: np.ndarray, out_path: Path) -> Path:

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
    return out_path
