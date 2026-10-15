
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np

from . import config

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class QCResult:


    valid: bool
    n_peaks: int
    peaks: List[int] = field(default_factory=list)
    curve: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    message: str = ""

    def summary(self) -> str:

        state = "有效" if self.valid else "无效"
        return f"质控：{state} (波峰数={self.n_peaks}, 需>={config.QC_MIN_PEAKS})"


def _row_mean_curve(crop_bgr: np.ndarray) -> np.ndarray:

    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr
    return gray.astype(np.float32).mean(axis=1)


def _normalize(curve: np.ndarray) -> np.ndarray:

    lo, hi = float(np.min(curve)), float(np.max(curve))
    if hi - lo < 1e-8:
        return np.zeros_like(curve)
    return (curve - lo) / (hi - lo)


def analyze_qc(crop_bgr: np.ndarray) -> QCResult:

    if crop_bgr is None or crop_bgr.size == 0:
        return QCResult(False, 0, [], np.zeros(0, dtype=np.float32), "质控区域为空")

    curve = _row_mean_curve(crop_bgr)
    if config.QC_INVERT:
        curve = float(np.max(curve)) - curve
    if config.QC_SMOOTH_SIGMA and config.QC_SMOOTH_SIGMA > 0 and _HAS_SCIPY:
        curve = gaussian_filter1d(curve, sigma=float(config.QC_SMOOTH_SIGMA))
    curve = _normalize(curve)

    if not _HAS_SCIPY:
        thr = max(float(config.QC_PEAK_HEIGHT), 0.5)
        above = curve > thr
        peaks = [i for i in range(1, len(curve) - 1)
                 if above[i] and curve[i] >= curve[i - 1] and curve[i] > curve[i + 1]]
        peaks = _merge_close(peaks, max(1, int(len(curve) * config.QC_PEAK_DISTANCE_FRAC)))
    else:
        distance = max(1, int(len(curve) * config.QC_PEAK_DISTANCE_FRAC))
        peaks, _ = find_peaks(
            curve,
            prominence=float(config.QC_PEAK_PROMINENCE),
            height=float(config.QC_PEAK_HEIGHT),
            distance=distance,
        )
        peaks = [int(p) for p in peaks]

    valid = len(peaks) >= int(config.QC_MIN_PEAKS)
    msg = (f"质控{'有效' if valid else '无效'}: 检出 {len(peaks)} 个波峰 "
           f"(需 >= {config.QC_MIN_PEAKS})")
    return QCResult(valid, len(peaks), peaks, curve, msg)


def _merge_close(peaks: List[int], min_distance: int) -> List[int]:

    if not peaks:
        return []
    merged = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] >= min_distance:
            merged.append(p)
    return merged


def save_qc_curve(result: QCResult, out_path: Path, title: str = "QC curve") -> None:

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        print(f"[qc] 跳过曲线绘图 (matplotlib 不可用: {exc})")
        return

    if result.curve.size == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=150)
    ax.plot(np.arange(result.curve.size), result.curve, lw=1.2, color="#1f77b4")
    if result.peaks:
        ax.scatter(result.peaks, result.curve[result.peaks], color="#d62728", zorder=3, s=18)
    ax.axhline(float(config.QC_PEAK_HEIGHT), color="gray", ls=":", lw=1.0)
    ax.set_title(f"{title}\n{result.summary()}", fontsize=8)
    ax.set_xlabel("row index", fontsize=8)
    ax.set_ylabel("normalized", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
