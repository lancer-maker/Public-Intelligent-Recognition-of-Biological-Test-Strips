

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from yolo11_common import FIGURES_DIR, REPORTS_DIR
from test_yolo11_cls import (
    plot_confusion,
    plot_pr,
    plot_roc,
    plot_threshold_scan,
)

ENSEMBLE_SOURCE = "ensemble"


def _read_reports_csv(name: str) -> pd.DataFrame:

    path = REPORTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"缺少报告文件: {path}")
    df = pd.read_csv(path)
    missing = {"source", "filename", "label", "prob_P", "threshold"} - set(df.columns)
    if missing:
        raise ValueError(f"{name} 缺少列: {sorted(missing)}")
    return df


def _group_by_source(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for src, g in df.groupby("source", sort=False):
        out[str(src)] = (g["label"].astype(int).to_numpy(),
                         g["prob_P"].astype(float).to_numpy())
    return out


def _check_same_labels(groups: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:

    y_ref: Optional[np.ndarray] = None
    ref_name = None
    for src, (y, _) in groups.items():
        if y_ref is None:
            y_ref, ref_name = y, src
        elif not np.array_equal(y_ref, y):
            raise ValueError(f"source={src} 的标签序列与 source={ref_name} 不一致")
    if y_ref is None:
        raise ValueError("预测 CSV 中没有任何 source")
    return y_ref


def _load_split(per_seed_name: str, ensemble_name: str):

    df = _read_reports_csv(per_seed_name)
    groups = _group_by_source(df)
    y_true = _check_same_labels(groups)

    ens_probs = groups.get(ENSEMBLE_SOURCE, (None, None))[1]
    if ens_probs is None:
        ens_df = _read_reports_csv(ensemble_name)
        ens_probs = ens_df["prob_P"].astype(float).to_numpy()
        if not np.array_equal(ens_df["label"].astype(int).to_numpy(), y_true):
            raise ValueError(f"{ensemble_name} 的标签序列与 {per_seed_name} 不一致")
        ens_thr = float(ens_df["threshold"].iloc[0])
    else:
        ens_thr = float(df.loc[df["source"] == ENSEMBLE_SOURCE, "threshold"].iloc[0])

    seed_curves: List[Tuple[str, np.ndarray]] = [
        (src, prob) for src, (_, prob) in groups.items() if src != ENSEMBLE_SOURCE
    ]
    return y_true, seed_curves, np.asarray(ens_probs, dtype=float), ens_thr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 outputs/YOLO11/reports 的 CSV 重建 figures/<tag>/ 下的 SVG 图"
    )
    parser.add_argument("--tag", default="yolo11n-cls_224",
                        help="训练分组名, 决定 figures/<tag>/ 输出目录 (默认 yolo11n-cls_224)")
    args = parser.parse_args()

    tag = args.tag
    figures_dir = FIGURES_DIR / tag
    figures_dir.mkdir(parents=True, exist_ok=True)

    y_test, test_seed_curves, ens_test, ens_thr = _load_split(
        "per_seed_predictions_test.csv", "ensemble_predictions_test.csv")
    y_val, val_seed_curves, ens_val, thr_val = _load_split(
        "per_seed_predictions_val.csv", "ensemble_predictions_val.csv")

    if ens_thr != thr_val:
        print(f"[warn] test/val 的阈值列不一致 ({ens_thr} vs {thr_val}), 以 test 文件为准")

    print(f"[*] tag={tag} | val={len(y_val)} (阳性 {int((y_val == 1).sum())}) | "
          f"test={len(y_test)} (阳性 {int((y_test == 1).sum())}) | "
          f"种子数={len(test_seed_curves)} | 集成阈值={ens_thr:.4f}")

    plot_roc(y_test, test_seed_curves + [(ENSEMBLE_SOURCE, ens_test)],
             figures_dir / "roc_test.svg", f"YOLO11-cls ROC (test, {tag})")
    plot_roc(y_val, val_seed_curves + [(ENSEMBLE_SOURCE, ens_val)],
             figures_dir / "roc_val.svg", f"YOLO11-cls ROC (val, {tag})")
    plot_pr(y_test, [(ENSEMBLE_SOURCE, ens_test)], figures_dir / "pr_test.svg",
            f"YOLO11-cls PR (test, {tag})")
    plot_confusion(y_test, ens_test, ens_thr, figures_dir / "confusion_matrix_test.svg",
                   f"YOLO11-cls ensemble (test, {tag})")
    plot_threshold_scan(y_test, ens_test, ens_thr, figures_dir / "threshold_scan_test.svg",
                        f"YOLO11-cls ensemble (test, {tag})")

    for name in ("roc_test", "roc_val", "pr_test", "confusion_matrix_test", "threshold_scan_test"):
        print(f"    - {figures_dir / (name + '.svg')}")


if __name__ == "__main__":
    main()
