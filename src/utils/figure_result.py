
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.metrics import (
        confusion_matrix as sk_confusion_matrix,
        precision_recall_curve,
    )
except Exception:
    precision_recall_curve = None
    sk_confusion_matrix = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PLOT_FIGSIZE = (18, 6)
PLOT_DPI = 300
BLUES = ["#1f77b4", "#aec7e8", "#3182bd", "#6baed6", "#7aa6d6",
         "#4f81bd", "#9ecae1", "#c6dbef", "#2f6f9f", "#d0e1f9"]


def apply_global_style(font_size: int = 14, label_size: int = 16) -> None:

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.labelsize"] = label_size
    plt.rcParams["xtick.labelsize"] = font_size
    plt.rcParams["ytick.labelsize"] = font_size
    plt.rcParams["legend.fontsize"] = font_size
    plt.rcParams["axes.titlesize"] = label_size + 1
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = PLOT_DPI
    plt.rcParams["axes.grid"] = False


def _moving_average(values: np.ndarray, window: int = 15) -> np.ndarray:

    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window // 2 - (window % 2 == 0)), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    if len(smoothed) > len(values):
        start = (len(smoothed) - len(values)) // 2
        smoothed = smoothed[start:start + len(values)]
    return smoothed[:len(values)]


def plot_three_panel(
    x_axis: Sequence[float],
    results: Sequence[float],
    smooth: Sequence[float],
    conf_matrix: np.ndarray,
    classes: List[str],
    pr_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    val_loss: Optional[Sequence[float]] = None,
    save_path: Optional[Path] = None,
    titles: Optional[List[str]] = None,
) -> plt.Figure:

    apply_global_style()

    titles = titles or ["Loss Curve", "Confusion Matrix", "PR Curves"]
    fig, axes = plt.subplots(1, 3, figsize=PLOT_FIGSIZE)

    x_axis = np.asarray(x_axis, dtype=float)
    results = np.asarray(results, dtype=float)
    smooth = np.asarray(smooth, dtype=float)

    ax = axes[0]
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.plot(x_axis, results, color="#1f77b4", marker="o", markersize=3,
            linestyle="-", linewidth=1.5, label="Train Loss (raw)", zorder=3)
    ax.plot(x_axis, smooth, color="#ffa64d", linestyle="--", linewidth=2,
            label="Train Loss (smooth)", zorder=2)
    if val_loss is not None:
        val_arr = np.asarray(val_loss, dtype=float)
        if len(val_arr) == len(x_axis):
            ax.plot(x_axis, val_arr, color="#2ca02c", linestyle=":",
                    linewidth=2, label="Val Loss", zorder=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="best")
    ax.set_title(titles[0])

    ax = axes[1]
    conf = np.asarray(conf_matrix, dtype=float)
    sns.heatmap(conf, annot=True, fmt=".2f", cmap="Blues", cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(titles[1])

    ax = axes[2]
    grid, mean_prec, ap_cls, mAP = _pr_grid_summary(classes, pr_data, n=101)
    for i, cls in enumerate(classes):
        if cls not in pr_data:
            continue
        rec, pre = pr_data[cls]
        color = BLUES[i % len(BLUES)]
        label = f"{cls} {ap_cls[cls]:.3f}" if cls in ap_cls else cls
        ax.plot(rec, pre, color=color, linewidth=1.5, label=label)
    if len(ap_cls) > 0:
        ax.plot(grid, mean_prec, color="#6a0dad", linewidth=2.2,
                linestyle="-", label=f"all classes {mAP:.3f} mAP@0.5", zorder=4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="best")
    ax.set_title(titles[2])

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight")
    return fig


def _interpolate_precision(recall: np.ndarray, precision: np.ndarray,
                           grid: np.ndarray) -> np.ndarray:

    recall = np.asarray(recall, dtype=float)
    precision = np.asarray(precision, dtype=float)
    grid = np.asarray(grid, dtype=float)
    out = np.empty_like(grid)
    for j, g in enumerate(grid):
        mask = recall >= g
        out[j] = float(precision[mask].max()) if mask.any() else 1.0
    return out


def _trapz(y: np.ndarray, x: np.ndarray) -> float:

    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(y, x))


def _pr_grid_summary(classes: List[str],
                     pr_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                     n: int = 101) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], float]:

    grid = np.linspace(0.0, 1.0, n)
    interp: Dict[str, np.ndarray] = {}
    for cls in classes:
        if cls not in pr_data:
            continue
        rec, pre = pr_data[cls]
        interp[cls] = _interpolate_precision(rec, pre, grid)
    if not interp:
        return grid, np.zeros(n), {}, 0.0
    ap_cls = {cls: _trapz(interp[cls], grid) for cls in interp}
    mean_prec = np.mean(np.stack(list(interp.values())), axis=0)
    mAP = _trapz(mean_prec, grid)
    return grid, mean_prec, ap_cls, mAP


def make_mock_inputs() -> dict:

    rng = np.random.default_rng(42)

    x_axis = np.linspace(0.0, 400.0, 401)
    base = 0.75 * np.exp(-x_axis / 130.0) + 0.06
    results = base + rng.normal(0.0, 0.02, size=len(x_axis))
    smooth = _moving_average(results, window=15)
    val_loss = base + 0.10 + rng.normal(0.0, 0.02, size=len(x_axis))

    classes = ["N", "P", "Qc", "Qw", "Background"]
    counts = np.array([
        [95, 2, 1, 0, 2],
        [1, 88, 2, 1, 0],
        [1, 2, 90, 3, 1],
        [1, 1, 3, 92, 2],
        [2, 1, 1, 2, 93],
    ], dtype=float)
    conf_matrix = counts / counts.sum(axis=1, keepdims=True)

    pr_data, map_values = {}, {}
    n_samp = 300
    y_global = np.zeros((n_samp, len(classes)), dtype=int)
    for c in range(len(classes)):
        y_global[rng.choice(n_samp, size=120, replace=False), c] = 1
    for c, cls in enumerate(classes):
        y = y_global[:, c]
        margin = float(rng.uniform(0.55, 0.95))
        score = y.astype(float) * margin + rng.normal(0.0, 0.22, size=n_samp)
        score = np.clip(score, 0.0, 1.0)
        prec, rec, _ = precision_recall_curve(y, score)
        pr_data[cls] = (rec, prec)
    _, _, map_values, all_map = _pr_grid_summary(classes, pr_data, n=101)

    return {
        "x_axis": x_axis, "results": results, "smooth": smooth, "val_loss": val_loss,
        "conf_matrix": conf_matrix, "classes": classes,
        "pr_data": pr_data, "map_values": map_values, "all_map": all_map,
    }


def _moving_average_windowed(values: np.ndarray, window: int = 9) -> np.ndarray:

    return _moving_average(values, window=window)


def load_real_inputs(model_id: int = 5,
                     history_csv: Optional[Path] = None,
                     pred_csv: Optional[Path] = None,
                     out_dir: Optional[Path] = None) -> dict:

    if history_csv is None:
        history_csv = PROJECT_ROOT / "outputs" / "checkpoints" / "Tweak" / str(model_id) / "history.csv"
    if pred_csv is None:
        pred_csv = (PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "predictions"
                    / f"{model_id}_best_model_preds.csv")
    if out_dir is None:
        out_dir = PROJECT_ROOT / "outputs" / "reports" / "three_panel" / f"model{model_id}"
    history_csv, pred_csv, out_dir = Path(history_csv), Path(pred_csv), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not history_csv.exists():
        raise FileNotFoundError(f"找不到训练 history: {history_csv}")
    if not pred_csv.exists():
        raise FileNotFoundError(f"找不到测试预测: {pred_csv}")

    hist = pd.read_csv(history_csv)
    train_loss = hist["train_loss"].astype(float).to_numpy()
    x_axis = np.arange(len(train_loss), dtype=float)
    smooth = _moving_average_windowed(train_loss, window=9)
    if "val_loss" in hist.columns:
        val_loss = (pd.to_numeric(hist["val_loss"], errors="coerce")
                    .interpolate(limit_direction="both").to_numpy())
    else:
        val_loss = np.full(len(train_loss), np.nan)
    loss_df = pd.DataFrame({"step": x_axis.astype(int),
                            "stage": hist["stage"].to_numpy(),
                            "epoch": hist["epoch"].to_numpy(),
                            "results": train_loss, "smooth": smooth,
                            "val_loss": val_loss})
    loss_df.to_csv(out_dir / "loss_curve.csv", index=False, encoding="utf-8-sig")

    pred = pd.read_csv(pred_csv)
    y_true = pred["test_labels"].astype(int).to_numpy()
    y_prob = pred["test_probs"].astype(float).to_numpy()
    y_pred = pred["pred_label"].astype(int).to_numpy()

    classes = ["N", "P"]
    cm_raw = sk_confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    conf_matrix = cm_raw / row_sums if row_sums.sum() > 0 else cm_raw

    cm_out = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    cm_out.index.name = "True"
    cm_out.columns.name = "Predicted"
    cm_out.to_csv(out_dir / "confusion_matrix.csv", encoding="utf-8-sig")

    pr_data, map_values = {}, {}
    pr_rows = []
    for i, cls in enumerate(classes):
        if cls == "P":
            y_bin = (y_true == 1).astype(int)
            score = y_prob
        else:
            y_bin = (y_true == 0).astype(int)
            score = 1.0 - y_prob
        prec, rec, _ = precision_recall_curve(y_bin, score)
        pr_data[cls] = (rec, prec)
        for r, p in zip(rec, prec):
            pr_rows.append({"class": cls, "recall": float(r), "precision": float(p)})
    _, _, map_values, all_map = _pr_grid_summary(classes, pr_data, n=101)

    pd.DataFrame(pr_rows).to_csv(out_dir / "pr_curves.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"class": cls, "AP": map_values[cls]} for cls in classes]
                 + [{"class": "all", "AP": all_map}]
                 ).to_csv(out_dir / "pr_metrics.csv", index=False, encoding="utf-8-sig")

    return {
        "x_axis": x_axis, "results": train_loss, "smooth": smooth, "val_loss": val_loss,
        "conf_matrix": conf_matrix, "classes": classes,
        "pr_data": pr_data, "map_values": map_values, "all_map": all_map,
        "out_dir": out_dir,
    }


def export_real_figure(model_id: int = 5, out_png: Optional[str] = None,
                       history_csv: Optional[Path] = None,
                       pred_csv: Optional[Path] = None,
                       out_dir: Optional[Path] = None) -> str:

    data = load_real_inputs(model_id=int(model_id), history_csv=history_csv,
                            pred_csv=pred_csv, out_dir=out_dir)
    png = Path(out_png) if out_png else data["out_dir"] / f"three_panel_model{int(model_id)}.png"
    fig = plot_three_panel(
        x_axis=data["x_axis"], results=data["results"], smooth=data["smooth"],
        val_loss=data.get("val_loss"), conf_matrix=data["conf_matrix"],
        classes=data["classes"], pr_data=data["pr_data"], save_path=png,
    )
    plt.close(fig)
    return str(png)


def main() -> None:
    parser = argparse.ArgumentParser(description="论文三联图 (Loss | Confusion | PR)")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock",
                        help="mock=模拟数据(默认); real=读取本项目真实 CSV 并新增输出")
    parser.add_argument("--model-id", type=int, default=5, help="real 模式使用的模型编号 (默认 5)")
    parser.add_argument("--out", type=str, default="",
                        help="输出 PNG 路径 (默认 outputs/reports/three_panel/)")
    args = parser.parse_args()

    if args.mode == "mock":
        data = make_mock_inputs()
        out_png = args.out or str(PROJECT_ROOT / "outputs" / "reports" / "three_panel" / "three_panel_mock.png")
        print("[*] 使用模拟数据生成三联图")
        fig = plot_three_panel(
            x_axis=data["x_axis"], results=data["results"], smooth=data["smooth"],
            val_loss=data.get("val_loss"), conf_matrix=data["conf_matrix"],
            classes=data["classes"], pr_data=data["pr_data"], save_path=out_png,
        )
        plt.close(fig)
    else:
        out_png = export_real_figure(model_id=args.model_id, out_png=args.out or None)
        print(f"[*] 使用真实数据(模型 {args.model_id}), 新增输出目录: "
              f"{PROJECT_ROOT / 'outputs' / 'reports' / 'three_panel' / f'model{args.model_id}'}")
    print(f"[✓] 已保存: {out_png}")


if __name__ == "__main__":
    main()
