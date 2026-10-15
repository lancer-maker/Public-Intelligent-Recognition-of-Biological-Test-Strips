
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.metrics import compute_metrics
from src.models.dual_stream_net import DualStreamStripNet
from src.models.gated_fusion import GatedDualStreamNet
from src.models.image_stream_only import ImageOnlyBase
from src.utils.Fine_Tuning_helper import build_test_loader, load_config
from src.utils.figure_result import PLOT_DPI, apply_global_style
from src.utils.model_helper import load_model_state_dict
from new_env_test import load_validation_threshold, natural_sort_key

DEFAULT_CHECKPOINTS = [
    r"outputs\checkpoints\E",
    r"outputs\checkpoints\C",
    r"outputs\checkpoints\D"
]
TEST_CSV = r"data\test\labels\labels.csv"
TEST_RAW_DIR = ""
BATCH_SIZE = 8
PROJ_AUG = False
REPORT_DIR_NAME = "ensemble_report"
REPORT_DIR_OVERRIDES = {
    "Tweak": r"outputs\reports\new_env_test",
}
PLOT_INDIVIDUAL_CURVES = False
MAX_MODELS = 0
CLASSES = ["Negative", "Positive"]

C_ENSEMBLE = "#1f77b4"
C_INDIV = "#9ecae1"
C_SENS = "#d62728"
C_SPEC = "#1f77b4"
C_MARK = "#2ca02c"


def discover_models(mode_dir: Path, max_models: int = 0) -> List[Path]:

    paths = sorted(mode_dir.glob("*/best_model.pth"),
                   key=lambda p: natural_sort_key(p.parent.name))
    if max_models and max_models > 0:
        paths = paths[:max_models]
    return paths


def build_inference_model(weight_path: Path, model_cfg: dict):

    feature_dim = int(model_cfg.get("feature_dim", 128))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))
    state = load_model_state_dict(str(weight_path))
    keys = list(state.keys())

    if any(str(k).startswith("gate_mlp") for k in keys):
        w = state.get("gate_mlp.net.2.weight")
        gate_hidden = int(w.shape[1]) if isinstance(w, torch.Tensor) and w.ndim == 2 else 64
        model = GatedDualStreamNet(pretrained_2d=False, feature_dim=feature_dim,
                                   dropout_rate=dropout_rate, gate_hidden=gate_hidden)
        is_dual = True
    elif any(str(k).startswith("proj_stream") for k in keys):
        model = DualStreamStripNet(pretrained_2d=False, feature_dim=feature_dim,
                                   dropout_rate=dropout_rate)
        is_dual = True
    else:
        model = ImageOnlyBase(pretrained_2d=False, feature_dim=feature_dim,
                              dropout_rate=dropout_rate)
        is_dual = False

    model_keys = set(model.state_dict().keys())
    try:
        model.load_state_dict(state)
    except RuntimeError:
        missing = sorted(model_keys - set(keys))
        unexpected = sorted(set(keys) - model_keys)
        print(f"    [跳过] 权重与当前模型结构不兼容: {weight_path}")
        if missing:
            print(f"           模型需要但权重缺失 {len(missing)} 个键: {missing[:4]}"
                  f"{' ...' if len(missing) > 4 else ''}")
        if unexpected:
            print(f"           权重多余 {len(unexpected)} 个键: {unexpected[:4]}"
                  f"{' ...' if len(unexpected) > 4 else ''}")
        print("           可能原因: 该权重由旧版模型训练而得 (缺少 image_stream.strip_head "
              "等新增模块); 请用对应版本代码评估, 或重新训练。")
        return None
    return model, is_dual


def predict_probs(weight_path: Path, loader, device: torch.device,
                  model_cfg: dict) -> Optional[np.ndarray]:

    built = build_inference_model(weight_path, model_cfg)
    if built is None:
        return None
    model, is_dual = built
    model.to(device)
    model.eval()
    probs_list: List[float] = []
    with torch.no_grad():
        for imgs, projs, _ in loader:
            imgs = imgs.to(device)
            if is_dual:
                logits = model(imgs, projs.to(device))
            else:
                logits = model(imgs)
            probs_list.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
    return np.asarray(probs_list, dtype=np.float32)


def youden_threshold(y: np.ndarray, probs: np.ndarray) -> float:

    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, ths = roc_curve(y, probs)
    j = tpr - fpr
    best = float(ths[int(np.argmax(j))])
    if not np.isfinite(best) or best > 1.0:
        best = float(np.max(probs))
    return best


def ensemble_threshold(weight_paths: List[Path], y: np.ndarray,
                       probs: np.ndarray) -> Tuple[float, str]:

    ths = []
    for wp in weight_paths:
        val = load_validation_threshold(wp)
        if val is not None:
            ths.append(float(val[0]))
    if ths:
        return float(np.median(ths)), "validation_median"
    return youden_threshold(y, probs), "test_youden"


def sens_spec_at(y: np.ndarray, probs: np.ndarray, threshold: float) -> Tuple[float, float]:

    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return float(sens), float(spec)


def pr_curve_standard(y: np.ndarray, probs: np.ndarray,
                      n: int = 201) -> Tuple[np.ndarray, np.ndarray, float]:

    prec, rec, _ = precision_recall_curve(y, probs)
    if len(prec) == len(rec) + 1:
        prec = prec[:-1]
    grid = np.linspace(0.0, 1.0, n)
    out = np.empty_like(grid)
    for j, g in enumerate(grid):
        mask = rec >= g
        out[j] = float(prec[mask].max()) if mask.any() else 1.0
    trap = getattr(np, "trapezoid", np.trapz)
    ap = float(trap(out, grid))
    return grid, out, ap


def plot_roc_curve(y: np.ndarray, probs: np.ndarray, individual: List[np.ndarray],
                   out_path: Path, title: str) -> float:

    apply_global_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if PLOT_INDIVIDUAL_CURVES and individual:
        for i, p in enumerate(individual):
            f, t, _ = roc_curve(y, p)
            ax.plot(f, t, color=C_INDIV, linewidth=1.0, alpha=0.7,
                    label="Single models" if i == 0 else None, zorder=1)
    fpr, tpr, _ = roc_curve(y, probs)
    auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else float("nan")
    ax.plot(fpr, tpr, color=C_ENSEMBLE, linewidth=2.5,
            label=f"Ensemble (AUC = {auc:.4f})", zorder=3)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0,
            label="Chance", zorder=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return auc


def plot_pr_curve(y: np.ndarray, probs: np.ndarray, individual: List[np.ndarray],
                  out_path: Path, title: str) -> float:

    apply_global_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if PLOT_INDIVIDUAL_CURVES and individual:
        for i, p in enumerate(individual):
            g, pg, _ = pr_curve_standard(y, p)
            ax.plot(g, pg, color=C_INDIV, linewidth=1.0, alpha=0.7,
                    label="Single models" if i == 0 else None, zorder=1)
    grid, p_grid, ap = pr_curve_standard(y, probs)
    baseline = float(np.mean(y))
    ax.plot(grid, p_grid, color="#6a0dad", linewidth=2.5,
            label=f"Ensemble (AP = {ap:.4f})", zorder=3)
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.0,
               label=f"Baseline (prevalence = {baseline:.3f})", zorder=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return ap


def plot_threshold_sens_spec(y: np.ndarray, probs: np.ndarray, threshold: float,
                             out_path: Path, title: str, n: int = 201) -> None:

    apply_global_style()
    ths = np.linspace(0.0, 1.0, n)
    sens = np.full(n, np.nan)
    spec = np.full(n, np.nan)
    for i, th in enumerate(ths):
        sens[i], spec[i] = sens_spec_at(y, probs, float(th))

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot(ths, sens, color=C_SENS, linewidth=2.0, label="Sensitivity", zorder=3)
    ax.plot(ths, spec, color=C_SPEC, linewidth=2.0, label="Specificity", zorder=3)
    s_t, sp_t = sens_spec_at(y, probs, threshold)
    ax.axvline(threshold, color=C_MARK, linestyle="--", linewidth=1.6, zorder=2,
               label=f"Ensemble threshold = {threshold:.4f}")
    ax.plot([threshold], [s_t], marker="o", color=C_SENS, markersize=7, zorder=4)
    ax.plot([threshold], [sp_t], marker="o", color=C_SPEC, markersize=7, zorder=4)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y: np.ndarray, probs: np.ndarray, threshold: float,
                   out_path: Path, title: str) -> np.ndarray:

    apply_global_style()
    pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 14})
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Counts")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True, ax=axes[1],
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 14})
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Row-normalized (diagonal = recall)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return cm


def evaluate_mode(mode_dir: Path, loader, labels: np.ndarray, device: torch.device,
                  model_cfg: dict, report_dir_name: str, max_models: int = 0) -> Optional[dict]:

    if not mode_dir.exists():
        print(f"[警告] 模式目录不存在, 跳过: {mode_dir}")
        return None
    model_paths = discover_models(mode_dir, max_models=max_models)
    if not model_paths:
        print(f"[警告] {mode_dir} 下未找到任何 */best_model.pth, 跳过。")
        return None

    report_base = mode_dir
    override = REPORT_DIR_OVERRIDES.get(mode_dir.name)
    if override:
        ob = Path(override)
        report_base = ob if ob.is_absolute() else PROJECT_ROOT / ob
    report_dir = report_base / report_dir_name
    print(f"\n{'=' * 70}")
    print(f">>> 模式: {mode_dir.name} | 单模型 {len(model_paths)} 个 | 输出: {report_dir}")
    print("=" * 70)

    individual_probs: List[np.ndarray] = []
    valid_paths: List[Path] = []
    skipped_models: List[str] = []
    for idx, wp in enumerate(model_paths, start=1):
        probs = predict_probs(wp, loader, device, model_cfg)
        if probs is None:
            skipped_models.append(wp.parent.name)
            continue
        individual_probs.append(probs)
        valid_paths.append(wp)
        print(f"    [{idx}/{len(model_paths)}] {wp.parent.name}: 单模型 AUC="
              f"{roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan'):.4f}")

    if not individual_probs:
        print(f"[警告] {mode_dir.name} 下全部权重均因结构不兼容被跳过, 跳过该模式。")
        return None
    if skipped_models:
        print(f"    [提示] 已跳过 {len(skipped_models)} 个结构不兼容模型: {skipped_models}")

    ens_probs = np.mean(np.stack(individual_probs), axis=0)
    th, source = ensemble_threshold(valid_paths, labels, ens_probs)
    sens, spec = sens_spec_at(labels, ens_probs, th)
    m = compute_metrics(labels, ens_probs, threshold=th)
    auc = float(roc_auc_score(labels, ens_probs)) if len(np.unique(labels)) > 1 else float("nan")
    ap_sklearn = (float(average_precision_score(labels, ens_probs))
                  if len(np.unique(labels)) > 1 else float("nan"))

    print(f"    [阈值] {th:.4f} (来源: {source}) | 正在生成结果图 ...")
    plot_roc_curve(labels, ens_probs, individual_probs,
                   report_dir / "ensemble_roc.png",
                   f"{mode_dir.name} — Ensemble ROC")
    ap = plot_pr_curve(labels, ens_probs, individual_probs,
                       report_dir / "ensemble_pr.png",
                       f"{mode_dir.name} — Ensemble PR")
    plot_threshold_sens_spec(labels, ens_probs, th,
                             report_dir / "ensemble_threshold_sens_spec.png",
                             f"{mode_dir.name} — Threshold vs Sensitivity/Specificity")
    plot_confusion(labels, ens_probs, th,
                   report_dir / "ensemble_confusion_matrix.png",
                   f"{mode_dir.name} — Ensemble Confusion Matrix (thr={th:.4f})")

    row: Dict = {
        "mode": mode_dir.name,
        "report_dir": str(report_dir),
        "n_models": len(valid_paths),
        "model_ids": ";".join(p.parent.name for p in valid_paths),
        "skipped_models": ";".join(skipped_models),
        "auc": auc,
        "ap": ap,
        "ap_sklearn": ap_sklearn,
        "best_threshold": float(th),
        "threshold_source": source,
        "accuracy": float(m["accuracy"]),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": int(((ens_probs >= th) & (labels == 1)).sum()),
        "fn": int(((ens_probs < th) & (labels == 1)).sum()),
        "tn": int(((ens_probs < th) & (labels == 0)).sum()),
        "fp": int(((ens_probs >= th) & (labels == 0)).sum()),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(report_dir / "ensemble_summary.csv", index=False)
    pd.DataFrame({
        "sample_id": np.arange(1, len(labels) + 1),
        "test_labels": labels,
        "ensemble_probs": ens_probs,
        "pred_label": (ens_probs >= th).astype(int),
    }).to_csv(report_dir / "ensemble_predictions.csv", index=False)

    print(f"    AUC={auc:.4f} | AP={ap:.4f} | thr={th:.4f} [{source}] | "
          f"Sens={sens:.4f} Spec={spec:.4f} Acc={m['accuracy']:.4f}")
    print(f"    已保存: ensemble_roc.png | ensemble_pr.png | "
          f"ensemble_threshold_sens_spec.png | ensemble_confusion_matrix.png | "
          f"ensemble_summary.csv | ensemble_predictions.csv")
    return row


def main():
    global PLOT_INDIVIDUAL_CURVES
    parser = argparse.ArgumentParser(
        description="B/C/D/E 多模式集成模型评估与出图 (Soft Voting, 每模式单独报告)"
    )
    parser.add_argument("checkpoint_dirs", nargs="*", default=None,
                        help="模式目录, 如 outputs/checkpoints/Tweak outputs/checkpoints/E "
                             "(不传则使用脚本配置区 DEFAULT_CHECKPOINTS)")
    parser.add_argument("--test-csv", type=str, default=TEST_CSV)
    parser.add_argument("--test-raw-dir", type=str, default=TEST_RAW_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--proj-aug", action="store_true", default=PROJ_AUG,
                        help="测试期开启投影随机增强 (默认关闭, 保证结果可复现)")
    parser.add_argument("--report-dir-name", type=str, default=REPORT_DIR_NAME)
    parser.add_argument("--max-models", type=int, default=MAX_MODELS,
                        help="每模式最多使用前 N 个模型 (0=全部)")
    parser.add_argument("--plot-individual", action="store_true",
                        default=PLOT_INDIVIDUAL_CURVES,
                        help="ROC/PR 图上叠加各单模型淡色曲线")
    args = parser.parse_args()

    PLOT_INDIVIDUAL_CURVES = bool(args.plot_individual)

    raw_dirs = args.checkpoint_dirs if args.checkpoint_dirs else DEFAULT_CHECKPOINTS
    mode_dirs: List[Path] = []
    for d in raw_dirs:
        dp = Path(d)
        mode_dirs.append(dp if dp.is_absolute() else PROJECT_ROOT / dp)

    if not Path(args.test_csv).exists() and not Path(str(args.test_csv) + ".csv").exists():
        print(f"[错误] 测试集 CSV 不存在: {args.test_csv}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    test_loader, labels = build_test_loader(
        args.test_csv, cfg["data"], batch_size=args.batch_size,
        image_dir=args.test_raw_dir, projection_aug=bool(args.proj_aug),
    )
    print(f"[*] 设备: {device} | 测试集: {args.test_csv} | 样本数 {len(labels)} "
          f"(阳性 {int(labels.sum())} / 阴性 {int((1 - labels).sum())})")
    print(f"[*] 待评估模式 ({len(mode_dirs)}): {[p.name for p in mode_dirs]}")

    rows: List[dict] = []
    for mode_dir in mode_dirs:
        row = evaluate_mode(mode_dir, test_loader, labels, device, cfg["model"],
                            args.report_dir_name, max_models=args.max_models)
        if row is not None:
            rows.append(row)

    if not rows:
        print("\n[错误] 没有任何模式评估成功。")
        return

    print("\n" + "=" * 70)
    print(">>> 全部模式集成评估完成")
    summary = pd.DataFrame(rows)
    print(summary[["mode", "n_models", "auc", "ap", "best_threshold",
                   "threshold_source", "sensitivity", "specificity", "accuracy"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 70)


if __name__ == "__main__":
    main()
