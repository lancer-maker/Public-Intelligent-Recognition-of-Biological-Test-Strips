

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from yolo11_common import (
    FIGURES_DIR,
    REPORTS_DIR,
    RUNS_DIR,
    SEEDS,
    dump_json,
    ensure_output_dirs,
    environment_fingerprint,
    format_metric_line,
    load_image_paths,
    metric_row,
    predict_positive_probs,
    prediction_frame,
    pr_curve_standard,
    set_seed,
    write_csv,
    youden_threshold,
)


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="YOLO11 分类模型评估与多种子集成")
    parser.add_argument("--tag", default=None,
                        help="训练分组名 (默认自动选择 outputs/YOLO11/runs 下最近一次训练的分组)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="推理输入尺寸, 必须与训练时一致 (默认 224)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help=f"参与评估/集成的种子 (默认自动发现该 tag 下全部; 推荐 {SEEDS})")
    parser.add_argument("--batch", type=int, default=32, help="推理批大小")
    parser.add_argument("--device", default="0", help="设备: 0 / cpu")
    parser.add_argument("--no-figures", action="store_true", help="不出图 (跳过 matplotlib 绘图)")
    return parser.parse_args()


def resolve_tag(tag: Optional[str]) -> str:

    if tag:
        return tag
    if not RUNS_DIR.is_dir():
        raise FileNotFoundError(f"未找到训练输出目录: {RUNS_DIR}, 请先运行 train_yolo11_cls.py")
    candidates = [d for d in RUNS_DIR.iterdir() if d.is_dir() and any(d.glob("seed_*/weights/best.pt"))]
    if not candidates:
        raise FileNotFoundError(f"{RUNS_DIR} 下没有可用的 seed_*/weights/best.pt, 请先训练")
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    print(f"[info] 未指定 --tag, 自动选择最近训练的分组: {candidates[0].name}")
    return candidates[0].name


def discover_seed_runs(tag: str, seeds: Optional[Sequence[int]] = None) -> List[Tuple[int, Path]]:

    group_dir = RUNS_DIR / tag
    if not group_dir.is_dir():
        raise FileNotFoundError(f"训练分组目录不存在: {group_dir}")
    found: List[Tuple[int, Path]] = []
    for run_dir in sorted(group_dir.glob("seed_*")):
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            continue
        try:
            seed = int(run_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            print(f"[warn] 无法从目录名解析种子, 已跳过: {run_dir.name}")
            continue
        found.append((seed, best))
    if seeds is not None:
        wanted = set(int(s) for s in seeds)
        found = [x for x in found if x[0] in wanted]
    found.sort(key=lambda x: x[0])
    if not found:
        raise FileNotFoundError(f"{group_dir} 下没有匹配的权重 (seed_*/weights/best.pt)")
    return found


def _pyplot():

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_roc(y_true: np.ndarray, curves: List[Tuple[str, np.ndarray]],
             out_path: Path, title: str) -> None:

    from sklearn.metrics import roc_auc_score, roc_curve

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    for label, probs in curves:
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
        ax.plot(fpr, tpr, lw=1.6, label=f"{label} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr(y_true: np.ndarray, curves: List[Tuple[str, np.ndarray]],
            out_path: Path, title: str) -> None:

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    for label, probs in curves:
        grid, envelope, ap = pr_curve_standard(y_true, probs)
        ax.plot(grid, envelope, lw=1.6, label=f"{label} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                   out_path: Path, title: str) -> None:

    plt = _pyplot()
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    for ax, data, subtitle, fmt in (
        (axes[0], cm, "counts", "d"),
        (axes[1], cm_norm, "row-normalized (recall)", ".2f"),
    ):
        vmax = data.max() if data.max() > 0 else 1
        im = ax.pcolormesh(np.arange(3), np.arange(3), data, cmap="Blues",
                           vmin=0, vmax=vmax, edgecolors="white", linewidth=0.5)
        ax.set_xticks([0.5, 1.5], labels=["N (0)", "P (1)"])
        ax.set_yticks([0.5, 1.5], labels=["N (0)", "P (1)"])
        ax.invert_yaxis()
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{subtitle}")
        for i in range(2):
            for j in range(2):
                value = data[i, j]
                txt = f"{int(value):d}" if fmt == "d" else f"{value:.2f}"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                        color="white" if value > (vmax * 0.6) else "black",
                        fontsize=11)
        from matplotlib.patches import Rectangle

        cax = ax.inset_axes([1.03, 0.0, 0.045, 1.0])
        n_steps = 64
        for k in range(n_steps):
            frac = k / (n_steps - 1)
            cax.add_patch(Rectangle((0.0, frac - 0.5 / n_steps), 1.0, 1.0 / n_steps,
                                    facecolor=im.cmap(frac), edgecolor="none"))
        cax.set_xlim(0.0, 1.0)
        cax.set_ylim(0.0, 1.0)
        cax.set_xticks([])
        cax.set_yticks([0.0, 0.5, 1.0])
        cax.set_yticklabels([f"{0:g}", f"{vmax / 2:g}", f"{vmax:g}"], fontsize=8)
    fig.suptitle(f"{title} (threshold={float(threshold):.3f})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_threshold_scan(y_true: np.ndarray, y_prob: np.ndarray, threshold: float,
                        out_path: Path, title: str) -> None:

    from sklearn.metrics import roc_curve

    plt = _pyplot()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, 501)
    sens, spec, acc = [], [], []
    for th in grid:
        pred = (y_prob >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        sens.append(tp / (tp + fn) if (tp + fn) else 0.0)
        spec.append(tn / (tn + fp) if (tn + fp) else 0.0)
        acc.append((tp + tn) / max(len(y_true), 1))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.plot(grid, sens, label="Sensitivity", lw=1.6)
    ax.plot(grid, spec, label="Specificity", lw=1.6)
    ax.plot(grid, acc, label="Accuracy", lw=1.6, ls="--")
    ax.axvline(float(threshold), color="crimson", ls=":", lw=1.4,
               label=f"chosen threshold={float(threshold):.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:

    args = parse_args()
    ensure_output_dirs()
    set_seed(SEEDS[0] if SEEDS else 42)

    tag = resolve_tag(args.tag)
    seed_runs = discover_seed_runs(tag, args.seeds)
    print("=" * 78)
    print(f"YOLO11 分类评估 | tag={tag} | imgsz={args.imgsz} | 种子={[s for s, _ in seed_runs]}")
    for key, value in environment_fingerprint().items():
        print(f"  {key}: {value}")
    print("=" * 78)

    val_paths, y_val, val_names = load_image_paths("val")
    test_paths, y_test, test_names = load_image_paths("test")
    print(f"验证集 {len(val_paths)} 张 (阳性 {int((y_val == 1).sum())}), "
          f"测试集 {len(test_paths)} 张 (阳性 {int((y_test == 1).sum())})")

    figures_dir = FIGURES_DIR / tag
    metric_rows: List[Dict[str, object]] = []
    val_pred_frames: List[pd.DataFrame] = []
    test_pred_frames: List[pd.DataFrame] = []
    per_seed: Dict[int, Dict[str, object]] = {}

    for seed, weights in seed_runs:
        print(f"\n[seed {seed}] 推理: {weights}")
        p_val = predict_positive_probs(weights, val_paths, args.imgsz, args.device, args.batch)
        p_test = predict_positive_probs(weights, test_paths, args.imgsz, args.device, args.batch)
        thr_val = youden_threshold(y_val, p_val)
        per_seed[seed] = {"weights": str(weights), "val": p_val, "test": p_test, "threshold": thr_val}

        source = f"seed_{seed}"
        for split, y_true, prob in (("val", y_val, p_val), ("test", y_test, p_test)):
            metric_rows.append(metric_row(split, source, "fixed_0.5", 0.5, y_true, prob))
            metric_rows.append(metric_row(split, source, "val_youden", thr_val, y_true, prob))
        val_pred_frames.append(prediction_frame("val", source, val_names, y_val, p_val, thr_val))
        test_pred_frames.append(prediction_frame("test", source, test_names, y_test, p_test, thr_val))
        print("  " + format_metric_line(metric_row("val", source, "val_youden", thr_val, y_val, p_val)))
        print("  " + format_metric_line(metric_row("test", source, "val_youden", thr_val, y_test, p_test)))

    ens_val = np.mean([per_seed[s]["val"] for s, _ in seed_runs], axis=0)
    ens_test = np.mean([per_seed[s]["test"] for s, _ in seed_runs], axis=0)
    ens_thr = youden_threshold(y_val, ens_val)
    member_thr = float(np.median([per_seed[s]["threshold"] for s, _ in seed_runs]))

    for split, y_true, prob in (("val", y_val, ens_val), ("test", y_test, ens_test)):
        metric_rows.append(metric_row(split, "ensemble", "fixed_0.5", 0.5, y_true, prob))
        metric_rows.append(metric_row(split, "ensemble", "val_youden", ens_thr, y_true, prob))
    val_pred_frames.append(prediction_frame("val", "ensemble", val_names, y_val, ens_val, ens_thr))
    test_pred_frames.append(prediction_frame("test", "ensemble", test_names, y_test, ens_test, ens_thr))
    print("\n[集成 soft-voting]")
    print("  " + format_metric_line(metric_row("val", "ensemble", "val_youden", ens_thr, y_val, ens_val)))
    print("  " + format_metric_line(metric_row("test", "ensemble", "val_youden", ens_thr, y_test, ens_test)))

    metrics_df = pd.DataFrame(metric_rows)
    write_csv(metrics_df, REPORTS_DIR / "per_seed_metrics.csv")
    write_csv(pd.concat(val_pred_frames, ignore_index=True), REPORTS_DIR / "per_seed_predictions_val.csv")
    write_csv(pd.concat(test_pred_frames, ignore_index=True), REPORTS_DIR / "per_seed_predictions_test.csv")

    ens_metrics = metrics_df[metrics_df["source"] == "ensemble"].copy()
    write_csv(ens_metrics, REPORTS_DIR / "ensemble_metrics.csv")
    write_csv(val_pred_frames[-1], REPORTS_DIR / "ensemble_predictions_val.csv")
    write_csv(test_pred_frames[-1], REPORTS_DIR / "ensemble_predictions_test.csv")

    def best_seed_by(split: str, y_true: np.ndarray) -> Tuple[int, float]:

        scored = []
        for seed, _ in seed_runs:
            prob = per_seed[seed]["val"] if split == "val" else per_seed[seed]["test"]
            row = metric_row(split, f"seed_{seed}", "fixed_0.5", 0.5, y_true, prob)
            scored.append((float(row["auc"]), seed))
        scored.sort(reverse=True)
        return scored[0][1], scored[0][0]

    best_val_seed, best_val_auc = best_seed_by("val", y_val)
    best_test_seed, best_test_auc = best_seed_by("test", y_test)
    best_val_row = metric_row("test", f"best_single_val_auc(seed_{best_val_seed})", "val_youden",
                              per_seed[best_val_seed]["threshold"], y_test, per_seed[best_val_seed]["test"])
    best_test_row = metric_row("test", f"best_single_test_auc(seed_{best_test_seed})", "val_youden",
                               per_seed[best_test_seed]["threshold"], y_test, per_seed[best_test_seed]["test"])

    ens_test_row = ens_metrics[(ens_metrics["split"] == "test")
                               & (ens_metrics["threshold_source"] == "val_youden")].iloc[0].to_dict()
    ens_val_row = ens_metrics[(ens_metrics["split"] == "val")
                              & (ens_metrics["threshold_source"] == "val_youden")].iloc[0].to_dict()

    summary_rows = [
        {
            "method": "YOLO11-cls ensemble (soft voting)",
            "n_models": len(seed_runs),
            "seeds": ",".join(str(s) for s, _ in seed_runs),
            "threshold": ens_thr,
            "member_threshold_median": member_thr,
            "val_auc": ens_val_row["auc"], "test_auc": ens_test_row["auc"],
            "test_auc_ci_low": ens_test_row["auc_ci_low"], "test_auc_ci_high": ens_test_row["auc_ci_high"],
            "test_ap": ens_test_row["ap"], "test_accuracy": ens_test_row["accuracy"],
            "test_sensitivity": ens_test_row["sensitivity"], "test_specificity": ens_test_row["specificity"],
            "test_precision": ens_test_row["precision"], "test_f1": ens_test_row["f1_score"],
            "note": "阈值=集成验证集概率约登; 测试集仅用于报告",
        },
        {
            "method": f"YOLO11-cls best single (by val AUC, seed {best_val_seed})",
            "n_models": 1, "seeds": str(best_val_seed),
            "threshold": per_seed[best_val_seed]["threshold"], "member_threshold_median": np.nan,
            "val_auc": best_val_auc, "test_auc": best_val_row["auc"],
            "test_auc_ci_low": best_val_row["auc_ci_low"], "test_auc_ci_high": best_val_row["auc_ci_high"],
            "test_ap": best_val_row["ap"], "test_accuracy": best_val_row["accuracy"],
            "test_sensitivity": best_val_row["sensitivity"], "test_specificity": best_val_row["specificity"],
            "test_precision": best_val_row["precision"], "test_f1": best_val_row["f1_score"],
            "note": "无泄漏选择: 仅依据验证集 AUC 挑种子",
        },
        {
            "method": f"YOLO11-cls best single (by test AUC, seed {best_test_seed})",
            "n_models": 1, "seeds": str(best_test_seed),
            "threshold": per_seed[best_test_seed]["threshold"], "member_threshold_median": np.nan,
            "val_auc": metric_row("val", "x", "fixed_0.5", 0.5, y_val, per_seed[best_test_seed]["val"])["auc"],
            "test_auc": best_test_auc, "test_auc_ci_low": best_test_row["auc_ci_low"],
            "test_auc_ci_high": best_test_row["auc_ci_high"],
            "test_ap": best_test_row["ap"], "test_accuracy": best_test_row["accuracy"],
            "test_sensitivity": best_test_row["sensitivity"], "test_specificity": best_test_row["specificity"],
            "test_precision": best_test_row["precision"], "test_f1": best_test_row["f1_score"],
            "note": "⚠ 仅供参考(使用测试集选择, 存在选择性偏差), 论文中不建议作为主结果",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    write_csv(summary_df, REPORTS_DIR / "ensemble_summary.csv")

    leaky_rows = []
    for name, prob in [("ensemble", ens_test)] + [(f"seed_{s}", per_seed[s]["test"]) for s, _ in seed_runs]:
        thr_test = youden_threshold(y_test, prob)
        leaky_rows.append(metric_row("test", name, "test_youden(ref)", thr_test, y_test, prob))
    write_csv(pd.DataFrame(leaky_rows), REPORTS_DIR / "leaky_reference_test_youden.csv")

    if not args.no_figures:
        try:
            seed_curves = [(f"seed {s}", per_seed[s]["test"]) for s, _ in seed_runs]
            plot_roc(y_test, seed_curves + [("ensemble", ens_test)],
                     figures_dir / "roc_test.svg", f"YOLO11-cls ROC (test, {tag})")
            plot_roc(y_val, [(f"seed {s}", per_seed[s]["val"]) for s, _ in seed_runs] + [("ensemble", ens_val)],
                     figures_dir / "roc_val.svg", f"YOLO11-cls ROC (val, {tag})")
            plot_pr(y_test, [("ensemble", ens_test)], figures_dir / "pr_test.svg",
                    f"YOLO11-cls PR (test, {tag})")
            plot_confusion(y_test, ens_test, ens_thr, figures_dir / "confusion_matrix_test.svg",
                           f"YOLO11-cls ensemble (test, {tag})")
            plot_threshold_scan(y_test, ens_test, ens_thr, figures_dir / "threshold_scan_test.svg",
                                f"YOLO11-cls ensemble (test, {tag})")
            print(f"\n图表已保存: {figures_dir}")
        except Exception as exc:
            print(f"[warn] 绘图失败 ({exc}), 指标 CSV 已正常保存")

    dump_json(REPORTS_DIR / f"eval_config_{tag}.json", {
        "tag": tag,
        "imgsz": args.imgsz,
        "seeds": [s for s, _ in seed_runs],
        "weights": {str(s): per_seed[s]["weights"] for s, _ in seed_runs},
        "val_thresholds": {str(s): float(per_seed[s]["threshold"]) for s, _ in seed_runs},
        "ensemble_threshold": float(ens_thr),
        "member_threshold_median": member_thr,
        "environment": environment_fingerprint(),
    })

    print("\n" + "=" * 78)
    print("结果汇总 (测试集, 阈值=验证集约登)")
    print(summary_df[["method", "threshold", "test_auc", "test_ap", "test_accuracy",
                      "test_sensitivity", "test_specificity"]].to_string(index=False))
    print("=" * 78)
    print(f"指标文件: {REPORTS_DIR / 'per_seed_metrics.csv'}")
    print(f"逐样本概率: {REPORTS_DIR / 'per_seed_predictions_test.csv'}")
    print(f"汇总表: {REPORTS_DIR / 'ensemble_summary.csv'}")


if __name__ == "__main__":
    main()
