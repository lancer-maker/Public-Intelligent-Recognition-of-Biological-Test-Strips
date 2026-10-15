
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state
from src.evaluation.metrics import compute_metrics
from src.utils.Fine_Tuning_helper import (
    build_test_loader,
    compute_metrics_with_best_threshold,
    load_config,
    predict_probs,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports" / "new_env_test"
DEFAULT_TEST_CSV = "data/test/labels/labels.csv"


def natural_sort_key(name: str) -> List:

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def make_name(path: Path) -> str:

    return f"{path.parent.name}_{path.stem}"


def parse_model_paths(paths_arg: List[str]) -> List[Path]:

    result: List[Path] = []
    for p in paths_arg:
        for piece in p.replace(";", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            path = Path(piece)
            if path.is_dir():
                found = sorted(path.rglob("*.pth"), key=lambda x: natural_sort_key(x.name))
                if not found:
                    print(f"[警告] 目录下无 .pth 文件: {piece}")
                result.extend(found)
            elif path.is_file():
                result.append(path)
            else:
                print(f"[警告] 模型路径不存在, 跳过: {piece}")
    seen = set()
    out: List[Path] = []
    for p in result:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def extract_state_dict(ckpt: dict) -> dict:

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def build_model(feature_dim: int, dropout_rate: float, device: torch.device) -> DualStreamStripNet:

    model = DualStreamStripNet(pretrained_2d=False, feature_dim=feature_dim, dropout_rate=dropout_rate)
    return model.to(device)


def load_validation_threshold(weight_path: Path) -> Optional[Tuple[float, float, float]]:

    metrics_csv = weight_path.parent / "metrics.csv"
    if not metrics_csv.exists():
        return None
    try:
        row = pd.read_csv(metrics_csv).iloc[0]
        th = row.get("best_threshold")
        if th is None or pd.isna(th):
            return None
        lo = row.get("threshold_ci_low", np.nan)
        hi = row.get("threshold_ci_high", np.nan)
        return (float(th),
                float(lo) if not pd.isna(lo) else np.nan,
                float(hi) if not pd.isna(hi) else np.nan)
    except Exception:
        return None


def _metric_row(
    name: str,
    weight_path: Path,
    auc: float,
    best_threshold: float,
    threshold_source: str,
    sens: float, spec: float, acc: float,
    sens_05: float, spec_05: float, acc_05: float,
    n_models: int,
    model_ids: str,
    th_ci_low: float = np.nan,
    th_ci_high: float = np.nan,
) -> dict:

    return {
        "name": name,
        "weight_path": str(weight_path),
        "n_models": n_models,
        "model_ids": model_ids,
        "auc": float(auc),
        "best_threshold": float(best_threshold),
        "threshold_source": threshold_source,
        "threshold_ci_low": float(th_ci_low) if not pd.isna(th_ci_low) else np.nan,
        "threshold_ci_high": float(th_ci_high) if not pd.isna(th_ci_high) else np.nan,
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "sens_05": float(sens_05),
        "spec_05": float(spec_05),
        "acc_05": float(acc_05),
    }


def evaluate_single_weight(
    weight_path: Path,
    loader,
    labels: np.ndarray,
    device: torch.device,
    model_cfg: dict,
) -> Optional[Tuple[dict, np.ndarray]]:

    feature_dim = int(model_cfg.get("feature_dim", 128))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))
    model = build_model(feature_dim, dropout_rate, device)
    model = load_model_state(model, str(weight_path))
    probs = predict_probs(model, loader, device)
    m = compute_metrics_with_best_threshold(labels, probs)

    val = load_validation_threshold(weight_path)
    if val is not None:
        th, lo, hi = val
        source = "validation_bootstrap_youden"
        m_th = compute_metrics(labels, probs, threshold=th)
        sens, spec, acc = m_th["sensitivity"], m_th["specificity"], m_th["accuracy"]
        print(f"    [阈值] 使用训练时验证集阈值 {th:.4f} (来源: {weight_path.parent / 'metrics.csv'})")
    else:
        th, lo, hi = m["best_threshold"], np.nan, np.nan
        source = "test_youden"
        sens, spec, acc = m["sensitivity"], m["specificity"], m["accuracy"]
        print(f"    [警告] 未找到对应验证集报告 {weight_path.parent / 'metrics.csv'}, 回退测试集约登阈值 {th:.4f}")

    row = _metric_row(make_name(weight_path), weight_path, m["auc"], th, source,
                      sens, spec, acc, m["sens_05"], m["spec_05"], m["acc_05"],
                      n_models=1, model_ids=str([weight_path.name]),
                      th_ci_low=lo, th_ci_high=hi)
    return row, probs


def evaluate_ensemble_ckpt(
    weight_path: Path,
    loader,
    labels: np.ndarray,
    device: torch.device,
    model_cfg: dict,
    ckpt: Optional[dict] = None,
) -> Optional[Tuple[dict, np.ndarray]]:

    if ckpt is None:
        ckpt = torch.load(weight_path, map_location="cpu")
    feature_dim = int(ckpt.get("feature_dim", int(model_cfg.get("feature_dim", 128))))
    dropout_rate = float(ckpt.get("dropout_rate", float(model_cfg.get("dropout_rate", 0.5))))
    model_ids = ckpt.get("model_ids") or []
    weights = ckpt.get("weights") or {}

    probs_list: List[np.ndarray] = []
    used_ids: List = []
    for pid in model_ids:
        state = weights.get(pid)
        if state is None:
            continue
        model = build_model(feature_dim, dropout_rate, device)
        model.load_state_dict(extract_state_dict(state))
        probs_list.append(predict_probs(model, loader, device))
        used_ids.append(pid)

    if not probs_list:
        print(f"[警告] 集合权重 {weight_path} 无有效成员权重, 跳过。")
        return None

    probs = np.mean(np.stack(probs_list), axis=0)
    m = compute_metrics_with_best_threshold(labels, probs)
    row = _metric_row(make_name(weight_path), weight_path, m["auc"], m["best_threshold"], "test_youden",
                      m["sensitivity"], m["specificity"], m["accuracy"],
                      m["sens_05"], m["spec_05"], m["acc_05"],
                      n_models=len(used_ids), model_ids=str(used_ids))
    return row, probs


def build_voting_ensemble(
    name: str,
    entries: List[Tuple[str, Path, np.ndarray]],
    labels: np.ndarray,
) -> Tuple[dict, np.ndarray]:

    probs = np.mean(np.stack([e[2] for e in entries]), axis=0)
    m = compute_metrics_with_best_threshold(labels, probs)

    ths: List[float] = []
    for _, wp, _ in entries:
        val = load_validation_threshold(wp)
        if val is not None:
            ths.append(val[0])
    if ths:
        th = float(np.median(ths))
        source = "validation_median"
        m_th = compute_metrics(labels, probs, threshold=th)
        sens, spec, acc = m_th["sensitivity"], m_th["specificity"], m_th["accuracy"]
        print(f"    [集合阈值] 各成员验证集阈值中位数 {th:.4f}")
    else:
        th, source = m["best_threshold"], "test_youden"
        sens, spec, acc = m["sensitivity"], m["specificity"], m["accuracy"]

    row = _metric_row(name, Path("(soft_voting)"), m["auc"], th, source,
                      sens, spec, acc, m["sens_05"], m["spec_05"], m["acc_05"],
                      n_models=len(entries), model_ids=str([e[0] for e in entries]))
    return row, probs


def save_predictions(out_dir: Path, name: str, labels: np.ndarray, probs: np.ndarray, best_threshold: float) -> None:

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "sample_id": np.arange(1, len(labels) + 1),
        "test_labels": labels,
        "test_probs": probs,
        "pred_label": (probs >= best_threshold).astype(int),
    })
    pred_df.to_csv(pred_dir / f"{name}_preds.csv", index=False)
    np.save(pred_dir / f"{name}_probs.npy", probs)


def _try_export_three_panel(row: dict, weight_path: Path) -> None:

    wdir = Path(weight_path).parent
    if not (wdir / "history.csv").exists():
        return
    m = re.match(r"^(\d+)_", str(row.get("name", "")))
    if m is None:
        return
    model_id = int(m.group(1))
    pred_csv = PROJECT_ROOT / "outputs" / "reports" / "new_env_test" / "predictions" \
        / f"{model_id}_best_model_preds.csv"
    if not pred_csv.exists():
        return
    try:
        from src.utils.figure_result import export_real_figure
        png = export_real_figure(model_id=model_id)
        print(f"    [三联图] 已生成: {png}")
    except Exception as exc:
        print(f"    [三联图] 生成失败 (跳过): {exc}")



def main():
    parser = argparse.ArgumentParser(
        description="新环境模型测试评估: 输入多少模型就评估多少, 并保留集合(Soft Voting)测试方式"
    )
    parser.add_argument("--test-csv", type=str, default=DEFAULT_TEST_CSV,
                        help=f"独立测试集 CSV 路径 (默认 {DEFAULT_TEST_CSV})")
    parser.add_argument("--test-raw-dir", type=str, default="",
                        help="测试集 raw 图片目录 (可选; 默认自动探测 CSV 同级/上一级 raw)")
    parser.add_argument("--model-paths", type=str, action="append", required=True,
                        help="要评估的模型权重路径 (可重复使用/逗号分隔/传目录递归收集 .pth)")
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch size (默认 8)")
    parser.add_argument("--no-ensemble-vote", action="store_true",
                        help="关闭对多个单模型自动生成 Soft Voting 集合评估")
    parser.add_argument("--no-three-panel", action="store_true",
                        help="关闭评估后自动生成论文三联图 (Loss|混淆矩阵|PR)")
    parser.add_argument("--proj-aug", action="store_true",
                        help="启用投影随机亮度/对比度增强 (默认关闭, 保证测试结果确定可复现)")
    parser.add_argument("--out-name", type=str, default="new_env_test", help="输出子目录名 (默认 new_env_test)")
    args = parser.parse_args()

    if not Path(args.test_csv).exists() and not Path(str(args.test_csv) + ".csv").exists():
        print(f"[错误] 测试集 CSV 不存在: {args.test_csv}")
        return

    model_paths = parse_model_paths(args.model_paths)
    if not model_paths:
        print("[错误] 未提供任何有效的模型权重路径。")
        return
    print(f"[*] 待评估模型权重: {len(model_paths)} 个")
    for p in model_paths:
        print(f"    - {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    test_loader, labels = build_test_loader(
        args.test_csv, data_cfg, batch_size=args.batch_size, image_dir=args.test_raw_dir,
        projection_aug=args.proj_aug,
    )
    print(f"[*] 设备: {device} | 测试集: {args.test_csv} | 样本数 {len(labels)} "
          f"(阳性 {int(labels.sum())} / 阴性 {int((1 - labels).sum())})")
    print(f"[*] 输出目录: {OUTPUT_DIR}")

    rows: List[dict] = []
    single_entries: List[Tuple[str, Path, np.ndarray]] = []
    for idx, wp in enumerate(model_paths, start=1):
        print(f"\n[{idx}/{len(model_paths)}] 评估: {wp.name}")
        ckpt = torch.load(wp, map_location="cpu")
        if isinstance(ckpt, dict) and "weights" in ckpt and "model_ids" in ckpt:
            ret = evaluate_ensemble_ckpt(wp, test_loader, labels, device, model_cfg, ckpt=ckpt)
        else:
            ret = evaluate_single_weight(wp, test_loader, labels, device, model_cfg)
        if ret is None:
            continue
        row, probs = ret
        rows.append(row)
        save_predictions(OUTPUT_DIR, row["name"], labels, probs, row["best_threshold"])
        if row["n_models"] == 1:
            single_entries.append((row["name"], wp, probs))
            if not args.no_three_panel:
                _try_export_three_panel(row, wp)
        print(f"    AUC={row['auc']:.4f} | thr={row['best_threshold']:.4f} "
              f"[{row['threshold_source']}] | Sens={row['sensitivity']:.4f} Spec={row['specificity']:.4f}")

    if not rows:
        print("[错误] 所有传入模型均评估失败, 退出。")
        return

    if not args.no_ensemble_vote and len(single_entries) >= 2:
        print(f"\n[集合评估] 对 {len(single_entries)} 个单模型做 Soft Voting 概率平均")
        ens_row, ens_probs = build_voting_ensemble("ensemble_vote", single_entries, labels)
        rows.append(ens_row)
        save_predictions(OUTPUT_DIR, ens_row["name"], labels, ens_probs, ens_row["best_threshold"])
        print(f"    AUC={ens_row['auc']:.4f} | thr={ens_row['best_threshold']:.4f} "
              f"[{ens_row['threshold_source']}] | Sens={ens_row['sensitivity']:.4f} Spec={ens_row['specificity']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "eval_summary.csv", index=False)

    per_model = summary_df[summary_df["n_models"] == 1]
    if not per_model.empty:
        per_model.to_csv(OUTPUT_DIR / "per_model_eval.csv", index=False)
    ensemble = summary_df[summary_df["n_models"] > 1]
    if not ensemble.empty:
        ensemble.to_csv(OUTPUT_DIR / "ensemble_eval.csv", index=False)

    print("\n" + "=" * 70)
    print(f">>> 新环境模型测试评估完成 | 输出目录: {OUTPUT_DIR}")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 70)


if __name__ == "__main__":
    main()
