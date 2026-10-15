
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.metrics import compute_metrics
from src.models.image_stream_only import ImageOnlyBase
from src.utils.model_helper import load_model_state
from src.utils.Fine_Tuning_helper import (
    build_test_loader,
    compute_metrics_with_best_threshold,
    load_config,
)
from new_env_test import (
    _metric_row,
    build_voting_ensemble,
    evaluate_single_weight,
    load_validation_threshold,
    make_name,
    natural_sort_key,
    save_predictions,
)

CHECKPOINT_NAME = "D"
TEST_CSV = r"data\test\labels\labels.csv"
TEST_RAW_DIR = ""
BATCH_SIZE = 8
DO_ENSEMBLE_VOTE = True
PROJECTION_AUG = False

CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / CHECKPOINT_NAME
REPORTS_DIR = CHECKPOINT_DIR / "reports"


def discover_models(checkpoint_dir: Path):

    return sorted(checkpoint_dir.glob("*/best_model.pth"),
                  key=lambda p: natural_sort_key(p.parent.name))


def evaluate_image_only_weight(weight_path, loader, labels, device, model_cfg):

    feature_dim = int(model_cfg.get("feature_dim", 128))
    dropout_rate = float(model_cfg.get("dropout_rate", 0.5))
    model = ImageOnlyBase(pretrained_2d=False, feature_dim=feature_dim, dropout_rate=dropout_rate)
    model = load_model_state(model, str(weight_path))
    model.to(device)
    model.eval()

    probs_list = []
    with torch.no_grad():
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs_list.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
    probs = np.asarray(probs_list, dtype=np.float32)

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

    row = _metric_row(make_name(weight_path), weight_path, m["auc"], th, source,
                      sens, spec, acc, m["sens_05"], m["spec_05"], m["acc_05"],
                      n_models=1, model_ids=str([weight_path.name]),
                      th_ci_low=lo, th_ci_high=hi)
    return row, probs


def _is_dual_stream(weight_path: Path) -> bool:

    try:
        ckpt = torch.load(weight_path, map_location="cpu")
    except Exception:
        return True
    return isinstance(ckpt, dict) and any(str(k).startswith("proj_stream") for k in ckpt)


def _try_export_three_panel(row: dict, weight_path: Path) -> None:

    wdir = Path(weight_path).parent
    if not (wdir / "history.csv").exists():
        return
    m = re.match(r"^(\d+)_", str(row.get("name", "")))
    if m is None:
        return
    model_id = int(m.group(1))
    pred_csv = REPORTS_DIR / "predictions" / f"{model_id}_best_model_preds.csv"
    if not pred_csv.exists():
        return
    try:
        from src.utils.figure_result import export_real_figure
        png = export_real_figure(
            model_id=model_id,
            history_csv=wdir / "history.csv",
            pred_csv=pred_csv,
            out_dir=REPORTS_DIR / "three_panel" / f"model{model_id}",
        )
        print(f"    [三联图] 已生成: {png}")
    except Exception as exc:
        print(f"    [三联图] 生成失败 (跳过): {exc}")


def main():
    if not CHECKPOINT_DIR.exists():
        print(f"[错误] 模型目录不存在: {CHECKPOINT_DIR}")
        return
    if not Path(TEST_CSV).exists() and not Path(str(TEST_CSV) + ".csv").exists():
        print(f"[错误] 测试集 CSV 不存在: {TEST_CSV}")
        return

    model_paths = discover_models(CHECKPOINT_DIR)
    if not model_paths:
        print(f"[错误] {CHECKPOINT_DIR} 下未找到任何 best_model.pth")
        return
    print(f"[*] 模型组: {CHECKPOINT_NAME} | 发现模型: {len(model_paths)} 个")
    for p in model_paths:
        print(f"    - {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    test_loader, labels = build_test_loader(
        TEST_CSV, data_cfg, batch_size=BATCH_SIZE, image_dir=TEST_RAW_DIR,
        projection_aug=PROJECTION_AUG,
    )
    print(f"[*] 设备: {device} | 测试集: {TEST_CSV} | 样本数 {len(labels)} "
          f"(阳性 {int(labels.sum())} / 阴性 {int((1 - labels).sum())})")
    print(f"[*] 输出目录: {REPORTS_DIR}")

    rows = []
    single_entries = []
    for idx, wp in enumerate(model_paths, start=1):
        print(f"\n[{idx}/{len(model_paths)}] 评估: {wp}")
        if _is_dual_stream(wp):
            ret = evaluate_single_weight(wp, test_loader, labels, device, model_cfg)
        else:
            print("    [提示] 单流模型 (无 proj_stream), 使用 ImageOnlyBase 评估")
            ret = evaluate_image_only_weight(wp, test_loader, labels, device, model_cfg)
        if ret is None:
            continue
        row, probs = ret
        rows.append(row)
        save_predictions(REPORTS_DIR, row["name"], labels, probs, row["best_threshold"])
        single_entries.append((row["name"], wp, probs))
        _try_export_three_panel(row, wp)
        print(f"    AUC={row['auc']:.4f} | thr={row['best_threshold']:.4f} "
              f"[{row['threshold_source']}] | Sens={row['sensitivity']:.4f} Spec={row['specificity']:.4f}")

    if not rows:
        print("[错误] 所有模型均评估失败, 退出。")
        return

    if DO_ENSEMBLE_VOTE and len(single_entries) >= 2:
        print(f"\n[集合评估] 对 {len(single_entries)} 个单模型做 Soft Voting 概率平均")
        ens_row, ens_probs = build_voting_ensemble("ensemble_vote", single_entries, labels)
        rows.append(ens_row)
        save_predictions(REPORTS_DIR, ens_row["name"], labels, ens_probs, ens_row["best_threshold"])
        print(f"    AUC={ens_row['auc']:.4f} | thr={ens_row['best_threshold']:.4f} "
              f"[{ens_row['threshold_source']}] | Sens={ens_row['sensitivity']:.4f} Spec={ens_row['specificity']:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(REPORTS_DIR / "eval_summary.csv", index=False)

    per_model = summary_df[summary_df["n_models"] == 1]
    if not per_model.empty:
        per_model.to_csv(REPORTS_DIR / "per_model_eval.csv", index=False)
    ensemble = summary_df[summary_df["n_models"] > 1]
    if not ensemble.empty:
        ensemble.to_csv(REPORTS_DIR / "ensemble_eval.csv", index=False)

    print("\n" + "=" * 70)
    print(f">>> {CHECKPOINT_NAME} 组模型测试评估完成 | 输出目录: {REPORTS_DIR}")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 70)


if __name__ == "__main__":
    main()
