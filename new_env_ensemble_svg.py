
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

import new_env_ensemble_test as ens_test

PRED_CSV = r"D:\work\temporary\ensemble_predictions.csv"
SUMMARY_CSV = r"D:\work\temporary\ensemble_summary.csv"

OUT_SVG_ROC = r"D:\work\temporary\ensemble_roc.svg"
OUT_SVG_PR = r"D:\work\temporary\ensemble_pr.svg"
OUT_SVG_CONFUSION = r"D:\work\temporary\ensemble_confusion_matrix.svg"
OUT_SVG_THRESHOLD = r"D:\work\temporary\ensemble_threshold_sens_spec.svg"

TITLE_TAG = ""
CLASSES = ["N", "P"]
PLOT_INDIVIDUAL_CURVES = False


def _resolve_col(df: pd.DataFrame, candidates) -> str:

    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"找不到所需列, 候选={candidates}, 实际={list(df.columns)}")


def _prepare_output(path_str: str) -> Path:

    p = Path(path_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main():
    pred_path, sum_path = Path(PRED_CSV), Path(SUMMARY_CSV)
    if not pred_path.exists():
        print(f"[错误] 找不到集成预测文件: {pred_path}")
        return

    ens_test.CLASSES = list(CLASSES)
    ens_test.PLOT_INDIVIDUAL_CURVES = bool(PLOT_INDIVIDUAL_CURVES)
    individual = []

    pred = pd.read_csv(pred_path)
    label_col = _resolve_col(pred, ["test_labels", "labels", "y_true"])
    prob_col = _resolve_col(pred, ["ensemble_probs", "test_probs", "probs"])
    y_true = pred[label_col].astype(int).to_numpy()
    y_prob = pred[prob_col].astype(float).to_numpy()

    threshold, source, tag = 0.5, "default(0.5)", TITLE_TAG
    if sum_path.exists():
        summ = pd.read_csv(sum_path)
        if len(summ) > 0:
            if "best_threshold" in summ.columns:
                threshold = float(summ.iloc[0]["best_threshold"])
                source = str(summ.iloc[0].get("threshold_source", "summary"))
            if not tag and "mode" in summ.columns:
                tag = str(summ.iloc[0]["mode"])
    prefix = f"{tag} — " if tag else ""

    out_roc = _prepare_output(OUT_SVG_ROC)
    out_pr = _prepare_output(OUT_SVG_PR)
    out_conf = _prepare_output(OUT_SVG_CONFUSION)
    out_thr = _prepare_output(OUT_SVG_THRESHOLD)

    print(f"[*] 输入预测: {pred_path}")
    print(f"[*] 输入汇总: {sum_path if sum_path.exists() else '(缺失, 阈值回退 0.5)'}")
    print(f"[*] 样本数: {len(y_true)} "
          f"(阳性 {int((y_true == 1).sum())} / 阴性 {int((y_true == 0).sum())})")
    print(f"[*] 决策阈值: {threshold:.4f} (来源: {source})")
    print("[*] 输出路径 (配置区原样使用):")
    for p in (out_roc, out_pr, out_conf, out_thr):
        print(f"      - {p}")

    auc = ens_test.plot_roc_curve(y_true, y_prob, individual, out_roc,
                                 f"{prefix}Ensemble ROC")
    ap = ens_test.plot_pr_curve(y_true, y_prob, individual, out_pr,
                               f"{prefix}Ensemble PR")
    ens_test.plot_confusion(y_true, y_prob, threshold, out_conf,
                           f"{prefix}Ensemble Confusion Matrix (thr={threshold:.4f})")
    ens_test.plot_threshold_sens_spec(y_true, y_prob, threshold, out_thr,
                                     f"{prefix}Threshold vs Sensitivity/Specificity")

    print(f"[*] AUC={auc:.4f} | AP={ap:.4f} | thr={threshold:.4f}")
    ok = True
    for p in (out_roc, out_pr, out_conf, out_thr):
        exists = p.exists()
        ok = ok and exists
        print(f"    [{'OK' if exists else '缺失'}] {p} "
              f"({p.stat().st_size if exists else 0} bytes)")
    print("[*] 4 张 SVG 图生成" + ("完成。" if ok else "存在缺失, 请检查上方日志。"))


if __name__ == "__main__":
    main()
