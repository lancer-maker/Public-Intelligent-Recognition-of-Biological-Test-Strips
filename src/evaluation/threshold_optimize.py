import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Tuple, Dict, List, Optional
from .metrics import compute_metrics


def calculate_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float, float]:

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        metrics_05 = compute_metrics(y_true, y_prob, threshold=0.5)
        return 0.5, metrics_05['sensitivity'], metrics_05['specificity'], metrics_05['accuracy'], metrics_05['auc']

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))

    youden_indices = tpr + (1.0 - fpr) - 1.0
    
    best_idx = np.argmax(youden_indices)
    best_threshold = float(thresholds[best_idx])

    if np.isinf(best_threshold):
        best_threshold = 1.0

    best_metrics = compute_metrics(y_true, y_prob, threshold=best_threshold)

    return (
        best_threshold,
        best_metrics['sensitivity'],
        best_metrics['specificity'],
        best_metrics['accuracy'],
        roc_auc
    )


def compute_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if len(np.unique(y_true)) < 2 or len(y_true) < 2:
        return None, None, None

    rng = np.random.RandomState(seed)
    n = len(y_true)
    idxs = np.arange(n)
    aucs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(idxs, size=n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(yt, yp)))
        except ValueError:
            continue

    if not aucs:
        return None, None, None

    aucs = np.asarray(aucs)
    lo = float(np.percentile(aucs, (1.0 - alpha) / 2.0 * 100.0))
    hi = float(np.percentile(aucs, (1.0 + alpha) / 2.0 * 100.0))
    return float(np.mean(aucs)), lo, hi


def summarize_loocv_results(fold_results: List[Dict[str, float]]) -> pd.DataFrame:

    df = pd.DataFrame(fold_results)
    
    summary_mean = df.mean(numeric_only=True)
    summary_std = df.std(numeric_only=True)

    print("\n==================== LOOCV PERFORMANCE REPORT ====================")
    print(f"AUC:         {summary_mean['auc']:.4f} ± {summary_std['auc']:.4f}")
    print(f"Sensitivity: {summary_mean['sensitivity']:.4f} ± {summary_std['sensitivity']:.4f}")
    print(f"Specificity: {summary_mean['specificity']:.4f} ± {summary_std['specificity']:.4f}")
    print(f"Accuracy:    {summary_mean['accuracy']:.4f} ± {summary_std['accuracy']:.4f}")
    print(f"Optimal TH:  {summary_mean['best_threshold']:.4f} ± {summary_std['best_threshold']:.4f}")
    if 'auc_ci_low' in df.columns and 'auc_ci_high' in df.columns:
        ci_low_mean = df['auc_ci_low'].mean()
        ci_high_mean = df['auc_ci_high'].mean()
        n_valid = df[['auc_ci_low', 'auc_ci_high']].dropna().shape[0]
        print(f"AUC 95% CI:  [{ci_low_mean:.4f}, {ci_high_mean:.4f}]  (跨 {n_valid}/{len(df)} 折可计算)")
        print("  -> 小样本(每折仅数张)导致区间较宽")
    print("==================================================================")

    return df