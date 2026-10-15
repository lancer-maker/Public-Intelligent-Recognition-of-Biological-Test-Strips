

from __future__ import annotations

import contextlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

YOLO11_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = YOLO11_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(YOLO11_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO11_DIR))

from dataset_config import (
    ALLOW_COUNT_MISMATCH,
    EXPECTED_COUNTS,
    NEGATIVE_DIR_ALIASES,
    POSITIVE_DIR_ALIASES,
    TEST_DIR,
    TRAIN_DIR,
    VAL_DIR,
)

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "YOLO11"
DATASET_DIR = OUTPUT_ROOT / "dataset"
WEIGHTS_DIR = OUTPUT_ROOT / "weights"
RUNS_DIR = OUTPUT_ROOT / "runs"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = REPORTS_DIR / "logs"
FIGURES_DIR = OUTPUT_ROOT / "figures"

SEEDS: List[int] = [42, 831, 2026, 7, 2691]

CLASS_NAMES: Tuple[str, str] = ("N", "P")

POSITIVE_INDEX: int = 1

SPLIT_KEYS: Tuple[str, str, str] = ("train", "val", "test")

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

SPLIT_DIR_CONFIG: Dict[str, str] = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}


def ensure_output_dirs() -> None:

    for d in (OUTPUT_ROOT, DATASET_DIR, WEIGHTS_DIR, RUNS_DIR, REPORTS_DIR, LOGS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class Tee:


    def __init__(self, *streams) -> None:

        self.streams = streams

    def write(self, data: str) -> int:

        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:

        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:

        return False


@contextlib.contextmanager
def tee_stdout(log_path: Path):

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    console = sys.stdout
    with open(log_path, "w", encoding="utf-8") as fh:
        with contextlib.redirect_stdout(Tee(console, fh)):
            yield log_path


def dump_json(path: Path, payload: Dict[str, Any]) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def environment_fingerprint() -> Dict[str, Any]:

    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = str(exc)
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        info["ultralytics_error"] = str(exc)
    return info


@dataclass
class SampleRecord:


    split: str
    filename: str
    label: int
    image_path: Path
    class_dir: Path


def _split_dir_value(split: str) -> str:

    if split not in SPLIT_DIR_CONFIG:
        raise ValueError(f"未知划分 '{split}', 应为 {list(SPLIT_DIR_CONFIG)}")
    return str(SPLIT_DIR_CONFIG[split]).strip()


def _config_constant_name(split: str) -> str:

    return {"train": "TRAIN_DIR", "val": "VAL_DIR", "test": "TEST_DIR"}[split]


def resolve_split_dir(split: str) -> Path:

    raw = _split_dir_value(split)
    const = _config_constant_name(split)
    if not raw:
        raise FileNotFoundError(
            f"[{split}] 数据集路径未配置: 请打开 YOLO11/dataset_config.py, 把 {const} 填成该划分的目录\n"
            f"  (目录下需含 negative/ 与 positive/ 两个类别子文件夹)"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"[{split}] {const} 指向的目录不存在: {path}")
    return path


def _match_class_dir(split_dir: Path, aliases: Sequence[str]) -> Optional[Path]:

    wanted = {a.strip().lower() for a in aliases}
    for child in sorted(split_dir.iterdir()):
        if child.is_dir() and child.name.strip().lower() in wanted:
            return child
    return None


def resolve_class_dirs(split: str) -> Tuple[Path, Path]:

    split_dir = resolve_split_dir(split)
    neg_dir = _match_class_dir(split_dir, NEGATIVE_DIR_ALIASES)
    pos_dir = _match_class_dir(split_dir, POSITIVE_DIR_ALIASES)
    if neg_dir is None or pos_dir is None:
        subdirs = [d.name for d in sorted(split_dir.iterdir()) if d.is_dir()]
        raise FileNotFoundError(
            f"[{split}] 目录 {split_dir} 下未同时找到阴性/阳性子文件夹\n"
            f"  阴性别名: {list(NEGATIVE_DIR_ALIASES)}\n"
            f"  阳性别名: {list(POSITIVE_DIR_ALIASES)}\n"
            f"  实际子文件夹: {subdirs}\n"
            f"  期望结构: {split_dir}/{{negative,positive}}/"
        )
    return neg_dir, pos_dir


def list_split_samples(split: str) -> List[SampleRecord]:

    neg_dir, pos_dir = resolve_class_dirs(split)
    records: List[SampleRecord] = []
    for class_dir, label in ((neg_dir, 0), (pos_dir, 1)):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(SampleRecord(split, path.name, label, path, class_dir))
    return records


def dataset_summary() -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []
    for split in SPLIT_KEYS:
        expected = EXPECTED_COUNTS.get(split)
        try:
            records = list_split_samples(split)
        except FileNotFoundError as exc:
            rows.append({
                "split": split, "configured": False, "path": "",
                "n": 0, "n_neg": 0, "n_pos": 0, "expected": expected,
                "match": False, "message": str(exc),
            })
            continue
        n_neg = sum(1 for r in records if r.label == 0)
        n_pos = sum(1 for r in records if r.label == 1)
        rows.append({
            "split": split, "configured": True, "path": str(resolve_split_dir(split)),
            "n": len(records), "n_neg": n_neg, "n_pos": n_pos, "expected": expected,
            "match": (expected is None or len(records) == expected), "message": "",
        })
    return rows


def check_expected_counts() -> List[Dict[str, Any]]:

    summary = dataset_summary()
    for row in summary:
        if not row["configured"] or row["expected"] is None or row["match"]:
            continue
        msg = (f"[warn] {row['split']} 样本数 {row['n']} 与期望 {row['expected']} 不一致 "
               f"(N={row['n_neg']}, P={row['n_pos']}): {row['path']}")
        if not ALLOW_COUNT_MISMATCH:
            raise ValueError(msg)
        print(msg)
    return summary


def build_normalized_dataset(force: bool = False, verbose: bool = True) -> Path:

    ensure_output_dirs()
    for split in SPLIT_KEYS:
        for class_dir, class_name in zip(resolve_class_dirs(split), CLASS_NAMES):
            dst_dir = DATASET_DIR / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted(class_dir.rglob("*")):
                if not (src.is_file() and src.suffix.lower() in IMAGE_EXTENSIONS):
                    continue
                dst = dst_dir / src.name
                if force or not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    shutil.copy2(src, dst)
            if verbose:
                print(f"[dataset] {split}/{class_name} -> {dst_dir}")
    check_expected_counts()
    return DATASET_DIR


def dataset_root_for_training(verbose: bool = True, force: bool = False) -> Path:

    dirs = [resolve_split_dir(s) for s in SPLIT_KEYS]
    check_expected_counts()
    if len({d.parent.resolve() for d in dirs}) == 1:
        root = dirs[0].parent.resolve()
        if verbose:
            print(f"[data] 使用用户数据根目录 (零拷贝): {root}")
        return root
    if verbose:
        print("[data] train/val/test 不在同一父目录, 归一化复制到 outputs/YOLO11/dataset ...")
    return build_normalized_dataset(force=force, verbose=verbose)


def load_image_paths(split: str) -> Tuple[List[Path], np.ndarray, List[str]]:

    records = list_split_samples(split)
    paths = [r.image_path for r in records]
    labels = np.asarray([r.label for r in records], dtype=int)
    names = [r.filename for r in records]
    return paths, labels, names


def resolve_pretrained_weights(model_name: str) -> Path:

    ensure_output_dirs()
    raw = Path(model_name)
    if raw.exists():
        return raw
    target = WEIGHTS_DIR / raw.name
    if target.exists():
        return target
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        downloaded = attempt_download_asset(raw.name, dir=str(WEIGHTS_DIR))
        if downloaded and Path(downloaded).exists():
            return Path(downloaded)
    except Exception as exc:
        print(f"[warn] 预训练权重下载失败 ({exc}), 交由 Ultralytics 自动处理: {raw.name}")
    return raw


def resolve_positive_index(names: Any) -> int:

    if isinstance(names, dict):
        items = list(names.items())
    else:
        items = list(enumerate(list(names or [])))
    for key, value in items:
        name = str(value).strip().lower()
        if name in ("p", "pos", "positive", "1", "阳", "阳性"):
            return int(key)
    for key, value in items:
        if "p" == str(value).strip().lower()[:1] and str(value).strip().lower() != "n":
            return int(key)
    return POSITIVE_INDEX


def load_val_transform(imgsz: int):

    from ultralytics.data.augment import classify_transforms

    return classify_transforms(size=imgsz)


def predict_positive_probs(weights_path: Path, image_paths: Sequence[Path], imgsz: int = 224,
                           device: str = "", batch_size: int = 32,
                           positive_index: Optional[int] = None) -> np.ndarray:

    import cv2
    import torch
    from PIL import Image
    from ultralytics import YOLO

    paths = [Path(p) for p in image_paths]
    if not paths:
        return np.zeros((0,), dtype=np.float32)

    model = YOLO(str(weights_path))
    pos_idx = int(positive_index) if positive_index is not None else resolve_positive_index(model.names)
    transform = load_val_transform(imgsz)

    net = model.model
    if device in ("", None, "auto"):
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif str(device).isdigit() or str(device) == "cuda":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(str(device))
    net = net.to(torch_device).eval()

    probs: List[float] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            tensors = []
            for path in chunk:
                bgr = cv2.imread(str(path))
                if bgr is None:
                    raise FileNotFoundError(f"无法读取图像: {path}")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensors.append(transform(Image.fromarray(rgb)))
            batch = torch.stack(tensors, 0).to(torch_device).float()
            logits = net(batch)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            logits = logits.float().reshape(batch.shape[0], -1)
            probs.extend(torch.softmax(logits, dim=1)[:, pos_idx].cpu().tolist())
    return np.asarray(probs, dtype=np.float32)


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:

    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    best = float(thresholds[int(np.argmax(j))])
    if not np.isfinite(best):
        best = 1.0
    return best


def pr_curve_standard(y_true: np.ndarray, y_prob: np.ndarray,
                      n_grid: int = 201) -> Tuple[np.ndarray, np.ndarray, float]:

    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return np.linspace(0, 1, n_grid), np.ones(n_grid), float("nan")
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    if len(prec) == len(rec) + 1:
        prec = prec[:-1]
    grid = np.linspace(0.0, 1.0, n_grid)
    envelope = np.empty_like(grid)
    for j, g in enumerate(grid):
        mask = rec >= g
        envelope[j] = float(prec[mask].max()) if mask.any() else 1.0
    trap = getattr(np, "trapezoid", np.trapz)
    return grid, envelope, float(trap(envelope, grid))


def metric_row(split: str, source: str, threshold_source: str, threshold: float,
               y_true: np.ndarray, y_prob: np.ndarray,
               ap_override: Optional[float] = None) -> Dict[str, Any]:

    from src.evaluation.metrics import compute_metrics
    from src.evaluation.threshold_optimize import compute_auc_ci

    base = compute_metrics(np.asarray(y_true), np.asarray(y_prob), threshold=float(threshold))
    _, _, ap = pr_curve_standard(y_true, y_prob)
    if ap_override is not None:
        ap = float(ap_override)
    y_true_int = np.asarray(y_true).astype(int)
    auc_mean, auc_lo, auc_hi = compute_auc_ci(y_true_int, np.asarray(y_prob, dtype=np.float32))
    return {
        "source": source,
        "split": split,
        "threshold_source": threshold_source,
        "threshold": float(threshold),
        "n": int(len(y_true_int)),
        "n_pos": int((y_true_int == 1).sum()),
        "n_neg": int((y_true_int == 0).sum()),
        "auc": float(base["auc"]),
        "auc_ci_low": auc_lo,
        "auc_ci_high": auc_hi,
        "ap": float(ap),
        "accuracy": float(base["accuracy"]),
        "sensitivity": float(base["sensitivity"]),
        "specificity": float(base["specificity"]),
        "precision": float(base["precision"]),
        "f1_score": float(base["f1_score"]),
        "tp": int(base["tp"]),
        "fp": int(base["fp"]),
        "tn": int(base["tn"]),
        "fn": int(base["fn"]),
    }


def prediction_frame(split: str, source: str, filenames: Sequence[str], y_true: np.ndarray,
                     y_prob: np.ndarray, threshold: float) -> "Any":

    import pandas as pd

    y_prob = np.asarray(y_prob, dtype=np.float64)
    return pd.DataFrame({
        "source": source,
        "split": split,
        "filename": list(filenames),
        "label": np.asarray(y_true).astype(int),
        "prob_P": y_prob,
        "threshold": float(threshold),
        "pred": (y_prob >= float(threshold)).astype(int),
    })


def write_csv(df: "Any", path: Path) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def format_metric_line(row: Dict[str, Any]) -> str:

    return (f"{row['source']:<12} {row['split']:<5} {row['threshold_source']:<11} "
            f"thr={row['threshold']:.3f} AUC={row['auc']:.4f} AP={row['ap']:.4f} "
            f"Acc={row['accuracy']:.4f} Sens={row['sensitivity']:.4f} Spec={row['specificity']:.4f}")
