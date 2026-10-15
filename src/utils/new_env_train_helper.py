
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_projection_transforms, get_train_transforms, get_val_transforms
from src.training.train_speed_up import make_dataloader
from src.utils.stage2_utils import (
    ProgressiveUnfreezeScheduler,
    create_stage2_optimizer,
    freeze_all_backbone,
)

N_BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 42


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(log_path: Path, logger_name: str = "new_env_train") -> logging.Logger:

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    file_handler = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr + (1.0 - fpr) - 1.0
    best_idx = int(np.argmax(youden))
    th = float(thresholds[best_idx])
    if np.isinf(th):
        th = 1.0
    return th


def bootstrap_median_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = N_BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    idxs = np.arange(n)
    thresholds = []
    for _ in range(n_bootstraps):
        idx = rng.choice(idxs, size=n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        thresholds.append(youden_threshold(yt, yp))
    if not thresholds:
        return 0.5, 0.5, 0.5
    thresholds = np.asarray(thresholds)
    return (float(np.median(thresholds)),
            float(np.percentile(thresholds, 2.5)),
            float(np.percentile(thresholds, 97.5)))


def read_csv_directly(csv_path: str) -> pd.DataFrame:

    if not Path(csv_path).exists() and Path(str(csv_path) + ".csv").exists():
        csv_path = str(csv_path) + ".csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"无法找到标注文件: {csv_path}")
    return pd.read_csv(csv_path)


def prepare_labels(df: pd.DataFrame, image_dir: Path) -> pd.DataFrame:

    required_columns = {'filename', 'label'}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"标注文件缺少必要列: {sorted(missing_columns)}；实际列为: {list(df.columns)}"
        )
    prepared = df.copy()
    prepared['filename'] = prepared['filename'].map(
        lambda filename: str(Path(filename).resolve())
        if Path(str(filename)).is_absolute()
        else str((image_dir / str(filename)).resolve())
    )
    return prepared


def build_dataloader(
    df: pd.DataFrame,
    image_dir: Path,
    transforms,
    projection_transforms,
    standard_dpi_size: Tuple[int, int],
    proj_length: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    proj_cache=None,
    skip_projection: bool = False,
) -> Tuple[DataLoader, np.ndarray]:

    prepared = prepare_labels(df, image_dir)
    ds = TestStripDataset(
        image_paths=prepared['filename'].tolist(),
        labels=prepared['label'].tolist(),
        transforms=transforms,
        projection_transforms=projection_transforms,
        standard_dpi_size=standard_dpi_size,
        proj_length=proj_length,
        proj_cache=proj_cache,
        skip_projection=skip_projection,
    )
    loader = make_dataloader(ds, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, device=device)
    return loader, prepared['label'].to_numpy(dtype=int)


def validate(model, val_loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray]:

    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, projs, labels in val_loader:
            imgs = imgs.to(device)
            projs = projs.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(imgs, projs)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    return val_loss, np.array(all_preds).flatten(), np.array(all_labels).flatten()


def load_dataset_frames(
    cfg: dict,
    project_root: Path,
    val_csv: str,
    val_raw_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:

    val_csv = str(val_csv)
    if not Path(val_csv).exists() and not Path(str(val_csv) + ".csv").exists():
        raise FileNotFoundError(f"验证集 CSV 不存在: {val_csv}")
    val_raw_dir = Path(val_raw_dir) if val_raw_dir else project_root / "data" / "raw"
    if not val_raw_dir.exists():
        raise FileNotFoundError(f"验证集 raw 目录不存在: {val_raw_dir}")

    df_p = prepare_labels(read_csv_directly(cfg['data']['orig_csv_path']),
                          project_root / "data" / "raw" / "P-24")
    anon_csv = cfg['data']['anon_csv_path']
    df_n = None
    if Path(anon_csv).exists() or Path(str(anon_csv) + ".csv").exists():
        df_n = prepare_labels(read_csv_directly(anon_csv),
                              project_root / "data" / "raw" / "N-16")
    train_df = pd.concat([df_p, df_n], ignore_index=True) if df_n is not None else df_p

    val_df = read_csv_directly(val_csv)
    if len(val_df) == 0:
        raise ValueError(f"验证集 CSV 为空 (只有表头?): {val_csv}")
    return train_df, val_df, val_raw_dir


def build_train_val_loaders(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_raw_dir: Path,
    device: torch.device,
    *,
    batch_size: Optional[int] = None,
    val_batch_size: int = 8,
    num_workers: int = 0,
    proj_cache=None,
    skip_projection: bool = False,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:

    standard_dpi_size = tuple(cfg['data']['standard_dpi_size'])
    proj_length = int(cfg['data']['proj_length'])
    train_batch = batch_size if batch_size else int(cfg['training']['stage1']['batch_size'])

    train_loader, _ = build_dataloader(
        train_df, PROJECT_ROOT / "data" / "raw" / "P-24",
        get_train_transforms(), get_projection_transforms(),
        standard_dpi_size, proj_length,
        batch_size=train_batch, shuffle=True,
        num_workers=num_workers, device=device, proj_cache=proj_cache,
        skip_projection=skip_projection,
    )
    val_loader, val_labels = build_dataloader(
        val_df, val_raw_dir,
        get_val_transforms(), get_projection_transforms(),
        standard_dpi_size, proj_length,
        batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, device=device, proj_cache=proj_cache,
        skip_projection=skip_projection,
    )
    return train_loader, val_loader, val_labels


def resolve_hyperparams(cfg: dict, args) -> None:

    if args.stage1_epochs is None:
        args.stage1_epochs = int(cfg['training']['stage1']['epochs'])
    if args.stage2_epochs is None:
        args.stage2_epochs = int(cfg['training']['stage2']['epochs'])
    if args.lr1 is None:
        args.lr1 = float(cfg['training']['stage1']['learning_rate'])
    if args.lr2 is None:
        args.lr2 = float(cfg['training']['stage2']['learning_rate'])
    if args.patience is None:
        args.patience = cfg['training']['early_stopping_patience']


def calc_pos_weight(train_loader: DataLoader) -> Tuple[float, int, int]:

    train_labels = np.asarray(train_loader.dataset.labels)
    pos_cnt = int((train_labels == 1).sum())
    neg_cnt = int((train_labels == 0).sum())
    pos_weight = neg_cnt / pos_cnt if pos_cnt > 0 else 1.0
    return float(pos_weight), pos_cnt, neg_cnt


def build_stage2_objects(
    cfg: dict,
    param_model,
    backbone_model=None,
    *,
    lr2: float,
    weight_decay: float,
    stage2_epochs: int,
):

    if backbone_model is None:
        backbone_model = param_model
    freeze_all_backbone(backbone_model)
    opt2, sched2 = create_stage2_optimizer(
        param_model, lr=lr2, weight_decay=weight_decay,
        t_max=stage2_epochs, eta_min=1e-7,
    )
    stage2_cfg = cfg['training']['stage2']
    unfreeze_interval = int(stage2_cfg.get('unfreeze_interval', 3))
    unfreeze_start = stage2_cfg.get('unfreeze_start_block')
    unfreeze_layers = stage2_cfg.get('unfreeze_num_layers')
    unfreezer = ProgressiveUnfreezeScheduler(
        backbone_model,
        interval=unfreeze_interval,
        start_block=int(unfreeze_start) if unfreeze_start is not None else None,
        num_layers=int(unfreeze_layers) if unfreeze_layers is not None else None,
    )
    return opt2, sched2, unfreezer
