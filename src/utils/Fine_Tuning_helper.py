

import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_projection_transforms, get_val_transforms
from src.data.utils import extract_1d_projection_resampled, read_image_rgb, resize_to_standard_dpi

DEFAULT_IMAGE_SIZE = (220, 505)
DEFAULT_PROJ_LENGTH = 512
CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "configs" / "main_config.yaml"
TEST_REPORT_ROOT = PROJECT_ROOT / "outputs" / "reports" / "test"
RUN_NAME = "ensemble_finetune"
RUN_DIR = TEST_REPORT_ROOT / RUN_NAME
FINETUNE_WEIGHT_NAME = "finetuned_classifier.pth"
ORIGINAL_WEIGHT_NAME = "best_model.pth"
TEST_CSV = ""


def set_seed(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str = str(CONFIG_PATH)) -> dict:

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_model_ids() -> List[int]:

    ids: set = set()
    for d in CHECKPOINT_ROOT.glob("P*"):
        if not d.is_dir():
            continue
        m = re.match(r"^P(\d+)$", d.name)
        if m and (d / ORIGINAL_WEIGHT_NAME).exists():
            ids.add(int(m.group(1)))
    return sorted(ids)


def parse_model_ids(spec: str) -> List[int]:

    if spec.strip().lower() == "all":
        return discover_model_ids()
    ids: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return sorted(set(ids))


def get_hard_augmentation() -> A.Compose:

    return A.Compose([
        A.RandomBrightnessContrast(p=0.8, brightness_limit=0.08, contrast_limit=0.08),
        A.CLAHE(clip_limit=1.5, tile_grid_size=(4, 4), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class HardMiningDataset(Dataset):


    def __init__(self, samples: List[Dict[str, object]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        return item["image_tensor"], item["proj_tensor"], torch.tensor(float(item["label"]), dtype=torch.float32)


def load_training_records() -> List[Dict[str, object]]:

    records: List[Dict[str, object]] = []
    root = PROJECT_ROOT / "data"
    for folder_name, csv_name in [("P-24", "P"), ("N-16", "N")]:
        csv_path = root / "labels" / csv_name / "labels.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            image_name = str(row["filename"]).strip()
            image_path = root / "raw" / folder_name / image_name
            if not image_path.exists():
                continue
            m = re.match(r"^P(\d+)_", image_name)
            patient_id = int(m.group(1)) if m else None
            records.append({
                "image_path": str(image_path),
                "label": int(row["label"]),
                "source": csv_name,
                "filename": image_name,
                "patient_id": patient_id,
            })
    return records


def split_records_by_patient(
    records: List[Dict[str, object]], heldout_patient_id: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:

    train_side = [r for r in records if r["patient_id"] != heldout_patient_id]
    val_side = [r for r in records if r["patient_id"] == heldout_patient_id]
    return train_side, val_side


def compute_branch_probs(model: nn.Module, image_path: str, device: torch.device) -> Dict[str, float]:

    model.eval()
    raw_rgb = read_image_rgb(image_path)
    proj_tensor = extract_1d_projection_resampled(raw_rgb, target_length=DEFAULT_PROJ_LENGTH).unsqueeze(0).to(device)
    dpi_aligned_rgb = resize_to_standard_dpi(raw_rgb, target_size=DEFAULT_IMAGE_SIZE)
    image_tensor = get_val_transforms()(image=dpi_aligned_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        img_feats = model.image_stream(image_tensor)
        proj_feats = model.proj_stream(proj_tensor)

        proj_zero = torch.zeros_like(proj_feats)
        prob_img = torch.sigmoid(model.classifier(torch.cat((img_feats, proj_zero), dim=1))).item()

        img_zero = torch.zeros_like(img_feats)
        prob_proj = torch.sigmoid(model.classifier(torch.cat((img_zero, proj_feats), dim=1))).item()

        prob_fusion = torch.sigmoid(model.classifier(torch.cat((img_feats, proj_feats), dim=1))).item()

    return {"image_only": float(prob_img), "proj_only": float(prob_proj), "fusion": float(prob_fusion)}


def augment_and_create_sample(item: Dict[str, object], device: torch.device, seed_offset: int) -> Dict[str, object]:

    raw_rgb = read_image_rgb(str(item["image_path"]))
    proj_tensor = extract_1d_projection_resampled(raw_rgb, target_length=DEFAULT_PROJ_LENGTH)

    rng = torch.Generator()
    rng.manual_seed(42 + seed_offset)
    noise = torch.randn(proj_tensor.shape, generator=rng) * 0.01
    proj_tensor = torch.clamp(proj_tensor + noise, 0.0, 1.0)

    dpi_aligned_rgb = resize_to_standard_dpi(raw_rgb, target_size=DEFAULT_IMAGE_SIZE)
    image_tensor = get_hard_augmentation()(image=dpi_aligned_rgb)["image"]

    return {
        "image_tensor": image_tensor,
        "proj_tensor": proj_tensor,
        "label": int(item["label"]),
        "filename": f"aug_{item['filename']}",
    }


def prepare_balanced_hard_dataset(
    model: nn.Module,
    records: List[Dict[str, object]],
    device: torch.device,
    target_pos: int = 10,
    target_neg: int = 10,
) -> Tuple[List[Dict[str, object]], int, int]:

    scored_records: List[Dict[str, object]] = []
    for r in records:
        probs = compute_branch_probs(model, str(r["image_path"]), device)
        scored_records.append({
            **r,
            "img_prob": float(probs["image_only"]),
            "proj_prob": float(probs["proj_only"]),
            "fusion": float(probs["fusion"]),
            "proj_bias": float(probs["proj_only"] - probs["image_only"]),
        })

    pos_hard_candidates = [
        x for x in scored_records
        if x["label"] == 1 and x["proj_bias"] > 0.05 and x["proj_prob"] >= 0.25
    ]
    pos_hard_candidates.sort(key=lambda x: (round(x["proj_bias"], 4), x["filename"]), reverse=True)

    neg_hard_candidates = [
        x for x in scored_records
        if x["label"] == 0 and (0.15 <= x["proj_prob"] <= 0.50)
    ]
    neg_hard_candidates.sort(key=lambda x: (round(abs(x["proj_prob"] - 0.35), 4), x["filename"]))

    if not pos_hard_candidates:
        all_pos = [x for x in scored_records if x["label"] == 1]
        all_pos.sort(key=lambda x: (round(x["proj_bias"], 4), x["filename"]), reverse=True)
        pos_hard_candidates = all_pos[:target_pos]
    if not neg_hard_candidates:
        all_neg = [x for x in scored_records if x["label"] == 0]
        all_neg.sort(key=lambda x: (round(abs(x["proj_prob"] - 0.35), 4), x["filename"]))
        neg_hard_candidates = all_neg[:target_neg]

    raw_pos_count = len(pos_hard_candidates)
    raw_neg_count = len(neg_hard_candidates)

    final_items: List[Dict[str, object]] = []

    for item in pos_hard_candidates[:target_pos]:
        raw_rgb = read_image_rgb(str(item["image_path"]))
        proj_tensor = extract_1d_projection_resampled(raw_rgb, target_length=DEFAULT_PROJ_LENGTH)
        dpi_rgb = resize_to_standard_dpi(raw_rgb, target_size=DEFAULT_IMAGE_SIZE)
        img_t = get_val_transforms()(image=dpi_rgb)["image"]
        final_items.append({"image_tensor": img_t, "proj_tensor": proj_tensor, "label": 1, "filename": item["filename"]})

    idx = 0
    while len([x for x in final_items if x["label"] == 1]) < target_pos:
        base_item = pos_hard_candidates[idx % len(pos_hard_candidates)]
        final_items.append(augment_and_create_sample(base_item, device, seed_offset=idx))
        idx += 1

    for item in neg_hard_candidates[:target_neg]:
        raw_rgb = read_image_rgb(str(item["image_path"]))
        proj_tensor = extract_1d_projection_resampled(raw_rgb, target_length=DEFAULT_PROJ_LENGTH)
        dpi_rgb = resize_to_standard_dpi(raw_rgb, target_size=DEFAULT_IMAGE_SIZE)
        img_t = get_val_transforms()(image=dpi_rgb)["image"]
        final_items.append({"image_tensor": img_t, "proj_tensor": proj_tensor, "label": 0, "filename": item["filename"]})

    idx = 0
    while len([x for x in final_items if x["label"] == 0]) < target_neg:
        base_item = neg_hard_candidates[idx % len(neg_hard_candidates)]
        final_items.append(augment_and_create_sample(base_item, device, seed_offset=100 + idx))
        idx += 1

    return final_items, raw_pos_count, raw_neg_count


def build_val_loader(
    val_records: List[Dict[str, object]], data_cfg: dict, batch_size: int = 8
) -> DataLoader:

    dataset = TestStripDataset(
        image_paths=[r["image_path"] for r in val_records],
        labels=[r["label"] for r in val_records],
        transforms=get_val_transforms(),
        projection_transforms=get_projection_transforms(),
        standard_dpi_size=tuple(data_cfg["standard_dpi_size"]),
        proj_length=int(data_cfg["proj_length"]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_validation(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Dict[str, float]:

    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, projs, labels in val_loader:
            imgs, projs = imgs.to(device), projs.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(imgs, projs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy().flatten())

    y_true = np.asarray(all_labels, dtype=int)
    y_prob = np.asarray(all_probs, dtype=np.float32)
    avg_loss = total_loss / max(len(val_loader), 1)

    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    else:
        auc = float("nan")

    preds = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = accuracy_score(y_true, preds)

    return {
        "loss": float(avg_loss),
        "auc": auc,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(acc),
    }


def build_test_loader(
    test_csv: str, data_cfg: dict, batch_size: int = 8, image_dir: str = "",
    projection_aug: bool = True,
) -> Tuple[DataLoader, np.ndarray]:

    df = pd.read_csv(test_csv)

    if image_dir:
        test_dir = Path(image_dir)
    else:
        same_level = Path(test_csv).parent / "raw"
        parent_level = Path(test_csv).parent.parent / "raw"
        test_dir = same_level if same_level.exists() else (parent_level if parent_level.exists() else Path(test_csv).parent)

    paths = []
    for f in df["filename"]:
        f = str(f)
        if Path(f).is_absolute() and Path(f).exists():
            paths.append(f)
        else:
            paths.append(str(test_dir / f))
    dataset = TestStripDataset(
        image_paths=paths,
        labels=df["label"].tolist(),
        transforms=get_val_transforms(),
        projection_transforms=get_projection_transforms() if projection_aug else False,
        standard_dpi_size=tuple(data_cfg["standard_dpi_size"]),
        proj_length=int(data_cfg["proj_length"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    labels = df["label"].astype(int).to_numpy()
    return loader, labels


def predict_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:

    model.eval()
    probs = []
    with torch.no_grad():
        for imgs, projs, _ in loader:
            imgs, projs = imgs.to(device), projs.to(device)
            logits = model(imgs, projs)
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())
    return np.asarray(probs, dtype=np.float32)


def compute_metrics_with_best_threshold(labels, probs) -> dict:

    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=np.float32)

    if len(np.unique(labels)) < 2:
        auc = np.nan
        best_threshold = 0.5
    else:
        auc = float(roc_auc_score(labels, probs))
        fpr, tpr, thresholds = roc_curve(labels, probs)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_threshold = float(thresholds[best_idx])
        if best_threshold > 1.0:
            best_threshold = float(np.max(probs))

    def _calc(th: float):
        preds = (probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        acc = accuracy_score(labels, preds)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return acc, sens, spec

    acc_b, sens_b, spec_b = _calc(best_threshold)
    acc_5, sens_5, spec_5 = _calc(0.5)
    return {
        "auc": float(auc),
        "best_threshold": float(best_threshold),
        "accuracy": float(acc_b),
        "sensitivity": float(sens_b),
        "specificity": float(spec_b),
        "acc_05": float(acc_5),
        "sens_05": float(sens_5),
        "spec_05": float(spec_5),
    }
