
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.utils import (
    read_image_rgb,
    extract_1d_projection_resampled,
    extract_col_avg_projection_resampled,
)
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

IMAGE_PATH = r"data/test/raw/N_048_1.png"
MODEL_WEIGHTS = r"outputs/checkpoints/E/3/best_model.pth"
CONFIG_PATH = "configs/main_config.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg['model']
    data_cfg = cfg['data']

    model = DualStreamStripNet(
        pretrained_2d=model_cfg['pretrained'],
        feature_dim=model_cfg['feature_dim'],
        dropout_rate=model_cfg['dropout_rate']
    )
    load_model_state(model, MODEL_WEIGHTS)
    model.to(DEVICE)
    model.eval()

    raw_rgb = read_image_rgb(IMAGE_PATH)

    use_col_avg = bool(data_cfg.get("proj_use_col_avg", False))
    proj_tensor = extract_1d_projection_resampled(
        raw_rgb, target_length=data_cfg['proj_length']
    )
    if use_col_avg:
        col_tensor = extract_col_avg_projection_resampled(
            raw_rgb,
            target_length=int(data_cfg.get("proj_col_length", 512)),
            roi_fraction=float(data_cfg.get("proj_col_roi_fraction", 0.33)),
        )
        proj_tensor = torch.cat([proj_tensor, col_tensor], dim=0)

    target_w, target_h = data_cfg['standard_dpi_size']
    h, w = raw_rgb.shape[:2]
    if (w, h) != (target_w, target_h):
        interp = cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA
        dpi_aligned_rgb = cv2.resize(raw_rgb, (target_w, target_h), interpolation=interp)
    else:
        dpi_aligned_rgb = raw_rgb

    val_transforms = get_val_transforms()
    augmented = val_transforms(image=dpi_aligned_rgb)
    image_tensor = augmented['image']

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    proj_tensor = proj_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_features = model.image_stream(image_tensor)

        proj_features = model.proj_stream(proj_tensor)

    img_vec = img_features.squeeze(0).cpu().numpy()
    proj_vec = proj_features.squeeze(0).cpu().numpy()

    torch.set_printoptions(precision=6, sci_mode=False, linewidth=120)
    print("=" * 80)
    print("图像流特征 (ImageStream 128-d):")
    print(torch.from_numpy(img_vec))
    print("\n投影流特征 (ProjStream 128-d):")
    print(torch.from_numpy(proj_vec))
    print("=" * 80)

if __name__ == "__main__":
    main()