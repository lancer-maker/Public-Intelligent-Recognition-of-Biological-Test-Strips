
import os
import sys
import yaml
import torch
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.utils import read_image_rgb, extract_1d_projection_resampled
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.utils.model_helper import load_model_state

IMAGE_PATH = "data/test/raw/N_026_1.jpg"
MODEL_WEIGHTS = "outputs/checkpoints/P24/best_model.pth"
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

    proj_tensor = extract_1d_projection_resampled(raw_rgb, target_length=data_cfg['proj_length'])
    proj_tensor = proj_tensor.unsqueeze(0).to(DEVICE)

    target_w, target_h = data_cfg['standard_dpi_size']
    h, w = raw_rgb.shape[:2]
    if (w, h) != (target_w, target_h):
        interp = cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA
        dpi_aligned_rgb = cv2.resize(raw_rgb, (target_w, target_h), interpolation=interp)
    else:
        dpi_aligned_rgb = raw_rgb
    val_transforms = get_val_transforms()
    augmented = val_transforms(image=dpi_aligned_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        img_feats = model.image_stream(image_tensor)
        proj_feats = model.proj_stream(proj_tensor)

        fused_normal = torch.cat((img_feats, proj_feats), dim=1)
        logits_normal = model.classifier(fused_normal)
        prob_normal = torch.sigmoid(logits_normal).item()

        img_zero = torch.zeros_like(img_feats)
        fused_only_proj = torch.cat((img_zero, proj_feats), dim=1)
        logits_only_proj = model.classifier(fused_only_proj)
        prob_only_proj = torch.sigmoid(logits_only_proj).item()

        proj_zero = torch.zeros_like(proj_feats)
        fused_only_img = torch.cat((img_feats, proj_zero), dim=1)
        logits_only_img = model.classifier(fused_only_img)
        prob_only_img = torch.sigmoid(logits_only_img).item()

    print("=" * 60)
    print(f"图像: {IMAGE_PATH}")
    print(f"正常预测概率: {prob_normal:.4f}")
    print(f"仅投影流概率: {prob_only_proj:.4f}")
    print(f"仅图像流概率: {prob_only_img:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()