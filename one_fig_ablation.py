
import os
import sys
import yaml
import torch
import cv2
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.utils import (
    read_image_rgb,
    extract_1d_projection_resampled,
    extract_col_avg_projection_resampled,
)
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet

IMAGE_PATH = r"data\test\raw\N_023_0.jpg"
MODEL_WEIGHTS = (
    r"outputs\checkpoints\Tweak\1\best_model.pth,"
    r"outputs\checkpoints\Tweak\2\best_model.pth,"
    r"outputs\checkpoints\Tweak\3\best_model.pth,"
    r"outputs\checkpoints\Tweak\4\best_model.pth,"
    r"outputs\checkpoints\Tweak\5\best_model.pth"
)
CONFIG_PATH = "configs/main_config.yaml"
REPORT_CSV = r"outputs\reports\one_fig_ablation.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config(CONFIG_PATH)
    model_cfg = cfg['model']
    data_cfg = cfg['data']

    raw_rgb = read_image_rgb(IMAGE_PATH)

    use_col_avg = bool(data_cfg.get("proj_use_col_avg", False))
    proj_tensor = extract_1d_projection_resampled(raw_rgb, target_length=data_cfg['proj_length'])
    if use_col_avg:
        col_tensor = extract_col_avg_projection_resampled(
            raw_rgb,
            target_length=int(data_cfg.get("proj_col_length", 512)),
            roi_fraction=float(data_cfg.get("proj_col_roi_fraction", 0.33)),
        )
        proj_tensor = torch.cat([proj_tensor, col_tensor], dim=0)
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

    weight_paths = [p.strip() for p in str(MODEL_WEIGHTS).split(",") if p.strip()]
    if not weight_paths:
        print("[错误] 未提供有效模型权重路径。")
        return

    print("=" * 80)
    print(f"图像: {IMAGE_PATH}")
    results = []
    for i, wp in enumerate(weight_paths, 1):
        model = DualStreamStripNet(
            pretrained_2d=model_cfg['pretrained'],
            feature_dim=model_cfg['feature_dim'],
            dropout_rate=model_cfg['dropout_rate']
        )
        checkpoint = torch.load(wp, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        with torch.no_grad():
            img_feats = model.image_stream(image_tensor)
            proj_feats = model.proj_stream(proj_tensor)

            logits_normal = model.classifier(torch.cat((img_feats, proj_feats), dim=1))
            prob_normal = torch.sigmoid(logits_normal).item()

            img_zero = torch.zeros_like(img_feats)
            logits_only_proj = model.classifier(torch.cat((img_zero, proj_feats), dim=1))
            prob_only_proj = torch.sigmoid(logits_only_proj).item()

            proj_zero = torch.zeros_like(proj_feats)
            logits_only_img = model.classifier(torch.cat((img_feats, proj_zero), dim=1))
            prob_only_img = torch.sigmoid(logits_only_img).item()

        results.append({
            "model_id": i,
            "weight_path": wp,
            "normal_prob": float(prob_normal),
            "only_proj_prob": float(prob_only_proj),
            "only_img_prob": float(prob_only_img),
        })
        print(f"  [{i}/{len(weight_paths)}] 模型权重: {Path(wp).parent.name}/{Path(wp).name}")

    print("\n消融汇总 (正常 / 仅投影 / 仅图像 概率):")
    print("-" * 80)
    for r in results:
        print(f"  模型 {r['model_id']}: 正常={r['normal_prob']:.4f} | "
              f"仅投影={r['only_proj_prob']:.4f} | 仅图像={r['only_img_prob']:.4f}")
    print("=" * 80)

    if REPORT_CSV:
        out = Path(REPORT_CSV)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out, index=False)
        print(f"消融汇总报告已保存: {out}")

if __name__ == "__main__":
    main()