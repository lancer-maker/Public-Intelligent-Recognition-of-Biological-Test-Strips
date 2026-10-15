
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset import TestStripDataset
from src.data.transforms import get_val_transforms
from src.models.dual_stream_net import DualStreamStripNet
from src.models.gated_fusion import GatedDualStreamNet
from src.utils.Fine_Tuning_helper import load_config
from src.utils.model_helper import load_model_state_dict

GATE_HIDDEN_FALLBACK = 128

TEST_LABELS_CSV = "data\\test\\labels\\labels.csv"
OUTPUT_DIR = "outputs\\heatmaps"


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, img_tensor, proj_tensor):
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(img_tensor, proj_tensor)
        score = logits.squeeze()

        score.backward(retain_graph=True)

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
            mode='bilinear', align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        prob = torch.sigmoid(logits).item()
        return cam, prob


def find_last_conv_layer(model):

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("未能自动在模型中找到 Conv2d 层，请手动指定 target_layer！")
    return last_conv


def parse_args():

    parser = argparse.ArgumentParser(
        description="单图 Grad-CAM: 输入单张图片 + 单个模型权重, 结构对齐 new_env_test.py"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="单张测试图片路径 (绝对或相对均可)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="单个模型权重路径 (裸 state_dict 或含 model_state_dict 的 ckpt 均可; 自动匹配范式)")
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "dual", "gated"],
                        help="模型范式: auto=按权重自动判定(默认); dual=直接拼接融合(Tweak/P 等); gated=E 模式门控融合")
    parser.add_argument("--label", type=int, default=None,
                        help="真实标签 0/1 (可选; 缺省则按文件名从测试集 CSV 自动匹配, 匹配不到显示 NA)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值 (缺省自动读取模型同目录 metrics.csv 的 best_threshold,\n"
                             "与 new_env_test.py 相同; 找不到则回退 0.5)")
    return parser.parse_args()


def resolve_label_from_csv(image_path: str):

    try:
        df = pd.read_csv(TEST_LABELS_CSV)
    except Exception:
        return None
    hit = df.loc[df["filename"] == Path(image_path).name, "label"]
    return int(hit.iloc[0]) if len(hit) else None


def resolve_threshold(weight_path: str, user_threshold: Optional[float]) -> float:

    if user_threshold is not None:
        return float(user_threshold)
    try:
        metrics_csv = Path(weight_path).parent / "metrics.csv"
        if metrics_csv.exists():
            row = pd.read_csv(metrics_csv).iloc[0]
            th = row.get("best_threshold")
            if th is not None and pd.notna(th):
                return float(th)
    except Exception:
        pass
    return 0.5


def build_single_sample(image_path: str, label, data_cfg: dict):

    ds = TestStripDataset(
        image_paths=[str(image_path)],
        labels=[0 if label is None else int(label)],
        transforms=get_val_transforms(),
        projection_transforms=False,
        standard_dpi_size=tuple(data_cfg["standard_dpi_size"]),
        proj_length=int(data_cfg["proj_length"]),
    )
    img_tensor, proj_tensor, _ = ds[0]
    return img_tensor, proj_tensor


def infer_gate_hidden(state_dict: dict, fallback: int = GATE_HIDDEN_FALLBACK) -> int:

    w = state_dict.get("gate_mlp.net.2.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[1])
    return int(fallback)


def detect_mode(state_dict: dict, mode: str = "auto") -> str:

    if mode in ("dual", "gated"):
        return mode
    return "gated" if any(k.startswith("gate_mlp.") for k in state_dict) else "dual"


def build_model_for_weights(weight_path: str, model_cfg: dict, device, mode: str = "auto"):

    state_dict = load_model_state_dict(str(weight_path))
    resolved = detect_mode(state_dict, mode)
    feature_dim = int(model_cfg["feature_dim"])
    dropout_rate = float(model_cfg["dropout_rate"])

    if resolved == "gated":
        gate_hidden = infer_gate_hidden(state_dict)
        model = GatedDualStreamNet(
            pretrained_2d=False,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
            gate_hidden=gate_hidden,
        )
        arch_desc = f"GatedDualStreamNet (E 模式门控融合, gate_hidden={gate_hidden})"
    else:
        model = DualStreamStripNet(
            pretrained_2d=False,
            feature_dim=feature_dim,
            dropout_rate=dropout_rate,
        )
        arch_desc = "DualStreamStripNet (直接拼接融合)"

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, arch_desc


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cfg = load_config()
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"正在加载模型: {args.model_path}")
    model, arch_desc = build_model_for_weights(
        args.model_path, model_cfg, device, mode=args.mode
    )
    print(f"模型范式: {arch_desc}")

    target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model, target_layer)

    label = args.label if args.label is not None else resolve_label_from_csv(args.image)
    label_text = str(label) if label is not None else "NA"
    threshold = resolve_threshold(args.model_path, args.threshold)
    print(f"图片: {args.image} | 真实标签: {label_text} | 设备: {device}")
    print(f"判定阈值: {threshold:.4f} (来源: {'--threshold' if args.threshold is not None else 'metrics.csv best_threshold(同new_env_test)'} 或回退0.5)")

    img_tensor, proj_tensor = build_single_sample(args.image, label, data_cfg)
    img_input = img_tensor.unsqueeze(0).to(device)
    proj_input = proj_tensor.unsqueeze(0).to(device)

    cam_map, pred_prob = grad_cam.generate(img_input, proj_input)

    raw_img = img_tensor.permute(1, 2, 0).cpu().numpy()
    raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
    raw_img = np.uint8(255 * raw_img)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * raw_img + 0.4 * heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Original Image\n(True Label: {label_text})", fontsize=12)
    axes[0].axis('off')
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis('off')
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (Attention Area)\nPred Prob: {pred_prob:.4f} (thr={threshold:.4f})", fontsize=12)
    axes[2].axis('off')

    base_name = Path(args.image).stem
    pred_tag = "POS" if pred_prob >= threshold else "NEG"
    save_filename = f"cam_000_{base_name}_true_{label_text}_pred_{pred_tag}_{pred_prob:.2f}.png"
    save_path = os.path.join(OUTPUT_DIR, save_filename)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {save_path}")


if __name__ == "__main__":
    main()