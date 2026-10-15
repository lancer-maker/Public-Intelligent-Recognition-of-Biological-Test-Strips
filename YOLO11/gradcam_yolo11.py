

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

YOLO11_DIR = Path(__file__).resolve().parent
if str(YOLO11_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO11_DIR))

from yolo11_common import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEEDS,
    dump_json,
    ensure_output_dirs,
    list_split_samples,
    load_val_transform,
    resolve_positive_index,
    set_seed,
)
from test_yolo11_cls import discover_seed_runs, resolve_tag

HEATMAP_DIR = OUTPUT_ROOT / "heatmaps"


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="YOLO11 分类模型单图 Grad-CAM 热力图 (输出到 outputs/YOLO11/heatmaps)"
    )
    parser.add_argument("--image", type=str, required=True,
                        help="单张图片路径 (绝对路径, 或相对项目根目录的路径)")
    parser.add_argument("--tag", type=str, default=None,
                        help="训练分组名 (如 yolo11n-cls_224); 缺省自动选择最近训练的分组")
    parser.add_argument("--seed", type=int, nargs="+", default=None,
                        help="参与可视化的种子 (可传多个); 缺省=该 tag 下全部已训练种子; "
                             "多于 1 个种子时额外输出等权平均 CAM")
    parser.add_argument("--weights", type=str, default=None,
                        help="直接指定权重路径 (优先于 --tag/--seed, 不依赖 runs 目录)")
    parser.add_argument("--imgsz", type=int, default=224,
                        help="网络输入尺寸, 必须与训练时一致 (默认 224)")
    parser.add_argument("--class-index", type=str, default="auto",
                        help="被解释的类别: auto=该模型预测类别(默认); 也可填 0(阴性) 或 1(阳性)")
    parser.add_argument("--target-layer", type=str, default="auto",
                        help="Grad-CAM 目标层: auto=分类头内的 1x1 卷积(默认); "
                             "或填 top-level 索引 (如 9 / 8; 语义更浅, 定位可能更精细)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="判定阈值; 缺省自动读 reports/eval_config_<tag>.json, 找不到回退 0.5")
    parser.add_argument("--device", type=str, default="0", help="设备: 0 / cpu")
    parser.add_argument("--alpha", type=float, default=0.45,
                        help="叠加图中热力图权重 0~1 (默认 0.45, 原图权重 1-alpha)")
    parser.add_argument("--dpi", type=int, default=300, help="保存图片 dpi")
    parser.add_argument("--out-dir", type=str, default=str(HEATMAP_DIR),
                        help=f"输出目录 (默认 {HEATMAP_DIR})")
    return parser.parse_args()


def to_project_path(raw: str) -> Path:

    path = Path(raw)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def resolve_device(device: str) -> torch.device:

    if device in ("", None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit() or str(device) == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(device))


def resolve_ground_truth(image_path: Path) -> Tuple[Optional[int], Optional[str]]:

    name = Path(image_path).name
    for split in ("test", "val", "train"):
        try:
            for record in list_split_samples(split):
                if record.filename == name:
                    return int(record.label), split
        except Exception:
            continue
    return None, None


def resolve_decision_threshold(tag: str, seed: Optional[int],
                               user_threshold: Optional[float]) -> Tuple[float, str]:

    if user_threshold is not None:
        return float(user_threshold), "--threshold"
    config_path = REPORTS_DIR / f"eval_config_{tag}.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if seed is not None:
                value = config.get("val_thresholds", {}).get(str(seed))
                if value is not None:
                    return float(value), f"eval_config.val_thresholds[{seed}]"
            value = config.get("ensemble_threshold")
            if value is not None:
                return float(value), "eval_config.ensemble_threshold"
        except Exception as exc:
            print(f"[warn] 读取 {config_path.name} 失败 ({exc}), 回退 0.5")
    return 0.5, "fallback 0.5"


def infer_tag_from_weights(weight_path: Path) -> str:

    if weight_path.parent.name == "weights" and weight_path.parent.parent.name.startswith("seed_"):
        return weight_path.parent.parent.parent.name
    return weight_path.parent.name


def load_weights_list(args: argparse.Namespace) -> Tuple[str, List[Tuple[Optional[int], Path]]]:

    if args.weights:
        weight_path = to_project_path(args.weights).resolve()
        if not weight_path.exists():
            raise FileNotFoundError(f"--weights 指定的权重不存在: {weight_path}")
        return infer_tag_from_weights(weight_path), [(None, weight_path)]

    tag = resolve_tag(args.tag)
    seed_runs = discover_seed_runs(tag, args.seed)
    return tag, [(seed, path) for seed, path in seed_runs]


def preprocess_image(image_path: Path, imgsz: int) -> Tuple[torch.Tensor, np.ndarray]:

    from PIL import Image
    from torchvision.transforms import CenterCrop, InterpolationMode, Resize

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    pil_image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    tensor = load_val_transform(imgsz)(pil_image).unsqueeze(0)
    resized = Resize(imgsz, interpolation=InterpolationMode.BILINEAR)(pil_image)
    display = np.asarray(CenterCrop(imgsz)(resized), dtype=np.uint8)
    return tensor, display


def locate_target_layer(net: torch.nn.Module, spec: str) -> Tuple[torch.nn.Module, str]:

    if spec.strip().lower() in ("auto", "head", "head.conv", "conv"):
        head = net.model[-1]
        inner = getattr(head, "conv", None)
        if inner is None:
            raise ValueError(
                "未能自动定位分类头内的卷积层, 请用 --target-layer 手动指定 top-level 索引 (如 9)"
            )
        return inner, "net.model[-1].conv (Classify 头内 1x1 卷积)"

    try:
        index = int(spec)
    except ValueError:
        raise ValueError(f"无法解析 --target-layer='{spec}', 应为 'auto' 或整数索引")

    try:
        layer = net.model[index]
    except (IndexError, KeyError, TypeError):
        raise ValueError(f"--target-layer={index} 越界, 可用范围 0~{len(net.model) - 1}")
    return layer, f"net.model[{index}] ({type(layer).__name__})"


def probe_layer_output(net: torch.nn.Module, layer: torch.nn.Module, imgsz: int,
                       device: torch.device) -> Optional[Tuple[int, ...]]:

    holder: Dict[str, Optional[Tuple[int, ...]]] = {"shape": None}

    def _capture(module, inputs, output) -> None:
        holder["shape"] = tuple(output.shape) if isinstance(output, torch.Tensor) else None

    handle = layer.register_forward_hook(_capture)
    try:
        with torch.no_grad():
            net(torch.zeros(1, 3, imgsz, imgsz, device=device))
    except Exception as exc:
        print(f"[warn] 目标层形状探测失败 ({exc}), 继续执行")
        return None
    finally:
        handle.remove()
    return holder["shape"]


def infer_probs(net: torch.nn.Module, img_tensor: torch.Tensor) -> np.ndarray:

    with torch.no_grad():
        output = net(img_tensor)
    logits = output[0] if isinstance(output, (list, tuple)) else output
    logits = logits.float().reshape(logits.shape[0], -1)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def parse_class_index(spec: str, num_classes: int, pred_index: int) -> int:

    if spec.strip().lower() == "auto":
        return int(pred_index)
    try:
        index = int(spec)
    except ValueError:
        raise ValueError(f"无法解析 --class-index='{spec}', 应为 'auto' 或 0/1")
    if index < 0 or index >= num_classes:
        raise ValueError(f"--class-index={index} 超出模型类别数 {num_classes}")
    return index


class YoloGradCAM:


    def __init__(self, net: torch.nn.Module, target_layer: torch.nn.Module) -> None:

        self.net = net
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:

        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output) -> None:

        self.gradients = grad_output[0]

    def generate(self, img_tensor: torch.Tensor, class_index: int) -> np.ndarray:

        self.net.eval()
        self.net.zero_grad(set_to_none=True)

        output = self.net(img_tensor)
        logits = output[0] if isinstance(output, (list, tuple)) else output
        logits = logits.float().reshape(logits.shape[0], -1)
        if not logits.requires_grad:
            raise RuntimeError(
                "logits 不含梯度图, 无法执行 Grad-CAM 反向传播。"
                "Ultralytics 载入权重时会把参数设为 requires_grad=False, "
                "请在使用本类前对模型执行 net.requires_grad_(True)"
            )

        logits[0, int(class_index)].backward()

        gradients = self.gradients.detach().float()
        activations = self.activations.detach().float()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))

        cam = torch.nn.functional.interpolate(
            cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max - cam_min > 1e-8 \
            else np.zeros_like(cam)

        self.activations, self.gradients = None, None
        return cam.astype(np.float32)


def make_overlay(raw_img: np.ndarray, cam_map: np.ndarray,
                 alpha: float) -> Tuple[np.ndarray, np.ndarray]:

    cam_uint8 = np.uint8(np.clip(cam_map, 0.0, 1.0) * 255)
    heatmap = cv2.cvtColor(cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.uint8(raw_img * (1.0 - float(alpha)) + heatmap * float(alpha))
    return heatmap, overlay


def save_triple_panel(raw_img: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray,
                      save_path: Path, title: str, true_label_text: str,
                      prob: float, threshold: float, dpi: int) -> None:

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Original Image\n(True Label: {true_label_text})", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Overlay (Attention Area)\nPred Prob: {prob:.4f} (thr={threshold:.4f})", fontsize=12
    )
    axes[2].axis("off")
    fig.suptitle(title, fontsize=13)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def normalize_cam(cam_map: np.ndarray) -> np.ndarray:

    cam_min, cam_max = float(cam_map.min()), float(cam_map.max())
    if cam_max - cam_min > 1e-8:
        return ((cam_map - cam_min) / (cam_max - cam_min)).astype(np.float32)
    return np.zeros_like(cam_map, dtype=np.float32)


def class_display_name(names, index: int) -> str:

    if isinstance(names, dict):
        return str(names.get(index, index))
    try:
        return str(names[index])
    except Exception:
        return str(index)


def main() -> None:

    args = parse_args()
    ensure_output_dirs()

    image_path = to_project_path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片: {image_path}")
    out_dir = to_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tag, weights_list = load_weights_list(args)
    set_seed(SEEDS[0])

    true_label, true_split = resolve_ground_truth(image_path)
    true_label_text = str(true_label) if true_label is not None else "NA"
    print("=" * 78)
    print(f"YOLO11 Grad-CAM | 图片: {image_path}")
    print(f"  tag={tag} | imgsz={args.imgsz} | device={device} | 权重数={len(weights_list)}")
    print(f"  真实标签: {true_label_text}"
          + (f" (来自 {true_split} 划分的文件夹名)" if true_split else " (未在数据集中匹配到)"))
    print("=" * 78)

    img_tensor, raw_display = preprocess_image(image_path, args.imgsz)
    img_input = img_tensor.to(device)

    from ultralytics import YOLO

    cams: List[np.ndarray] = []
    probs_pos: List[float] = []
    members: List[Dict[str, object]] = []
    target_desc = ""

    for seed, weight_path in weights_list:
        print(f"\n[权重] seed={seed if seed is not None else 'NA'} -> {weight_path}")
        model = YOLO(str(weight_path))
        net = model.model.to(device).eval()
        net.requires_grad_(True)
        positive_index = resolve_positive_index(model.names)

        target_layer, layer_desc = locate_target_layer(net, args.target_layer)
        shape = probe_layer_output(net, target_layer, args.imgsz, device)
        if shape is not None and len(shape) != 4:
            raise ValueError(
                f"目标层输出形状 {shape} 不是 4D 特征图 (B,C,H,W); "
                f"请改用 --target-layer auto 或指定卷积层 (如 9 / 8)"
            )
        target_desc = layer_desc
        print(f"  目标层: {layer_desc}" + (f", 输出特征图 {shape}" if shape else ""))

        probs = infer_probs(net, img_input)
        pred_index = int(np.argmax(probs))
        class_index = parse_class_index(args.class_index, len(probs), pred_index)
        prob_pos = float(probs[positive_index])

        grad_cam = YoloGradCAM(net, target_layer)
        cam_map = grad_cam.generate(img_input, class_index)

        threshold, threshold_source = resolve_decision_threshold(tag, seed, args.threshold)
        pred_tag = "POS" if prob_pos >= threshold else "NEG"
        seed_tag = f"seed{seed}" if seed is not None else weight_path.parent.parent.name
        class_name = class_display_name(model.names, class_index)

        heatmap, overlay = make_overlay(raw_display, cam_map, args.alpha)
        save_path = out_dir / (
            f"cam_yolo11_{tag}_{seed_tag}_{image_path.stem}"
            f"_true_{true_label_text}_pred_{pred_tag}_{prob_pos:.2f}.png"
        )
        save_triple_panel(
            raw_display, heatmap, overlay, save_path,
            title=f"YOLO11-cls {tag} | {seed_tag} | explained class: {class_name} (index {class_index})",
            true_label_text=true_label_text, prob=prob_pos, threshold=threshold, dpi=args.dpi,
        )
        print(f"  阳性概率: {prob_pos:.4f} | 判定: {pred_tag} "
              f"(thr={threshold:.4f}, 来源 {threshold_source})")
        print(f"  被解释类别: {class_name} (index {class_index}) | 模型预测类别 index {pred_index}")
        print(f"  已保存: {save_path}")

        cams.append(normalize_cam(cam_map))
        probs_pos.append(prob_pos)
        members.append({
            "seed": seed, "weights": str(weight_path), "target_layer": layer_desc,
            "class_index": int(class_index), "class_name": class_name,
            "prob_positive": prob_pos, "threshold": float(threshold),
            "threshold_source": threshold_source, "pred_tag": pred_tag,
            "figure": str(save_path),
        })

    if len(cams) > 1:
        mean_cam = normalize_cam(np.mean(np.stack(cams, axis=0), axis=0).astype(np.float32))
        prob_pos_ens = float(np.mean(probs_pos))
        threshold_ens, source_ens = resolve_decision_threshold(tag, None, args.threshold)
        pred_tag_ens = "POS" if prob_pos_ens >= threshold_ens else "NEG"

        heatmap, overlay = make_overlay(raw_display, mean_cam, args.alpha)
        save_path = out_dir / (
            f"cam_yolo11_{tag}_ensemble{len(cams)}_{image_path.stem}"
            f"_true_{true_label_text}_pred_{pred_tag_ens}_{prob_pos_ens:.2f}.png"
        )
        save_triple_panel(
            raw_display, heatmap, overlay, save_path,
            title=f"YOLO11-cls {tag} | ensemble of {len(cams)} seeds (equal-weight mean CAM)",
            true_label_text=true_label_text, prob=prob_pos_ens,
            threshold=threshold_ens, dpi=args.dpi,
        )
        print(f"\n[集成] {len(cams)} 个种子等权平均 CAM")
        print(f"  集成阳性概率: {prob_pos_ens:.4f} | 判定: {pred_tag_ens} "
              f"(thr={threshold_ens:.4f}, 来源 {source_ens})")
        print(f"  已保存: {save_path}")
        members.append({
            "seed": "ensemble", "weights": "equal-weight mean of members",
            "target_layer": target_desc, "class_index": None, "class_name": "mean-CAM",
            "prob_positive": prob_pos_ens, "threshold": float(threshold_ens),
            "threshold_source": source_ens, "pred_tag": pred_tag_ens,
            "figure": str(save_path),
        })

    record_path = REPORTS_DIR / f"gradcam_{image_path.stem}.json"
    dump_json(record_path, {
        "image": str(image_path),
        "tag": tag,
        "imgsz": args.imgsz,
        "device": str(device),
        "target_layer": target_desc,
        "class_index_arg": args.class_index,
        "alpha": args.alpha,
        "true_label": true_label,
        "true_split": true_split,
        "members": members,
    })
    print(f"\n本次可视化记录: {record_path}")


if __name__ == "__main__":
    main()
