
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from . import config

_COLORMAPS = {
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "MAGMA": cv2.COLORMAP_MAGMA,
    "HOT": cv2.COLORMAP_HOT,
}


def find_last_conv(model: torch.nn.Module) -> torch.nn.Module:

    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("模型中未找到 Conv2d 层, 无法生成 Grad-CAM。")
    return last


def find_last_conv_name(model: torch.nn.Module) -> str:

    last_name = ""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
    return last_name


class StripGradCAM:


    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):

        self.model = model
        self.target_layer = target_layer or find_last_conv(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):

        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):

        self.gradients = grad_output[0]

    def generate(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor
                 ) -> Tuple[np.ndarray, float]:

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(img_tensor, proj_tensor)
        score = logits.reshape(-1)[0]
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("未捕获到激活/梯度, Grad-CAM 生成失败。")

        gradients = self.gradients.detach()
        activations = self.activations.detach()
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * activations, dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=(img_tensor.shape[2], img_tensor.shape[3]),
            mode="bilinear", align_corners=False,
        ).squeeze().detach().cpu().numpy()

        lo, hi = float(np.min(cam)), float(np.max(cam))
        cam = (cam - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(cam)
        prob = float(torch.sigmoid(logits).item())
        self.model.zero_grad(set_to_none=True)
        return cam, prob


def overlay_heatmap(image_bgr: np.ndarray, cam: np.ndarray,
                    alpha: float = None, colormap: str = None) -> np.ndarray:

    alpha = float(config.GRADCAM_ALPHA if alpha is None else alpha)
    cmap = _COLORMAPS.get((colormap or config.GRADCAM_COLORMAP).upper(), cv2.COLORMAP_JET)

    h, w = cam.shape[:2]
    base = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA if image_bgr.shape[1] > w
                      else cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cmap)
    return cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0.0)


def save_heatmap(image_bgr: np.ndarray, cam: np.ndarray, out_path: Path,
                 title: str = "", alpha: float = None) -> Path:

    overlay = overlay_heatmap(image_bgr, cam, alpha=alpha)
    if title:
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(overlay, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)
    return out_path
