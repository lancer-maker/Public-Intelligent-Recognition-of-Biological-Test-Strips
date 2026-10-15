import cv2
import numpy as np
import torch
from typing import Tuple, Optional

try:
    from scipy.signal import detrend
except ImportError:
    detrend = None


def resize_to_standard_dpi(image_rgb: np.ndarray, target_size: Tuple[int, int] = (220, 505)) -> np.ndarray:

    if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
        raise ValueError(f"期望输入为 (H, W, 3) 的 RGB 图像，实际形状为 {image_rgb.shape}")

    h, w = image_rgb.shape[:2]
    target_w, target_h = target_size
    if (w, h) != (target_w, target_h):
        interpolation = cv2.INTER_CUBIC if w < target_w else cv2.INTER_AREA
        return cv2.resize(image_rgb, (target_w, target_h), interpolation=interpolation)
    return image_rgb


def read_image_rgb(image_path: str, target_size: Optional[Tuple[int, int]] = (220, 505)) -> np.ndarray:

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if target_size is not None:
        img_rgb = resize_to_standard_dpi(img_rgb, target_size=target_size)

    return img_rgb


def normalize_projection_baseline(signal: np.ndarray) -> np.ndarray:

    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal

    centered = signal - np.mean(signal)

    if detrend is not None:
        try:
            centered = detrend(centered, type='linear', overwrite=False)
        except TypeError:
            centered = detrend(centered, type='linear')

    return centered


def extract_1d_projection_resampled(image_rgb_raw: np.ndarray, target_length: int = 512) -> torch.Tensor:

    gray = cv2.cvtColor(image_rgb_raw, cv2.COLOR_RGB2GRAY)
    
    raw_proj = np.mean(gray, axis=1, dtype=np.float32)
    raw_length = len(raw_proj)
    
    raw_proj_norm = (raw_proj - raw_proj.min()) / (raw_proj.max() - raw_proj.min() + 1e-6)
    
    if raw_length != target_length:
        x_raw = np.linspace(0, 5.05, num=raw_length)
        x_target = np.linspace(0, 5.05, num=target_length)
        resampled_proj = np.interp(x_target, x_raw, raw_proj_norm)
    else:
        resampled_proj = raw_proj_norm

    resampled_proj = normalize_projection_baseline(resampled_proj)

    proj_tensor = torch.from_numpy(resampled_proj).float().unsqueeze(0)
    
    return proj_tensor


def extract_col_avg_projection_resampled(
    image_rgb_raw: np.ndarray,
    target_length: int = 512,
    roi_fraction: float = 0.33,
) -> torch.Tensor:

    gray = cv2.cvtColor(image_rgb_raw, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    roi_h = max(1, int(round(h * roi_fraction)))
    start_y = (h - roi_h) // 2
    roi = gray[start_y:start_y + roi_h, :]

    raw_proj = np.mean(roi, axis=0, dtype=np.float32)
    raw_length = len(raw_proj)

    raw_proj_norm = (raw_proj - raw_proj.min()) / (raw_proj.max() - raw_proj.min() + 1e-6)

    if raw_length != target_length:
        x_raw = np.linspace(0, 2.2, num=raw_length)
        x_target = np.linspace(0, 2.2, num=target_length)
        resampled_proj = np.interp(x_target, x_raw, raw_proj_norm)
    else:
        resampled_proj = raw_proj_norm

    resampled_proj = normalize_projection_baseline(resampled_proj)

    return torch.from_numpy(resampled_proj).float().unsqueeze(0)