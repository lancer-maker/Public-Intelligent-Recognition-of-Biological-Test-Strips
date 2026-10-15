
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def amp_enabled(device: torch.device) -> bool:

    return device.type == "cuda"


def make_amp_scaler(device: torch.device) -> "torch.amp.GradScaler":

    return torch.amp.GradScaler("cuda", enabled=amp_enabled(device))


def amp_train_step(model, optimizer, scaler, criterion, imgs, projs, labels) -> float:

    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", enabled=scaler.is_enabled()):
        logits = model(imgs, projs)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int,
                    device: torch.device, **kwargs) -> DataLoader:

    kwargs.setdefault("pin_memory", device.type == "cuda")
    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, **kwargs)


def should_validate(epoch: int, val_interval: int = 1) -> bool:

    val_interval = max(1, int(val_interval))
    return epoch % val_interval == 0


def make_epoch_tqdm(iterable, **kwargs):

    kwargs.setdefault("mininterval", 1.0)
    return tqdm(iterable, **kwargs)


class ProjectionCache:


    def __init__(self, cache_dir: str = "", enabled: bool = True):
        self.enabled = bool(enabled)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._mem: dict = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(image_path: str, use_col_avg: bool, proj_length: int,
             col_roi_fraction: float, col_length: int) -> str:
        return (f"{Path(image_path).resolve()}|col={int(bool(use_col_avg))}"
                f"|pl={proj_length}|rf={col_roi_fraction}|cl={col_length}")

    def _disk_path(self, key: str) -> Path:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{hashlib.md5(key.encode('utf-8')).hexdigest()}.pt"

    def _compute(self, image_path: str, use_col_avg: bool, proj_length: int,
                 col_roi_fraction: float, col_length: int) -> torch.Tensor:

        import cv2
        from src.data.utils import (
            extract_1d_projection_resampled,
            extract_col_avg_projection_resampled,
        )
        raw_bgr = cv2.imread(str(image_path))
        if raw_bgr is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        row_t = extract_1d_projection_resampled(raw_rgb, target_length=proj_length)
        if use_col_avg:
            col_t = extract_col_avg_projection_resampled(
                raw_rgb, target_length=col_length, roi_fraction=col_roi_fraction
            )
            return torch.cat([row_t, col_t], dim=0)
        return row_t

    def get_or_compute(self, image_path: str, use_col_avg: bool = False,
                       proj_length: int = 512, col_roi_fraction: float = 0.33,
                       col_length: int = 512) -> torch.Tensor:

        if not self.enabled:
            return self._compute(image_path, use_col_avg, proj_length,
                                 col_roi_fraction, col_length)
        key = self._key(image_path, use_col_avg, proj_length,
                        col_roi_fraction, col_length)
        cached = self._mem.get(key)
        if cached is not None:
            return cached.clone()
        disk = self._disk_path(key)
        if disk is not None and disk.exists():
            cached = torch.load(disk, map_location="cpu", weights_only=True)
            self._mem[key] = cached
            return cached.clone()
        cached = self._compute(image_path, use_col_avg, proj_length,
                               col_roi_fraction, col_length)
        self._mem[key] = cached
        if disk is not None:
            torch.save(cached, str(disk))
        return cached.clone()
