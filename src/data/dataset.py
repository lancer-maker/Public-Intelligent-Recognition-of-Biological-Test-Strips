import sys
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import cv2
import torch
import yaml
from torch.utils.data import Dataset
from configs.load_config import get_main_config

try:
    from .utils import extract_1d_projection_resampled, extract_col_avg_projection_resampled, resize_to_standard_dpi
    from .transforms import get_val_transforms, get_projection_transforms
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data.utils import extract_1d_projection_resampled, extract_col_avg_projection_resampled, resize_to_standard_dpi
    from data.transforms import get_val_transforms, get_projection_transforms



dataset_config = get_main_config()
config_dpi_size = tuple(dataset_config.get("data", {}).get("standard_dpi_size", [220, 505]))
config_proj_length = int(dataset_config.get("data", {}).get("proj_length", 512))


class TestStripDataset(Dataset):

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transforms: Optional[Callable] = None,
        projection_transforms: Optional[Callable] = None,
        standard_dpi_size: Optional[Tuple[int, int]] = None,
        proj_length: Optional[int] = None,
        use_col_avg: Optional[bool] = None,
        col_roi_fraction: Optional[float] = None,
        col_length: Optional[int] = None,
        proj_cache: Optional[object] = None,
        skip_projection: bool = False
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms if transforms is not None else get_val_transforms()
        self.projection_transforms = get_projection_transforms() if projection_transforms is None else projection_transforms
        self.standard_dpi_size = tuple(standard_dpi_size) if standard_dpi_size is not None else config_dpi_size
        self.proj_length = proj_length if proj_length is not None else config_proj_length
        self.proj_cache = proj_cache
        self.skip_projection = bool(skip_projection)

        _data_cfg = dataset_config.get("data", {})
        self.use_col_avg = bool(_data_cfg.get("proj_use_col_avg", False)) if use_col_avg is None else bool(use_col_avg)
        self.col_roi_fraction = float(_data_cfg.get("proj_col_roi_fraction", 0.33)) if col_roi_fraction is None else float(col_roi_fraction)
        self.col_length = int(_data_cfg.get("proj_col_length", 512)) if col_length is None else int(col_length)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        raw_bgr = cv2.imread(image_path)
        if raw_bgr is None:
            raise FileNotFoundError(f"无法读取文件: {image_path}")
        raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

        if self.skip_projection:
            proj_tensor = torch.zeros((1, self.proj_length), dtype=torch.float32)
        elif self.proj_cache is not None:
            proj_tensor = self.proj_cache.get_or_compute(
                image_path,
                use_col_avg=self.use_col_avg,
                proj_length=self.proj_length,
                col_roi_fraction=self.col_roi_fraction,
                col_length=self.col_length,
            )
        else:
            proj_image = raw_rgb.copy()
            if self.projection_transforms:
                proj_image = self.projection_transforms(image=proj_image)['image']
            proj_tensor = extract_1d_projection_resampled(proj_image, target_length=self.proj_length)
            if self.use_col_avg:
                col_tensor = extract_col_avg_projection_resampled(
                    proj_image, target_length=self.col_length, roi_fraction=self.col_roi_fraction
                )
                proj_tensor = torch.cat([proj_tensor, col_tensor], dim=0)


        dpi_aligned_rgb = resize_to_standard_dpi(raw_rgb, target_size=self.standard_dpi_size)

        augmented = self.transforms(image=dpi_aligned_rgb)
        image_tensor = augmented['image']

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return image_tensor, proj_tensor, label_tensor

if __name__ == "__main__":
    try:
        from .transforms import get_train_transforms
    except ImportError:
        from data.transforms import get_train_transforms

    test_paths = ["D:\\software\\vscode\\TOXY\\data\\raw\\P-24\\P0_01_1.jpg.png"]
    test_labels = [1]
    
    dataset = TestStripDataset(
        image_paths=test_paths, 
        labels=test_labels, 
        transforms=get_train_transforms()
    )
    
    img, proj, label = dataset[0]
    
    print("Image Tensor Shape:", img.shape)
    print("Proj Tensor Shape: ", proj.shape)
    print("Label Tensor:       ", label)