
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from . import config as dia_config

PROJECT_ROOT = dia_config.PROJECT_ROOT


def _natural_key(text: str):

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def default_weight_paths() -> List[Path]:

    if dia_config.E_WEIGHT_PATHS:
        return [Path(p) for p in dia_config.E_WEIGHT_PATHS]
    root = Path(dia_config.E_CHECKPOINT_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"E 模式权重目录不存在: {root}")
    paths = sorted(root.glob("*/best_model.pth"), key=lambda p: _natural_key(p.parent.name))
    if not paths:
        raise FileNotFoundError(f"{root} 下未找到 */best_model.pth")
    return paths


def read_metrics(weight_path: Path) -> Dict[str, float]:

    csv_path = Path(weight_path).parent / "metrics.csv"
    if not csv_path.exists():
        return {}
    try:
        row = pd.read_csv(csv_path).iloc[0]
    except Exception:
        return {}
    out: Dict[str, float] = {}
    for key in ("best_threshold", "best_val_auc", "val_auc", "th_sensitivity", "th_specificity"):
        if key in row and pd.notna(row[key]):
            try:
                out[key] = float(row[key])
            except (TypeError, ValueError):
                pass
    return out


class EnsemblePredictor:


    def __init__(self,
                 weight_paths: Optional[Sequence[Path]] = None,
                 device: Optional[str] = None,
                 threshold_override: Optional[float] = None):

        self.weight_paths: List[Path] = [Path(p) for p in (weight_paths or default_weight_paths())]
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold_override = threshold_override
        self.models: List[torch.nn.Module] = []
        self.metrics: List[Dict[str, float]] = []
        self.feature_dim = 128
        self.dropout_rate = 0.6
        self.standard_dpi_size = (220, 505)
        self.proj_length = 512
        self.primary_threshold = 0.5
        self.neg_max = 0.5
        self.pos_min = 0.5
        self._loaded = False

    def load(self) -> "EnsemblePredictor":

        if self._loaded:
            return self

        from src.models.gated_fusion import GatedDualStreamNet
        from src.utils.Fine_Tuning_helper import load_config
        from src.utils.model_helper import load_model_state_dict

        cfg = load_config()
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        self.feature_dim = int(model_cfg.get("feature_dim", 128))
        self.dropout_rate = float(model_cfg.get("dropout_rate", 0.6))
        self.standard_dpi_size = tuple(data_cfg.get("standard_dpi_size", (220, 505)))
        self.proj_length = int(data_cfg.get("proj_length", 512))

        for weight_path in self.weight_paths:
            if not weight_path.exists():
                print(f"[predictor] 跳过不存在的权重: {weight_path}")
                continue
            state = load_model_state_dict(str(weight_path))
            gate_w = state.get("gate_mlp.net.2.weight")
            gate_hidden = int(gate_w.shape[1]) if isinstance(gate_w, torch.Tensor) and gate_w.ndim == 2 else 64
            model = GatedDualStreamNet(
                pretrained_2d=False,
                feature_dim=self.feature_dim,
                dropout_rate=self.dropout_rate,
                gate_hidden=gate_hidden,
            )
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[predictor] [warn] {weight_path.name} 权重键不匹配: "
                      f"缺失 {len(missing)} / 多余 {len(unexpected)}")
            model.to(self.device).eval()
            self.models.append(model)
            self.metrics.append(read_metrics(weight_path))
            print(f"[predictor] 已加载 {weight_path} (gate_hidden={gate_hidden})")

        if not self.models:
            raise RuntimeError("没有任何可用的 E 集成模型权重。")

        self._resolve_thresholds()
        self._loaded = True
        return self

    def _resolve_thresholds(self) -> None:

        if self.threshold_override is not None:
            self.primary_threshold = float(self.threshold_override)
        else:
            vals = [m["best_threshold"] for m in self.metrics if "best_threshold" in m]
            self.primary_threshold = float(np.median(vals)) if vals else 0.5
        self.neg_max = float(dia_config.NEG_MAX) if dia_config.NEG_MAX is not None else self.primary_threshold
        self.pos_min = float(dia_config.POS_MIN) if dia_config.POS_MIN is not None else self.primary_threshold

    @property
    def best_index(self) -> int:

        aucs = [m.get("best_val_auc", m.get("val_auc", float("-inf"))) for m in self.metrics]
        return int(np.argmax(aucs)) if aucs else 0

    @property
    def best_model(self) -> torch.nn.Module:

        return self.models[self.best_index]

    @property
    def best_weight_path(self) -> Path:

        return self.weight_paths[self.best_index]

    def build_input(self, image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:

        from src.data.dataset import TestStripDataset
        from src.data.transforms import get_val_transforms

        ds = TestStripDataset(
            image_paths=[str(image_path)],
            labels=[0],
            transforms=get_val_transforms(),
            projection_transforms=False,
            standard_dpi_size=tuple(self.standard_dpi_size),
            proj_length=self.proj_length,
        )
        img_tensor, proj_tensor, _ = ds[0]
        return img_tensor.unsqueeze(0).to(self.device), proj_tensor.unsqueeze(0).to(self.device)

    def predict(self, image_path: Path) -> Dict[str, object]:

        self.load()
        img_t, proj_t = self.build_input(Path(image_path))
        member_probs: List[float] = []
        with torch.no_grad():
            for model in self.models:
                logits = model(img_t, proj_t)
                member_probs.append(float(torch.sigmoid(logits).item()))
        prob = float(np.mean(member_probs))
        return {
            "prob": prob,
            "member_probs": member_probs,
            "label": self.decide(prob),
        }

    def decide(self, prob: float) -> str:

        if prob < self.neg_max:
            return dia_config.LABEL_NEGATIVE
        if prob > self.pos_min:
            return dia_config.LABEL_POSITIVE
        return dia_config.LABEL_SUSPICIOUS

    def describe(self) -> str:

        names = ", ".join(p.parent.name for p in self.weight_paths)
        return (f"E 集成 {len(self.models)} 模型 [{names}] | 主阈值={self.primary_threshold:.4f} "
                f"| 阴性<{self.neg_max:.4f} / 阳性>{self.pos_min:.4f} | 设备={self.device}")
