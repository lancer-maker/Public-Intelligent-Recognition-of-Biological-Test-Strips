
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from configs.load_config import get_main_config
except ImportError:
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from configs.load_config import get_main_config

_config = get_main_config()
_model_cfg = _config.get("model", {})
config_guide_enabled = bool(_model_cfg.get("spatial_guide_enabled", True))
config_guide_sigma = float(_model_cfg.get("spatial_guide_sigma", 0.25))
config_guide_pos_center = float(_model_cfg.get("spatial_guide_pos_center", 0.5))


class SpatialGuidedPooling(nn.Module):


    def __init__(self, sigma: Optional[float] = None, pos_center: Optional[float] = None):
        super(SpatialGuidedPooling, self).__init__()
        self.sigma = float(config_guide_sigma if sigma is None else sigma)
        self.pos_center = float(config_guide_pos_center if pos_center is None else pos_center)

    def gaussian_weight_map(self, feat_map: torch.Tensor,
                            centers: Optional[torch.Tensor] = None) -> torch.Tensor:

        b, _, h, w = feat_map.shape
        device, dtype = feat_map.device, feat_map.dtype

        if centers is None:
            centers = torch.full((b,), self.pos_center, device=device, dtype=dtype)
        else:
            centers = centers.to(device=device, dtype=dtype).view(b)

        ys = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype).view(1, h)
        c = centers.view(b, 1)
        sigma = max(self.sigma, 1e-6)
        wgt = torch.exp(-((ys - c) ** 2) / (2.0 * sigma * sigma))
        wgt = wgt / (wgt.sum(dim=1, keepdim=True) + 1e-8)
        return wgt.view(b, 1, h, 1).expand(b, 1, h, w)

    def forward(self, feat_map: torch.Tensor,
                centers: Optional[torch.Tensor] = None) -> torch.Tensor:

        weight_map = self.gaussian_weight_map(feat_map, centers)
        weighted = feat_map * weight_map
        denom = weight_map.sum(dim=(2, 3)) + 1e-8
        return weighted.sum(dim=(2, 3)) / denom
