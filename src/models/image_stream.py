import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configs.load_config import get_main_config

try:
    from .spatial_guide import SpatialGuidedPooling
except ImportError:
    from spatial_guide import SpatialGuidedPooling

config = get_main_config()
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_spatial_guide_enabled = bool(config.get("model", {}).get("spatial_guide_enabled", True))

PRETRAINED_WEIGHTS_DIR = PROJECT_ROOT / "data" / "pretrained_weights"


def _load_local_pretrained_weights(model: nn.Module, model_name: str) -> None:

    weight_path = PRETRAINED_WEIGHTS_DIR / f"{model_name}.pth.bin"
    if not weight_path.exists():
        return

    checkpoint = torch.load(str(weight_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"预训练权重文件格式不正确: {weight_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_key = key.replace("module.", "")
        cleaned_state_dict[cleaned_key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)


class ImageStream(nn.Module):

    def __init__(self, pretrained: bool = True, feature_dim: int = config_feature_dim):
        super(ImageStream, self).__init__()

        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)

        if pretrained:
            _load_local_pretrained_weights(self.backbone, "efficientnet_b0")
            if not any(param.requires_grad for param in self.backbone.parameters()):
                pass

        in_features = self.backbone.num_features
        
        self.fc = nn.Linear(in_features, feature_dim)

        self.strip_head = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        self.use_guided_pool = config_spatial_guide_enabled
        self.guided_pool = SpatialGuidedPooling()

    def forward_features_map(self, x: torch.Tensor) -> torch.Tensor:

        return self.backbone.forward_features(x)

    def base_pooled(self, x: torch.Tensor) -> torch.Tensor:

        return F.relu(self.fc(self.backbone(x)))

    def predict_strip_center(self, x: torch.Tensor) -> torch.Tensor:

        gap = self.backbone.forward_features(x).mean(dim=(2, 3))
        return torch.sigmoid(self.strip_head(gap)[:, 1])

    def forward(self, x: torch.Tensor, guide_centers: torch.Tensor = None) -> torch.Tensor:

        if self.use_guided_pool:
            feat_map = self.backbone.forward_features(x)
            if guide_centers is None and not self.training:
                gap = feat_map.mean(dim=(2, 3))
                with torch.no_grad():
                    guide_centers = torch.sigmoid(self.strip_head(gap)[:, 1])
            feats = self.guided_pool(feat_map, centers=guide_centers)
        else:
            feats = self.backbone(x)
        out = F.relu(self.fc(feats))
        return out

    def freeze_backbone(self) -> None:

        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:

        for param in self.backbone.parameters():
            param.requires_grad = True