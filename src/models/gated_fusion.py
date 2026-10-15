
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.load_config import get_main_config

try:
    from .image_stream import ImageStream
    from .proj_stream import ProjStream
    from .image_stream_assistance import StripPositionHead
except ImportError:
    from image_stream import ImageStream
    from proj_stream import ProjStream
    from image_stream_assistance import StripPositionHead

config = get_main_config()
config_pretrained_2d = bool(config.get("model", {}).get("pretrained", True))
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_dropout_rate = float(config.get("model", {}).get("dropout_rate", 0.5))


class GateMLP(nn.Module):


    def __init__(self, feature_dim: int = config_feature_dim, hidden: int = 64):
        super(GateMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.net[2].bias, 1.0)

    def forward(self, proj_feats: torch.Tensor) -> torch.Tensor:

        return self.net(proj_feats)


class GatedDualStreamNet(nn.Module):


    def __init__(
        self,
        pretrained_2d: bool = config_pretrained_2d,
        feature_dim: int = config_feature_dim,
        dropout_rate: float = config_dropout_rate,
        gate_hidden: int = 64,
    ):
        super(GatedDualStreamNet, self).__init__()
        self.image_stream = ImageStream(pretrained=pretrained_2d, feature_dim=feature_dim)
        self.proj_stream = ProjStream(feature_dim=feature_dim)
        self.gate_mlp = GateMLP(feature_dim=feature_dim, hidden=gate_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor,
                guide_centers: torch.Tensor = None) -> torch.Tensor:
        img_feats = self.image_stream(img_tensor, guide_centers=guide_centers)
        proj_feats = self.proj_stream(proj_tensor)
        gate = self.gate_mlp(proj_feats)
        img_gated = img_feats * gate
        fused = torch.cat([img_gated, proj_feats], dim=1)
        return self.classifier(fused)

    def freeze_image_stream(self) -> None:

        self.image_stream.freeze_backbone()

    def unfreeze_image_stream(self) -> None:

        self.image_stream.unfreeze_backbone()


class GatedDualStreamAuxNet(nn.Module):


    def __init__(self, base: GatedDualStreamNet, position_head: StripPositionHead):
        super(GatedDualStreamAuxNet, self).__init__()
        self.base = base
        self.position_head = position_head

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor,
                guide_centers: torch.Tensor = None):

        img_feats = self.base.image_stream(img_tensor, guide_centers=guide_centers)
        proj_feats = self.base.proj_stream(proj_tensor)
        gate = self.base.gate_mlp(proj_feats)
        img_gated = img_feats * gate
        logits = self.base.classifier(torch.cat([img_gated, proj_feats], dim=1))
        aux_logits = self.position_head(img_feats)
        return logits, aux_logits


if __name__ == "__main__":
    base = GatedDualStreamNet(pretrained_2d=False)
    dummy_img = torch.randn(4, 3, 505, 220)
    dummy_proj = torch.randn(4, base.proj_stream.in_channels, 512)
    print("GatedDualStreamNet logits shape:", base(dummy_img, dummy_proj).shape)

    head = StripPositionHead(feature_dim=config_feature_dim)
    aux_net = GatedDualStreamAuxNet(base, head)
    logits, aux_logits = aux_net(dummy_img, dummy_proj)
    print("GatedDualStreamAuxNet logits shape:", logits.shape)
    print("GatedDualStreamAuxNet aux shape:   ", aux_logits.shape)
