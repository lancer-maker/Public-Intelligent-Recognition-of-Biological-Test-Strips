
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
    from .image_stream_assistance import StripPositionHead
except ImportError:
    from image_stream import ImageStream
    from image_stream_assistance import StripPositionHead

config = get_main_config()
config_pretrained_2d = bool(config.get("model", {}).get("pretrained", True))
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_dropout_rate = float(config.get("model", {}).get("dropout_rate", 0.5))


class ImageOnlyBase(nn.Module):


    def __init__(
        self,
        pretrained_2d: bool = config_pretrained_2d,
        feature_dim: int = config_feature_dim,
        dropout_rate: float = config_dropout_rate,
    ):
        super(ImageOnlyBase, self).__init__()
        self.image_stream = ImageStream(pretrained=pretrained_2d, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, img_tensor: torch.Tensor, guide_centers: torch.Tensor = None) -> torch.Tensor:

        img_feats = self.image_stream(img_tensor, guide_centers=guide_centers)
        return self.classifier(img_feats)

    def freeze_image_stream(self) -> None:

        self.image_stream.freeze_backbone()

    def unfreeze_image_stream(self) -> None:

        self.image_stream.unfreeze_backbone()


class ImageOnlyAuxNet(nn.Module):


    def __init__(self, base: ImageOnlyBase, position_head=None):
        super(ImageOnlyAuxNet, self).__init__()
        self.base = base
        self.position_head = position_head

    def forward(self, img_tensor: torch.Tensor, guide_centers: torch.Tensor = None):

        img_feats = self.base.image_stream(img_tensor, guide_centers=guide_centers)
        logits = self.base.classifier(img_feats)
        if self.position_head is not None:
            return logits, self.position_head(img_feats)
        return logits


if __name__ == "__main__":
    net_plain = ImageOnlyBase(pretrained_2d=False)
    dummy_img = torch.randn(4, 3, 505, 220)
    print("ImageOnlyBase logits shape:", net_plain(dummy_img).shape)

    aux_net = ImageOnlyAuxNet(ImageOnlyBase(pretrained_2d=False),
                              StripPositionHead(feature_dim=config_feature_dim))
    logits, aux_logits = aux_net(dummy_img)
    print("ImageOnlyAuxNet logits shape:", logits.shape)
    print("ImageOnlyAuxNet aux shape:   ", aux_logits.shape)
