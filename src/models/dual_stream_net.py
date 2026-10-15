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
except ImportError:
    from image_stream import ImageStream
    from proj_stream import ProjStream

config = get_main_config()
config_pretrained_2d = bool(config.get("model", {}).get("pretrained_2d", True))
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_dropout_rate = float(config.get("model", {}).get("dropout_rate", 0.5))

class DualStreamStripNet(nn.Module):

    def __init__(
        self, 
        pretrained_2d: bool = config_pretrained_2d, 
        feature_dim: int = config_feature_dim, 
        dropout_rate: float = config_dropout_rate
    ):
        super(DualStreamStripNet, self).__init__()
        
        self.image_stream = ImageStream(pretrained=pretrained_2d, feature_dim=feature_dim)
        self.proj_stream = ProjStream(feature_dim=feature_dim)
        
        combined_dim = feature_dim + feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config_feature_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(config_feature_dim, 1)
        )

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor) -> torch.Tensor:

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor,
                guide_centers: torch.Tensor = None) -> torch.Tensor:

        img_feats = self.image_stream(img_tensor, guide_centers=guide_centers)
        proj_feats = self.proj_stream(proj_tensor)
        
        fused_feats = torch.cat([img_feats, proj_feats], dim=1)
        
        logits = self.classifier(fused_feats)
        
        return logits

    def freeze_image_stream(self) -> None:

        self.image_stream.freeze_backbone()

    def unfreeze_image_stream(self) -> None:

        self.image_stream.unfreeze_backbone()

if __name__ == "__main__":
    try:
        from .inference_wrapper import InferenceWrapper
    except ImportError:
        from inference_wrapper import InferenceWrapper
    net = DualStreamStripNet(pretrained_2d=False)
    
    dummy_img = torch.randn(4, 3, 505, 220)
    dummy_proj = torch.randn(4, net.proj_stream.in_channels, 512)
    
    logits = net(dummy_img, dummy_proj)
    print("Logits Output Shape:", logits.shape)
    
    infer_model = InferenceWrapper(net)
    probs = infer_model(dummy_img, dummy_proj)
    print("Probs Output Shape: ", probs.shape)
    print("Sample Output Prob: ", probs[0].item())