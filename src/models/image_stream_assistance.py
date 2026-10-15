
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dual_stream_net import DualStreamStripNet


class StripPositionHead(nn.Module):


    def __init__(self, feature_dim: int = 128, hidden: int = 64):
        super(StripPositionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, 2),
        )

    def forward(self, img_feats: torch.Tensor) -> torch.Tensor:

        return self.net(img_feats)


class ImageStreamAuxNet(nn.Module):


    def __init__(self, base: DualStreamStripNet, position_head: StripPositionHead):
        super(ImageStreamAuxNet, self).__init__()
        self.base = base
        self.position_head = position_head

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor,
                guide_centers: torch.Tensor = None):

        img_feats = self.base.image_stream(img_tensor, guide_centers=guide_centers)
        proj_feats = self.base.proj_stream(proj_tensor)
        logits = self.base.classifier(torch.cat([img_feats, proj_feats], dim=1))
        aux_logits = self.position_head(img_feats)
        return logits, aux_logits


def make_strip_pseudo_labels(
    labels: torch.Tensor,
    pos_center: float = 0.5,
    jitter: float = 0.0,
) -> tuple:

    labels = labels.float().detach()
    presence = (labels > 0.5).float()
    if jitter > 0:
        noise = torch.randn_like(labels) * float(jitter) * presence
        y_target = (pos_center + noise).clamp(0.0, 1.0) * presence
    else:
        y_target = torch.full_like(labels, float(pos_center)) * presence
    return presence, y_target


def strip_position_loss(
    head_logits: torch.Tensor,
    presence: torch.Tensor,
    y_target: torch.Tensor,
    center_weight: float = 1.0,
) -> dict:

    presence = presence.float()
    y_target = y_target.float()
    presence_loss = F.binary_cross_entropy_with_logits(head_logits[:, 0], presence)
    mask = presence > 0.5
    if bool(mask.any()):
        center_loss = F.mse_loss(head_logits[:, 1][mask], y_target[mask])
    else:
        center_loss = head_logits.new_zeros(())
    total = presence_loss + center_weight * center_loss
    return {"presence_loss": presence_loss, "center_loss": center_loss, "total": total}
