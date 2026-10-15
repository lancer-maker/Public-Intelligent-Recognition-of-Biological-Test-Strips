import torch
import torch.nn as nn
import torch.optim as optim


def get_weighted_bce_loss(pos_weight_value: float, device: torch.device) -> nn.Module:

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def get_optimizer(
    model: nn.Module, 
    lr: float = 1e-4, 
    weight_decay: float = 1e-3
) -> optim.Optimizer:

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)