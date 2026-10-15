
from typing import List, Optional, Tuple

import torch


def get_backbone_block_names(model: torch.nn.Module) -> List[str]:

    blocks = model.image_stream.backbone.blocks
    n = len(blocks)
    return [f"blocks.{i}" for i in range(n)]


def freeze_all_backbone(model: torch.nn.Module) -> None:

    for param in model.image_stream.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone_block(model: torch.nn.Module, block_name: str) -> None:

    module = model.image_stream.backbone
    for part in block_name.split("."):
        module = getattr(module, part)
    for param in module.parameters():
        param.requires_grad = True


def create_stage2_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-3,
    t_max: int = 15,
    eta_min: float = 1e-7,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=eta_min
    )
    return optimizer, scheduler


class ProgressiveUnfreezeScheduler:


    def __init__(
        self,
        model: torch.nn.Module,
        interval: int = 4,
        start_block: Optional[int] = None,
        num_layers: Optional[int] = None,
    ):
        self.model = model
        self.interval = max(1, int(interval))
        self.unfreeze_order: List[str] = self._build_order(start_block, num_layers)
        self.next_idx = 0
        self.unfreezed: List[str] = []
        self.unfreeze_log: List[Tuple[int, str]] = []

    def _build_order(self, start_block: Optional[int], num_layers: Optional[int]) -> List[str]:

        names = get_backbone_block_names(self.model)
        n = len(names)
        start = n - 1 if start_block is None else int(start_block)
        start = max(0, min(n - 1, start))
        if num_layers is None:
            num = start + 1
        else:
            num = max(1, int(num_layers))
        num = min(num, start + 1)
        return [f"blocks.{i}" for i in range(start, start - num, -1)]

    @property
    def active_layers(self) -> List[str]:

        return list(self.unfreezed)

    def apply_for_epoch(self, epoch: int, stage_name: str) -> List[str]:

        newly: List[str] = []
        if stage_name != "Stage2":
            return newly
        while (
            self.next_idx < len(self.unfreeze_order)
            and epoch >= (self.next_idx + 1) * self.interval
        ):
            block = self.unfreeze_order[self.next_idx]
            unfreeze_backbone_block(self.model, block)
            self.unfreezed.append(block)
            self.unfreeze_log.append((epoch, block))
            newly.append(block)
            self.next_idx += 1
        return newly

    def schedule_str(self) -> str:

        return ", ".join(f"{blk}@e{ep}" for ep, blk in self.unfreeze_log)
