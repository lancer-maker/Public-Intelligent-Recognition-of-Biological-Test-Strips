import torch
from typing import Union


def load_model_state(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def load_model_state_dict(checkpoint_path: str) -> dict:

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    return checkpoint if isinstance(checkpoint, dict) else {}
