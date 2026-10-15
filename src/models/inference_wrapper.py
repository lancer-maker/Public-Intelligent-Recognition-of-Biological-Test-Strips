import torch
import torch.nn as nn


class InferenceWrapper(nn.Module):

    def __init__(self, model: nn.Module):
        super(InferenceWrapper, self).__init__()
        self.model = model
        self.model.eval()

    def forward(self, img_tensor: torch.Tensor, proj_tensor: torch.Tensor) -> torch.Tensor:

        logits = self.model(img_tensor, proj_tensor)
        probs = torch.sigmoid(logits)
        return probs