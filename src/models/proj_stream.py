import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from configs.load_config import get_main_config

config = get_main_config()
config_feature_dim = int(config.get("model", {}).get("feature_dim", 128))
config_proj_use_col_avg = bool(config.get("data", {}).get("proj_use_col_avg", False))
config_proj_signal_channels = 1 + int(config_proj_use_col_avg)
config_proj_length = int(config.get("data", {}).get("proj_length", 512))
config_proj_add_position = bool(config.get("model", {}).get("proj_add_position_channel", False))
config_proj_dropout = float(config.get("model", {}).get("proj_dropout_rate", 0.6))
config_proj_kernel_sizes = [int(k) for k in config.get("model", {}).get("proj_conv_kernel_sizes", [15, 9, 5, 3])]
config_proj_conv_channels = [int(c) for c in config.get("model", {}).get("proj_conv_channels", [32, 64, 128, 256])]


class ProjStream(nn.Module):


    def __init__(self, feature_dim: int = config_feature_dim, in_channels: int = None):
        super(ProjStream, self).__init__()
        if in_channels is None:
            in_channels = config_proj_signal_channels
        self.in_channels = int(in_channels)
        self.use_position = config_proj_add_position

        kernel_sizes = list(config_proj_kernel_sizes)
        conv_channels = list(config_proj_conv_channels)
        if len(kernel_sizes) != len(conv_channels):
            raise ValueError(
                f"proj_conv_kernel_sizes 与 proj_conv_channels 长度不一致: "
                f"{len(kernel_sizes)} vs {len(conv_channels)}"
            )

        conv_in = self.in_channels + (1 if self.use_position else 0)
        blocks: list = []
        cin = conv_in
        for i, (k, cout) in enumerate(zip(kernel_sizes, conv_channels)):
            blocks.append(nn.Conv1d(cin, cout, kernel_size=k, padding=k // 2))
            blocks.append(nn.BatchNorm1d(cout))
            blocks.append(nn.ReLU())
            if i < len(kernel_sizes) - 1:
                blocks.append(nn.MaxPool1d(kernel_size=2))
            cin = cout

        blocks.append(nn.AdaptiveAvgPool1d(1))
        self.conv_layers = nn.Sequential(*blocks)

        if self.use_position:
            self.learned_pos = nn.Parameter(torch.zeros(1, 1, config_proj_length))

        fc_in = conv_channels[-1]

        self.dropout = nn.Dropout(p=config_proj_dropout)
        self.fc = nn.Linear(fc_in, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.use_position:
            pos = self.learned_pos.expand(x.shape[0], 1, -1)
            x = torch.cat([x, pos], dim=1)

        feats = self.conv_layers(x)
        feats = torch.flatten(feats, 1)

        out = F.relu(self.fc(self.dropout(feats)))
        return out