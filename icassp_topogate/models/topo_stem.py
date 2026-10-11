from __future__ import annotations

import torch
import torch.nn as nn


class TopoStem(nn.Module):
    def __init__(self, in_channels: int = 3, channels: tuple[int, int, int, int] = (16, 32, 64, 128)):
        super().__init__()
        stages = []
        ch = in_channels
        for out_ch in channels:
            stages.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(num_groups=min(4, out_ch), num_channels=out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=min(4, out_ch), num_channels=out_ch),
                    nn.GELU(),
                )
            )
            ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats
