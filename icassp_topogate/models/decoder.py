from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerLikeDecoder(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], decoder_channels: int = 256, num_classes: int = 1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, decoder_channels, 1) for c in in_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_channels * len(in_channels), decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )

    def forward(self, feats: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        target_size = feats[0].shape[-2:]
        ups = []
        for feat, proj in zip(feats, self.proj):
            x = proj(feat)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            ups.append(x)
        x = torch.cat(ups, dim=1)
        x = self.fuse(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
