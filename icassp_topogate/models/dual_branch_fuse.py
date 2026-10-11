from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualBranchFuse(nn.Module):
    def __init__(self, rgb_channels: tuple[int, ...], topo_channels: tuple[int, ...], inject_layers: tuple[int, ...] = (1, 2, 3)):
        super().__init__()
        self.inject_layers = set(inject_layers)
        self.proj = nn.ModuleList()
        for rc, tc in zip(rgb_channels, topo_channels):
            self.proj.append(
                nn.Sequential(
                    nn.Conv2d(rc + tc, rc, 1, bias=False),
                    nn.BatchNorm2d(rc),
                    nn.GELU(),
                )
            )

    def forward(self, rgb_feats: list[torch.Tensor], topo_feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        outs = []
        for i, feat in enumerate(rgb_feats):
            if i not in self.inject_layers:
                outs.append(feat)
                continue
            topo = topo_feats[i]
            if topo.shape[-2:] != feat.shape[-2:]:
                topo = F.interpolate(topo, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            outs.append(self.proj[i](torch.cat([feat, topo], dim=1)))
        return outs, []
