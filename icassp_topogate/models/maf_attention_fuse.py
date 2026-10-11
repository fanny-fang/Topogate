from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAFStyleAttentionFuse(nn.Module):
    """Image-guided topographic attention fusion baseline.

    This is a compact MAF-style baseline rather than a full MAF-UNet copy. RGB
    features guide spatial and channel selection of topo features, then selected
    topo features are fused back by concat + 1x1 projection.
    """

    def __init__(
        self,
        rgb_channels: tuple[int, ...],
        topo_channels: tuple[int, ...],
        inject_layers: tuple[int, ...] = (1, 2, 3),
        reduction: int = 4,
    ):
        super().__init__()
        self.inject_layers = set(inject_layers)
        self.topo_proj = nn.ModuleList()
        self.spatial_attn = nn.ModuleList()
        self.channel_attn = nn.ModuleList()
        self.fuse = nn.ModuleList()
        for rc, tc in zip(rgb_channels, topo_channels):
            hidden = max(16, rc // reduction)
            self.topo_proj.append(nn.Conv2d(tc, rc, 1, bias=False))
            self.spatial_attn.append(
                nn.Sequential(
                    nn.Conv2d(rc * 2, hidden, 1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, 1, 1, bias=True),
                    nn.Sigmoid(),
                )
            )
            self.channel_attn.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(rc * 2, hidden, 1, bias=True),
                    nn.GELU(),
                    nn.Conv2d(hidden, rc, 1, bias=True),
                    nn.Sigmoid(),
                )
            )
            self.fuse.append(
                nn.Sequential(
                    nn.Conv2d(rc * 2, rc, 1, bias=False),
                    nn.BatchNorm2d(rc),
                    nn.GELU(),
                )
            )

    def forward(self, rgb_feats: list[torch.Tensor], topo_feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        outs = []
        attn_maps = []
        for i, feat in enumerate(rgb_feats):
            if i not in self.inject_layers:
                outs.append(feat)
                continue
            topo = topo_feats[i]
            if topo.shape[-2:] != feat.shape[-2:]:
                topo = F.interpolate(topo, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            topo = self.topo_proj[i](topo)
            joint = torch.cat([feat, topo], dim=1)
            spatial = self.spatial_attn[i](joint)
            channel = self.channel_attn[i](joint)
            selected_topo = topo * spatial * channel
            outs.append(self.fuse[i](torch.cat([feat, selected_topo], dim=1)))
            attn_maps.append(spatial)
        return outs, attn_maps
