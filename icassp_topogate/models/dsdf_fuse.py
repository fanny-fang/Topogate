from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_heads(channels: int, requested: int) -> int:
    for heads in range(min(requested, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class DSDFFuse(nn.Module):
    """Controlled dual-stage DEM-guided feature fusion transplant.

    The early branch predicts a bounded multiplicative RGB gain from terrain.
    The late branch uses RGB queries and terrain keys/values at one stage. The
    current project's TopoStem is retained so this remains a same-backbone
    component comparison rather than a reproduction of the full D2FLS-Net.
    """

    def __init__(
        self,
        topo_in_channels: int,
        rgb_channels: tuple[int, ...],
        topo_channels: tuple[int, ...],
        early_hidden: int = 32,
        early_gain_max: float = 1.0,
        late_stage: int = 3,
        num_heads: int = 4,
        late_residual: bool = False,
    ):
        super().__init__()
        if late_stage < 0 or late_stage >= len(rgb_channels):
            raise ValueError(f"late_stage={late_stage} is outside the feature pyramid.")
        self.late_stage = int(late_stage)
        self.early_gain_max = float(early_gain_max)
        self.late_residual = bool(late_residual)
        self.early_guidance = nn.Sequential(
            nn.Conv2d(topo_in_channels, early_hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(early_hidden),
            nn.GELU(),
            nn.Conv2d(early_hidden, 1, 1, bias=True),
        )
        channels = rgb_channels[self.late_stage]
        self.topo_proj = nn.Conv2d(topo_channels[self.late_stage], channels, 1, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            channels,
            num_heads=_valid_heads(channels, num_heads),
            batch_first=True,
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )

    def guide_rgb(self, rgb: torch.Tensor, topo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.early_guidance(topo))
        if gate.shape[-2:] != rgb.shape[-2:]:
            gate = F.interpolate(gate, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return rgb * (1.0 + self.early_gain_max * gate), gate

    def forward(
        self,
        rgb_feats: list[torch.Tensor],
        topo_feats: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        outs = list(rgb_feats)
        index = self.late_stage
        rgb = rgb_feats[index]
        topo = topo_feats[index]
        if topo.shape[-2:] != rgb.shape[-2:]:
            topo = F.interpolate(topo, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        topo = self.topo_proj(topo)
        batch, channels, height, width = rgb.shape
        query = rgb.flatten(2).transpose(1, 2)
        key_value = topo.flatten(2).transpose(1, 2)
        attended, _ = self.cross_attention(query, key_value, key_value, need_weights=False)
        attended = attended.transpose(1, 2).reshape(batch, channels, height, width)
        attended = self.output_proj(attended)
        outs[index] = rgb + attended if self.late_residual else attended
        return outs, []
