from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class _ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, groups: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups, bias=False),
            _gn(out_channels),
            nn.GELU(),
        )


class _TopoGateBlock(nn.Module):
    """RGB-guided terrain selection for one feature scale."""

    def __init__(
        self,
        rgb_channels: int,
        topo_channels: int,
        reduction: int = 4,
        min_gate: float = 0.05,
    ):
        super().__init__()
        hidden = max(16, rgb_channels // reduction)
        self.min_gate = min_gate
        self.topo_proj = nn.Sequential(
            _ConvNormAct(topo_channels, rgb_channels, kernel_size=1),
            _ConvNormAct(rgb_channels, rgb_channels, kernel_size=3, groups=rgb_channels),
            _ConvNormAct(rgb_channels, rgb_channels, kernel_size=1),
        )
        self.joint_reduce = nn.Sequential(
            _ConvNormAct(rgb_channels * 2, hidden, kernel_size=1),
            _ConvNormAct(hidden, hidden, kernel_size=3),
        )
        self.spatial_head = nn.Conv2d(hidden, 1, 1, bias=True)
        self.channel_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rgb_channels * 2, hidden, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, rgb_channels, 1, bias=True),
        )
        self.residual_proj = nn.Sequential(
            _ConvNormAct(rgb_channels, rgb_channels, kernel_size=3, groups=rgb_channels),
            nn.Conv2d(rgb_channels, rgb_channels, 1, bias=False),
        )
        self.modulation_head = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, rgb: torch.Tensor, topo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topo = self.topo_proj(topo)
        joint = torch.cat([rgb, topo], dim=1)
        context = self.joint_reduce(joint)
        spatial = torch.sigmoid(self.spatial_head(context))
        if self.min_gate > 0:
            spatial = self.min_gate + (1.0 - self.min_gate) * spatial
        channel = torch.sigmoid(self.channel_head(joint))
        if self.min_gate > 0:
            channel = self.min_gate + (1.0 - self.min_gate) * channel
        selected_topo = topo * spatial * channel
        residual = self.residual_proj(selected_topo)
        modulation = torch.tanh(self.modulation_head(context))
        return residual, modulation, spatial


class TopoGate(nn.Module):
    """TopoGate implementation with spatial-channel terrain selection.

    RGB and terrain features jointly produce spatial and channel selection
    coefficients. Selected terrain residuals and bounded RGB modulation are
    added while the RGB identity path is preserved.
    """

    def __init__(
        self,
        topo_channels: tuple[int, ...],
        rgb_channels: tuple[int, ...],
        inject_layers: tuple[int, ...] = (2, 3),
        reduction: int = 4,
        residual_init: float = 0.15,
        modulation_init: float = 0.05,
        min_gate: float = 0.05,
        use_identity: bool = True,
        use_residual: bool = True,
        use_modulation: bool = True,
    ):
        super().__init__()
        self.inject_layers = set(inject_layers)
        self.blocks = nn.ModuleList(
            [
                _TopoGateBlock(
                    rgb_channels=rc,
                    topo_channels=tc,
                    reduction=reduction,
                    min_gate=min_gate,
                )
                for rc, tc in zip(rgb_channels, topo_channels)
            ]
        )
        self.residual_scale = nn.Parameter(torch.full((len(rgb_channels),), float(residual_init)))
        self.modulation_scale = nn.Parameter(torch.full((len(rgb_channels),), float(modulation_init)))
        self.use_identity = bool(use_identity)
        self.use_residual = bool(use_residual)
        self.use_modulation = bool(use_modulation)

    def forward(self, rgb_feats: list[torch.Tensor], topo_feats: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        outs = []
        gates = []
        for i, feat in enumerate(rgb_feats):
            if i not in self.inject_layers:
                outs.append(feat)
                continue
            topo = topo_feats[i]
            if topo.shape[-2:] != feat.shape[-2:]:
                topo = F.interpolate(topo, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            residual, modulation, spatial = self.blocks[i](feat, topo)
            fused = torch.zeros_like(feat)
            if self.use_identity:
                fused = fused + feat
            if self.use_residual:
                fused = fused + self.residual_scale[i] * residual
            if self.use_modulation:
                fused = fused + self.modulation_scale[i] * feat * modulation
            outs.append(fused)
            gates.append(spatial)
        return outs, gates
