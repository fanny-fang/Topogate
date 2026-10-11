from __future__ import annotations

import torch
import torch.nn as nn

from icassp_topogate.models.backbones import build_backbone
from icassp_topogate.models.decoder import SegFormerLikeDecoder
from icassp_topogate.models.dual_branch_fuse import DualBranchFuse
from icassp_topogate.models.dsdf_fuse import DSDFFuse
from icassp_topogate.models.maf_attention_fuse import MAFStyleAttentionFuse
from icassp_topogate.models.topo_gate import TopoGate
from icassp_topogate.models.topo_stem import TopoStem


class TopoSegmentationModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        fusion_type = cfg["fusion"]["type"]
        self.fusion_type = fusion_type
        backbone_cfg = cfg["backbone"]
        in_channels = backbone_cfg.get("in_channels", 3)
        self.backbone = build_backbone(
            backbone_cfg.get("type", "segformer_b1"),
            in_channels=in_channels,
            pretrained=backbone_cfg.get("pretrained", True),
            allow_fallback=backbone_cfg.get("allow_fallback", True),
            extra_channel_init=backbone_cfg.get("extra_channel_init", "mean"),
        )
        rgb_channels = tuple(self.backbone.out_channels)
        self.topo_stem = None
        self.fuser = None
        if fusion_type in {"topogate", "dualbranch", "maf_attention", "dsdf"}:
            topo_in = len(cfg.get("topo_channels") or ["rel_dem", "slope", "curvature"])
            topo_channels = tuple(max(16, min(c // 4, 128)) for c in rgb_channels)
            self.topo_stem = TopoStem(in_channels=topo_in, channels=topo_channels)
            layers = tuple(cfg["fusion"].get("inject_layers", [0, 1, 2, 3]))
            if fusion_type == "topogate":
                self.fuser = TopoGate(
                    topo_channels=topo_channels,
                    rgb_channels=rgb_channels,
                    inject_layers=layers,
                    reduction=cfg["fusion"].get("reduction", 4),
                    residual_init=cfg["fusion"].get("residual_init", 0.15),
                    modulation_init=cfg["fusion"].get("modulation_init", 0.05),
                    min_gate=cfg["fusion"].get("min_gate", 0.05),
                    use_identity=cfg["fusion"].get("use_identity", True),
                    use_residual=cfg["fusion"].get("use_residual", True),
                    use_modulation=cfg["fusion"].get("use_modulation", True),
                )
            elif fusion_type == "dsdf":
                self.fuser = DSDFFuse(
                    topo_in_channels=topo_in,
                    rgb_channels=rgb_channels,
                    topo_channels=topo_channels,
                    early_hidden=cfg["fusion"].get("early_hidden", 32),
                    early_gain_max=cfg["fusion"].get("early_gain_max", 1.0),
                    late_stage=cfg["fusion"].get("late_stage", 3),
                    num_heads=cfg["fusion"].get("num_heads", 4),
                    late_residual=cfg["fusion"].get("late_residual", False),
                )
            elif fusion_type == "dualbranch":
                self.fuser = DualBranchFuse(rgb_channels=rgb_channels, topo_channels=topo_channels, inject_layers=layers)
            elif fusion_type == "maf_attention":
                self.fuser = MAFStyleAttentionFuse(
                    rgb_channels=rgb_channels,
                    topo_channels=topo_channels,
                    inject_layers=layers,
                    reduction=cfg["fusion"].get("reduction", 4),
                )
        dec = cfg.get("decoder", {})
        self.decoder = SegFormerLikeDecoder(
            in_channels=rgb_channels,
            decoder_channels=dec.get("channels", 256),
            num_classes=dec.get("num_classes", 1),
        )

    def forward(self, rgb: torch.Tensor, topo: torch.Tensor | None = None) -> dict[str, object]:
        output_size = rgb.shape[-2:]
        if self.fusion_type == "early":
            if topo is None:
                raise ValueError("early fusion requires topo")
            x = torch.cat([rgb, topo], dim=1)
            feats = self.backbone(x)
            gates = []
        elif self.fusion_type == "dsdf":
            if topo is None:
                raise ValueError("dsdf requires topo")
            guided_rgb, early_gate = self.fuser.guide_rgb(rgb, topo)
            feats = self.backbone(guided_rgb)
            topo_feats = self.topo_stem(topo)
            feats, late_gates = self.fuser(feats, topo_feats)
            gates = [early_gate, *late_gates]
        else:
            feats = self.backbone(rgb)
            gates = []
            if self.fuser is not None:
                if topo is None:
                    raise ValueError(f"{self.fusion_type} requires topo")
                topo_feats = self.topo_stem(topo)
                feats, gates = self.fuser(feats, topo_feats)
        logits = self.decoder(feats, output_size=output_size)
        return {"logits": logits, "gates": gates}

    def set_epoch_state(self, epoch: int, max_epochs: int) -> None:
        if hasattr(self.fuser, "set_epoch_state"):
            self.fuser.set_epoch_state(epoch, max_epochs)


def build_model(cfg: dict) -> TopoSegmentationModel:
    return TopoSegmentationModel(cfg)

