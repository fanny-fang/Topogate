from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn


BACKBONE_DIMS = {
    "segformer_b1": (64, 128, 320, 512),
    "swin_tiny": (96, 192, 384, 768),
    "convnext_tiny": (96, 192, 384, 768),
    "resnet50": (256, 512, 1024, 2048),
}

TIMM_NAMES = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "convnext_tiny": "convnext_tiny",
    "resnet50": "resnet50",
}

HF_SEGFORMER_NAMES = {
    "segformer_b1": "nvidia/mit-b1",
}

SEGFORMER_CONFIGS = {
    "segformer_b1": {
        "num_channels": 3,
        "hidden_sizes": [64, 128, 320, 512],
        "depths": [2, 2, 2, 2],
        "num_attention_heads": [1, 2, 5, 8],
        "sr_ratios": [8, 4, 2, 1],
        "patch_sizes": [7, 3, 3, 3],
        "strides": [4, 2, 2, 2],
        "mlp_ratios": [4, 4, 4, 4],
    },
}


class ConvStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleHierarchicalBackbone(nn.Module):
    """Fallback feature extractor for smoke tests only.

    This is not a formal SegFormer/Swin/ConvNeXt implementation. Formal paper
    experiments should set ``allow_fallback=False`` in the config so missing
    official backbones fail loudly instead of silently changing the model.
    """

    def __init__(self, in_channels: int, dims: tuple[int, int, int, int]):
        super().__init__()
        stages = []
        ch = in_channels
        for dim in dims:
            stages.append(ConvStage(ch, dim))
            ch = dim
        self.stages = nn.ModuleList(stages)
        self.out_channels = dims

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class HFSegformerBackbone(nn.Module):
    def __init__(
        self,
        kind: str,
        in_channels: int = 3,
        pretrained: bool = True,
        extra_channel_init: str = "mean",
    ):
        super().__init__()
        from transformers import SegformerConfig, SegformerModel

        model_name = HF_SEGFORMER_NAMES[kind]
        if pretrained:
            source = os.environ.get("ICASSP_SEGFORMER_PRETRAINED_SOURCE", model_name)
            local_files_only = os.environ.get("ICASSP_HF_LOCAL_FILES_ONLY", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            self.model = SegformerModel.from_pretrained(
                source,
                local_files_only=local_files_only,
                use_safetensors=False,
            )
        else:
            cfg = SEGFORMER_CONFIGS[kind].copy()
            self.model = SegformerModel(SegformerConfig(**cfg))
        if in_channels != int(self.model.config.num_channels):
            self._adapt_first_patch_embedding(in_channels, extra_channel_init)
            self.model.config.num_channels = in_channels
        self.model.config.output_hidden_states = True
        self.out_channels = tuple(int(v) for v in self.model.config.hidden_sizes)

    def _first_patch_embedding(self):
        if hasattr(self.model, "stages"):
            return self.model.stages[0].patch_embeddings
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "patch_embeddings"):
            return self.model.encoder.patch_embeddings[0]
        raise RuntimeError("Cannot locate SegFormer first patch embedding module.")

    def _adapt_first_patch_embedding(self, in_channels: int, extra_channel_init: str) -> None:
        patch0 = self._first_patch_embedding()
        old_proj = getattr(patch0, "proj", None)
        if old_proj is None or not isinstance(old_proj, nn.Conv2d):
            raise RuntimeError("Cannot locate SegFormer first patch embedding projection for channel adaptation.")
        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            dilation=old_proj.dilation,
            groups=old_proj.groups,
            bias=old_proj.bias is not None,
            padding_mode=old_proj.padding_mode,
        )
        with torch.no_grad():
            new_proj.weight.zero_()
            copy_channels = min(in_channels, old_proj.in_channels)
            new_proj.weight[:, :copy_channels] = old_proj.weight[:, :copy_channels]
            if in_channels > old_proj.in_channels:
                if extra_channel_init not in {"zero", "mean", "mean_scaled"}:
                    raise ValueError(
                        "extra_channel_init must be 'zero', 'mean', or 'mean_scaled'; "
                        f"got {extra_channel_init!r}."
                    )
                extra = old_proj.weight.mean(dim=1, keepdim=True)
                if extra_channel_init == "zero":
                    extra.zero_()
                elif extra_channel_init == "mean_scaled":
                    extra.mul_(old_proj.in_channels / float(in_channels))
                new_proj.weight[:, old_proj.in_channels :] = extra.repeat(1, in_channels - old_proj.in_channels, 1, 1)
            if old_proj.bias is not None and new_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        patch0.proj = new_proj

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = self.model(pixel_values=x, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("SegFormer did not return hidden states.")
        feats = list(hidden_states[-4:])
        return feats


class TimmFeaturesBackbone(nn.Module):
    def __init__(self, name: str, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        import timm

        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.out_channels = tuple(info["num_chs"] for info in self.model.feature_info)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.model(x))


def build_backbone(
    kind: str,
    in_channels: int = 3,
    pretrained: bool = True,
    allow_fallback: bool = True,
    extra_channel_init: str = "mean",
) -> nn.Module:
    dims = BACKBONE_DIMS.get(kind, BACKBONE_DIMS["segformer_b1"])
    if kind in HF_SEGFORMER_NAMES:
        try:
            return HFSegformerBackbone(
                kind,
                in_channels=in_channels,
                pretrained=pretrained,
                extra_channel_init=extra_channel_init,
            )
        except Exception as exc:
            message = (
                f"Failed to build official HuggingFace SegFormer backbone kind={kind!r}, "
                f"pretrained={pretrained}, in_channels={in_channels}. Install transformers and ensure "
                f"the pretrained weights {HF_SEGFORMER_NAMES[kind]!r} are reachable or cached. "
                "For paper-closure runs, execute tools/prepare_hf_backbone.py first and set "
                "ICASSP_HF_LOCAL_FILES_ONLY=1."
            )
            if not allow_fallback:
                raise RuntimeError(message) from exc
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    timm_name = TIMM_NAMES.get(kind)
    if timm_name:
        try:
            return TimmFeaturesBackbone(timm_name, in_channels=in_channels, pretrained=pretrained)
        except Exception as exc:
            message = (
                f"Failed to build official timm backbone kind={kind!r}, timm_name={timm_name!r}, "
                f"pretrained={pretrained}, in_channels={in_channels}."
            )
            if not allow_fallback:
                raise RuntimeError(message) from exc
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    elif not allow_fallback:
        raise RuntimeError(f"No official backbone mapping is registered for kind={kind!r}.")
    return SimpleHierarchicalBackbone(in_channels=in_channels, dims=dims)
