from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_MODES = (
    "gaussian_noise",
    "gaussian_blur",
    "local_mask",
    "channel_dropout",
    "shift",
    "local_spike",
    "local_flatten",
    "normalization",
)

EVAL_DEGRADATION_CONDITIONS = (
    "clean",
    "noise025",
    "noise050",
    "noise100",
    "zero",
    "shift8",
    "shift16",
    "lowres4",
    "lowres8",
    "drop_rel_dem",
    "drop_slope",
    "drop_curvature",
)


def _random_rectangle(height: int, width: int, device: torch.device) -> tuple[int, int, int, int]:
    rect_h = int(torch.randint(max(1, height // 8), max(2, height // 2), (1,), device=device).item())
    rect_w = int(torch.randint(max(1, width // 8), max(2, width // 2), (1,), device=device).item())
    y0 = int(torch.randint(0, max(1, height - rect_h + 1), (1,), device=device).item())
    x0 = int(torch.randint(0, max(1, width - rect_w + 1), (1,), device=device).item())
    return y0, y0 + rect_h, x0, x0 + rect_w


def _gaussian_blur(sample: torch.Tensor, kernel_size: int = 5, sigma: float = 1.2) -> torch.Tensor:
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=sample.device, dtype=sample.dtype)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d.expand(sample.shape[1], 1, kernel_size, kernel_size)
    return F.conv2d(sample, kernel, padding=radius, groups=sample.shape[1])


@torch.no_grad()
def apply_eval_topography_degradation(
    topo: torch.Tensor,
    condition: str,
    channel_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply one deterministic, validation-time DEM degradation condition."""

    if topo.ndim != 4:
        raise ValueError(f"Expected NCHW topography, got shape={tuple(topo.shape)}.")
    if condition not in EVAL_DEGRADATION_CONDITIONS:
        raise ValueError(
            f"Unknown evaluation degradation {condition!r}; "
            f"expected one of {EVAL_DEGRADATION_CONDITIONS}."
        )
    names = list(channel_names or [f"channel_{index}" for index in range(topo.shape[1])])
    if len(names) != topo.shape[1]:
        raise ValueError(f"channel_names has {len(names)} entries for {topo.shape[1]} channels.")
    degraded = topo.clone()
    details: dict[str, object] = {"condition": condition, "channel_names": names}
    if condition == "clean":
        return degraded, details
    if condition == "zero":
        degraded.zero_()
        return degraded, details
    if condition.startswith("noise"):
        factor = {"noise025": 0.25, "noise050": 0.50, "noise100": 1.00}[condition]
        spatial_std = degraded.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        degraded.add_(torch.randn_like(degraded) * spatial_std * factor)
        details["sigma_over_channel_std"] = factor
        return degraded, details
    if condition.startswith("shift"):
        pixels = int(condition.removeprefix("shift"))
        degraded = torch.roll(degraded, shifts=(pixels, pixels), dims=(-2, -1))
        degraded[..., :pixels, :] = 0.0
        degraded[..., :, :pixels] = 0.0
        details["shift_pixels_yx"] = [pixels, pixels]
        return degraded, details
    if condition.startswith("lowres"):
        factor = int(condition.removeprefix("lowres"))
        height, width = degraded.shape[-2:]
        low_size = (max(1, height // factor), max(1, width // factor))
        degraded = F.interpolate(degraded, size=low_size, mode="area")
        degraded = F.interpolate(degraded, size=(height, width), mode="bilinear", align_corners=False)
        details["downsample_factor"] = factor
        details["intermediate_size"] = list(low_size)
        return degraded, details
    if condition.startswith("drop_"):
        target_name = condition.removeprefix("drop_")
        if target_name not in names:
            raise ValueError(f"Cannot apply {condition}: available topography channels are {names}.")
        index = names.index(target_name)
        degraded[:, index].zero_()
        details["dropped_channel_index"] = index
        return degraded, details
    raise AssertionError(f"Unhandled evaluation degradation: {condition}")


@torch.no_grad()
def corrupt_topography(
    topo: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply per-sample terrain degradation and return a spatial reliability target."""

    probability = float(cfg.get("probability", 0.5))
    modes = tuple(cfg.get("modes") or DEFAULT_MODES)
    if not modes:
        return topo, torch.ones_like(topo[:, :1]), {}

    corrupted = topo.clone()
    reliability = torch.ones_like(topo[:, :1])
    target_values = cfg.get("target_reliability", {})
    mode_counts = {mode: torch.zeros((), device=topo.device) for mode in modes}
    batch, channels, height, width = topo.shape

    for index in range(batch):
        if torch.rand((), device=topo.device).item() >= probability:
            continue
        mode = modes[int(torch.randint(0, len(modes), (1,), device=topo.device).item())]
        mode_counts[mode] += 1.0
        sample = corrupted[index : index + 1]

        if mode == "gaussian_noise":
            sigma = 0.05 + 0.15 * torch.rand((), device=topo.device)
            sample.add_(torch.randn_like(sample) * sigma)
            default_target = (1.0 - sigma / 0.25).clamp(0.20, 0.80)
            reliability[index].fill_(float(target_values.get(mode, default_target.item())))
        elif mode == "gaussian_blur":
            corrupted[index : index + 1] = _gaussian_blur(sample)
            reliability[index].fill_(float(target_values.get(mode, 0.35)))
        elif mode in {"local_mask", "local_spike", "local_flatten"}:
            y0, y1, x0, x1 = _random_rectangle(height, width, topo.device)
            reliability[index, :, y0:y1, x0:x1] = 0.0
            if mode == "local_mask":
                sample[:, :, y0:y1, x0:x1] = 0.0
            elif mode == "local_spike":
                sample[:, :, y0:y1, x0:x1] += 2.0 * torch.randn_like(sample[:, :, y0:y1, x0:x1])
            else:
                mean_value = sample.mean(dim=(-2, -1), keepdim=True)
                sample[:, :, y0:y1, x0:x1] = mean_value
        elif mode in {"channel_dropout", "missing_curvature"}:
            channel = min(2, channels - 1) if mode == "missing_curvature" else int(
                torch.randint(0, channels, (1,), device=topo.device).item()
            )
            sample[:, channel].zero_()
            default_target = 1.0 - 1.0 / max(float(channels), 1.0)
            reliability[index].fill_(float(target_values.get(mode, default_target)))
        elif mode == "shift":
            shifts = tuple(cfg.get("shift_pixels", (1, 2, 4, 8)))
            shift = int(shifts[int(torch.randint(0, len(shifts), (1,), device=topo.device).item())])
            dim = -1 if torch.rand((), device=topo.device).item() < 0.5 else -2
            sign = -1 if torch.rand((), device=topo.device).item() < 0.5 else 1
            shifted = torch.roll(sample, shifts=sign * shift, dims=dim)
            if dim == -1:
                if sign > 0:
                    shifted[..., :shift] = 0.0
                else:
                    shifted[..., -shift:] = 0.0
            else:
                if sign > 0:
                    shifted[..., :shift, :] = 0.0
                else:
                    shifted[..., -shift:, :] = 0.0
            corrupted[index : index + 1] = shifted
            reliability[index].zero_()
        elif mode == "normalization":
            scale = 0.7 + 0.6 * torch.rand((), device=topo.device)
            offset = -0.2 + 0.4 * torch.rand((), device=topo.device)
            sample.mul_(scale).add_(offset)
            reliability[index].fill_(float(target_values.get(mode, 0.60)))
        else:
            raise ValueError(f"Unknown terrain corruption mode: {mode}")

    stats = {f"corrupt_{mode}": count / max(float(batch), 1.0) for mode, count in mode_counts.items()}
    stats["corrupt_fraction"] = (1.0 - reliability).mean()
    return corrupted, reliability, stats


def reliability_supervision_loss(
    gates: list[torch.Tensor],
    reliability_target: torch.Tensor,
) -> torch.Tensor:
    if not gates:
        return reliability_target.new_zeros(())
    losses = []
    for gate in gates:
        target = F.interpolate(reliability_target, size=gate.shape[-2:], mode="nearest")
        losses.append(F.binary_cross_entropy(gate.clamp(1e-5, 1.0 - 1e-5), target))
    return torch.stack(losses).mean()


def binary_kl_to_rgb(
    logits: torch.Tensor,
    rgb_logits: torch.Tensor,
    reliability_target: torch.Tensor,
) -> torch.Tensor:
    teacher = torch.sigmoid(rgb_logits.detach()).clamp(1e-5, 1.0 - 1e-5)
    student = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
    kl = teacher * (teacher.log() - student.log())
    kl = kl + (1.0 - teacher) * ((1.0 - teacher).log() - (1.0 - student).log())
    unreliable = 1.0 - reliability_target
    denominator = unreliable.sum().clamp_min(1.0)
    return (kl * unreliable).sum() / denominator
