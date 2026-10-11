from __future__ import annotations

import torch
import torch.nn.functional as F


def _valid_mask(target: torch.Tensor, ignore_index: int | None) -> torch.Tensor:
    if ignore_index is None:
        return torch.ones_like(target, dtype=torch.bool)
    return target != float(ignore_index)


def _target_binary(target: torch.Tensor) -> torch.Tensor:
    return (target >= 0.5).float()


def dice_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    ignore_index: int | None = 255,
) -> torch.Tensor:
    valid = _valid_mask(target, ignore_index)
    target = torch.where(valid, _target_binary(target), torch.zeros_like(target))
    prob = torch.sigmoid(logits) * valid.float()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def focal_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int | None = 255,
) -> torch.Tensor:
    valid = _valid_mask(target, ignore_index)
    target_bin = _target_binary(target)
    if not valid.any():
        return logits.sum() * 0.0
    logits_v = logits[valid]
    target_v = target_bin[valid]
    bce = F.binary_cross_entropy_with_logits(logits_v, target_v, reduction="none")
    prob = torch.sigmoid(logits_v)
    p_t = prob * target_v + (1 - prob) * (1 - target_v)
    alpha_t = alpha * target_v + (1 - alpha) * (1 - target_v)
    return (alpha_t * (1 - p_t).pow(gamma) * bce).mean()


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=x.device, dtype=x.dtype)
    kx = kx.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    ky = ky.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def boundary_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None = 255,
) -> torch.Tensor:
    valid = _valid_mask(target, ignore_index)
    target = torch.where(valid, _target_binary(target), torch.zeros_like(target))
    prob = torch.sigmoid(logits) * valid.float()
    pred_edge = _sobel_magnitude(prob)
    target_edge = _sobel_magnitude(target)
    return F.smooth_l1_loss(pred_edge, target_edge)


def tversky_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float | None = None,
    eps: float = 1e-6,
    ignore_index: int | None = 255,
) -> torch.Tensor:
    valid = _valid_mask(target, ignore_index)
    target = torch.where(valid, _target_binary(target), torch.zeros_like(target))
    prob = torch.sigmoid(logits) * valid.float()
    dims = tuple(range(1, prob.ndim))
    tp = (prob * target).sum(dim=dims)
    fp = (prob * (1 - target)).sum(dim=dims)
    fn = ((1 - prob) * target).sum(dim=dims)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = 1 - score
    if gamma is not None:
        loss = loss.pow(gamma)
    return loss.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    focal_weight: float = 0.0,
    tversky_weight: float = 0.0,
    focal_tversky_weight: float = 0.0,
    boundary_weight: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_tversky_gamma: float = 1.33,
    ignore_index: int | None = 255,
) -> torch.Tensor:
    valid = _valid_mask(target, ignore_index)
    target_bin = _target_binary(target)
    loss = logits.sum() * 0.0
    if bce_weight:
        if valid.any():
            bce = F.binary_cross_entropy_with_logits(logits[valid], target_bin[valid])
        else:
            bce = logits.sum() * 0.0
        loss = loss + bce_weight * bce
    if dice_weight:
        loss = loss + dice_weight * dice_loss_with_logits(logits, target, ignore_index=ignore_index)
    if focal_weight:
        loss = loss + focal_weight * focal_loss_with_logits(
            logits,
            target,
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=ignore_index,
        )
    if tversky_weight:
        loss = loss + tversky_weight * tversky_loss_with_logits(
            logits,
            target,
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_index=ignore_index,
        )
    if focal_tversky_weight:
        loss = loss + focal_tversky_weight * tversky_loss_with_logits(
            logits,
            target,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=focal_tversky_gamma,
            ignore_index=ignore_index,
        )
    if boundary_weight:
        loss = loss + boundary_weight * boundary_loss_with_logits(logits, target, ignore_index=ignore_index)
    return loss


def gate_l2_loss(gates: list[torch.Tensor] | None) -> torch.Tensor:
    if not gates:
        return torch.tensor(0.0)
    return torch.stack([g.pow(2).mean() for g in gates]).sum()


def region_expert_advantage_target(
    maf_logits: torch.Tensor,
    topogate_logits: torch.Tensor,
    target: torch.Tensor,
    region_size: int = 32,
    ignore_index: int | None = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return train-only TopoGate-better labels and a mask that excludes exact ties."""

    if maf_logits.shape != topogate_logits.shape or maf_logits.shape != target.shape:
        raise ValueError(
            "MAF, TopoGate, and target tensors must have identical shapes, got "
            f"{maf_logits.shape}, {topogate_logits.shape}, and {target.shape}."
        )
    if target.shape[-2] % region_size or target.shape[-1] % region_size:
        raise ValueError(
            f"Target shape {tuple(target.shape[-2:])} is not divisible by "
            f"region_size={region_size}."
        )

    valid = _valid_mask(target, ignore_index)
    truth = target >= 0.5
    maf_prediction = maf_logits >= 0
    topogate_prediction = topogate_logits >= 0

    def region_sum(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            value.float(),
            kernel_size=region_size,
            stride=region_size,
        ) * float(region_size * region_size)

    truth_count = region_sum(truth & valid)
    maf_tp = region_sum(maf_prediction & truth & valid)
    maf_fp = region_sum(maf_prediction & ~truth & valid)
    maf_fn = truth_count - maf_tp
    topogate_tp = region_sum(topogate_prediction & truth & valid)
    topogate_fp = region_sum(topogate_prediction & ~truth & valid)
    topogate_fn = truth_count - topogate_tp

    maf_denominator = maf_tp + maf_fp + maf_fn
    topogate_denominator = topogate_tp + topogate_fp + topogate_fn
    positive_region = truth_count > 0
    topogate_better = positive_region & (
        topogate_tp * maf_denominator > maf_tp * topogate_denominator
    )
    maf_better = positive_region & (
        maf_tp * topogate_denominator > topogate_tp * maf_denominator
    )

    background_region = ~positive_region
    topogate_better |= background_region & (topogate_fp < maf_fp)
    maf_better |= background_region & (maf_fp < topogate_fp)

    unresolved = ~(topogate_better | maf_better)
    maf_errors = maf_fp + maf_fn
    topogate_errors = topogate_fp + topogate_fn
    topogate_better |= unresolved & (topogate_errors < maf_errors)
    maf_better |= unresolved & (maf_errors < topogate_errors)
    supervised = topogate_better | maf_better
    return topogate_better.float(), supervised.float()


def region_router_losses(
    router_logits: torch.Tensor,
    a: torch.Tensor,
    maf_logits: torch.Tensor,
    topogate_logits: torch.Tensor,
    target: torch.Tensor,
    region_size: int = 32,
    ignore_index: int | None = 255,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        route_target, route_mask = region_expert_advantage_target(
            maf_logits,
            topogate_logits,
            target,
            region_size=region_size,
            ignore_index=ignore_index,
        )
    if router_logits.shape != route_target.shape:
        raise ValueError(
            f"Router grid {router_logits.shape} does not match regional target "
            f"{route_target.shape}."
        )

    raw_route = F.binary_cross_entropy_with_logits(
        router_logits,
        route_target,
        reduction="none",
    )
    route_loss = (raw_route * route_mask).sum() / route_mask.sum().clamp_min(1.0)
    vertical_tv = (
        (a[:, :, 1:] - a[:, :, :-1]).abs().mean()
        if a.shape[-2] > 1
        else a.sum() * 0.0
    )
    horizontal_tv = (
        (a[:, :, :, 1:] - a[:, :, :, :-1]).abs().mean()
        if a.shape[-1] > 1
        else a.sum() * 0.0
    )
    total_variation = vertical_tv + horizontal_tv
    deviation = a.abs().mean()
    with torch.no_grad():
        prediction = router_logits >= 0
        correct = ((prediction == (route_target >= 0.5)) * route_mask.bool()).sum()
        route_accuracy = correct.float() / route_mask.sum().clamp_min(1.0)
        supervised_fraction = route_mask.mean()
        topogate_target_fraction = (
            (route_target * route_mask).sum() / route_mask.sum().clamp_min(1.0)
        )
    return {
        "route": route_loss,
        "tv": total_variation,
        "deviation": deviation,
        "route_accuracy": route_accuracy,
        "route_supervised_fraction": supervised_fraction,
        "route_topogate_target_fraction": topogate_target_fraction,
    }
