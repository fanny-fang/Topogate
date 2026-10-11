from __future__ import annotations

import torch
import torch.nn.functional as F


def confusion_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    ignore_index: int | None = 255,
) -> dict[str, float]:
    pred = (torch.sigmoid(logits) >= threshold).float()
    if ignore_index is not None:
        valid = target != float(ignore_index)
        pred = pred[valid]
        target = target[valid]
    target = (target >= 0.5).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _binary_metrics(tp: float, fp: float, fn: float, tn: float, eps: float) -> dict[str, float]:
    total = tp + fp + fn + tn
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (total + eps)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "pred_pixels": tp + fp,
        "target_pixels": tp + fn,
        "pred_ratio": (tp + fp) / (total + eps),
        "target_ratio": (tp + fn) / (total + eps),
    }


def metrics_from_confusion(c: dict[str, float], eps: float = 1e-7) -> dict[str, float]:
    """Legacy foreground metrics kept for existing single-GPU scripts.

    The historical key ``miou`` in this scaffold actually stores landslide IoU.
    New experiment scripts should use ``class_metrics_from_confusion`` so the
    background, landslide, and mean rows are explicit.
    """
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    m = _binary_metrics(tp, fp, fn, tn, eps)
    return {
        "miou": m["iou"],
        "dice": m["dice"],
        "precision": m["precision"],
        "recall": m["recall"],
        "accuracy": m["accuracy"],
    }


def class_metrics_from_confusion(c: dict[str, float], eps: float = 1e-7) -> dict[str, dict[str, float]]:
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    background = _binary_metrics(tn, fn, fp, tp, eps)
    landslide = _binary_metrics(tp, fp, fn, tn, eps)
    mean = {
        key: (background[key] + landslide[key]) / 2
        for key in ("iou", "dice", "precision", "recall")
    }
    total = tp + fp + fn + tn
    mean["accuracy"] = (tp + tn) / (total + eps)
    mean["pred_landslide_ratio"] = (tp + fp) / (total + eps)
    mean["target_landslide_ratio"] = (tp + fn) / (total + eps)
    return {"background": background, "landslide": landslide, "mean": mean}


def binary_boundary(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    mask = (mask >= 0.5).float()
    eroded = 1 - F.max_pool2d(1 - mask, kernel_size=2 * radius + 1, stride=1, padding=radius)
    dilated = F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (dilated - eroded).clamp(0, 1)


def boundary_f1(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, radius: int = 1) -> float:
    pred = (torch.sigmoid(logits) >= threshold).float()
    pb = binary_boundary(pred, radius)
    tb = binary_boundary(target, radius)
    tp = (pb * tb).sum()
    precision = tp / (pb.sum() + 1e-7)
    recall = tp / (tb.sum() + 1e-7)
    return (2 * precision * recall / (precision + recall + 1e-7)).item()
