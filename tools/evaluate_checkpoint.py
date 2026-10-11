from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from icassp_topogate.config import load_config
from icassp_topogate.data.dataset import PatchManifestDataset
from icassp_topogate.metrics.segmentation import boundary_f1, class_metrics_from_confusion, confusion_from_logits
from icassp_topogate.models.factory import build_model
from icassp_topogate.train.terrain_corruption import (
    EVAL_DEGRADATION_CONDITIONS,
    apply_eval_topography_degradation,
)


def count_parameters(model) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_M": total / 1e6, "trainable_M": trainable / 1e6}


def print_table(metrics: dict[str, dict[str, float]]) -> None:
    print("class        iou      dice     precision recall   accuracy pred_ratio target_ratio", flush=True)
    for name in ("background", "landslide", "mean"):
        row = metrics[name]
        pred_ratio = row.get("pred_ratio", row.get("pred_landslide_ratio"))
        target_ratio = row.get("target_ratio", row.get("target_landslide_ratio"))
        print(
            f"{name:<12}"
            f"{row.get('iou', float('nan')):>8.4f} "
            f"{row.get('dice', float('nan')):>8.4f} "
            f"{row.get('precision', float('nan')):>9.4f} "
            f"{row.get('recall', float('nan')):>7.4f} "
            f"{row.get('accuracy', float('nan')):>8.4f} "
            f"{pred_ratio if pred_ratio is not None else float('nan'):>10.6f} "
            f"{target_ratio if target_ratio is not None else float('nan'):>12.6f}",
            flush=True,
        )


def update_gate_statistics(accumulators: list[dict[str, float]], gates: list[torch.Tensor]) -> None:
    while len(accumulators) < len(gates):
        accumulators.append(
            {
                "count": 0.0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
                "near_lower_count": 0.0,
                "near_upper_count": 0.0,
            }
        )
    for index, gate in enumerate(gates):
        values = gate.detach().float()
        stats = accumulators[index]
        stats["count"] += float(values.numel())
        stats["sum"] += float(values.sum().item())
        stats["sum_sq"] += float(values.square().sum().item())
        stats["min"] = min(stats["min"], float(values.min().item()))
        stats["max"] = max(stats["max"], float(values.max().item()))
        stats["near_lower_count"] += float((values <= 0.06).sum().item())
        stats["near_upper_count"] += float((values >= 0.99).sum().item())


def finalize_gate_statistics(accumulators: list[dict[str, float]]) -> list[dict[str, float]]:
    rows = []
    for index, stats in enumerate(accumulators):
        count = max(stats["count"], 1.0)
        mean = stats["sum"] / count
        variance = max(stats["sum_sq"] / count - mean * mean, 0.0)
        rows.append(
            {
                "gate_index": index,
                "count": int(stats["count"]),
                "mean": mean,
                "std": variance ** 0.5,
                "min": stats["min"],
                "max": stats["max"],
                "near_lower_fraction": stats["near_lower_count"] / count,
                "near_upper_fraction": stats["near_upper_count"] / count,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--topo-corruption", choices=EVAL_DEGRADATION_CONDITIONS, default="clean")
    parser.add_argument("--corruption-seed", type=int, default=2026)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    ignore_index = loss_cfg.get("ignore_index", 255)
    device = torch.device(args.device)

    build_cfg = copy.deepcopy(cfg)
    # A formal checkpoint contains the complete backbone. Build offline and
    # then load strictly so evaluation never depends on Hugging Face access.
    build_cfg["backbone"]["pretrained"] = False
    model = build_model(build_cfg).to(device)
    model.cfg = cfg
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    torch.manual_seed(args.corruption_seed)

    ds = PatchManifestDataset(args.manifest, topo_indices=data_cfg.get("topo_indices"))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    total = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    bf1_values: list[float] = []
    gate_accumulators: list[dict[str, float]] = []
    degradation_details: dict[str, object] = {"condition": args.topo_corruption}
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            topo = batch.get("topo")
            if topo is not None:
                topo = topo.to(device, non_blocking=True)
                topo, degradation_details = apply_eval_topography_degradation(
                    topo,
                    args.topo_corruption,
                    channel_names=data_cfg.get("topo_channel_names"),
                )
            mask = batch["mask"].to(device, non_blocking=True)
            model_output = model(rgb, topo)
            logits = model_output["logits"]
            update_gate_statistics(gate_accumulators, model_output.get("gates") or [])
            c = confusion_from_logits(logits, mask, threshold=args.threshold, ignore_index=ignore_index)
            for key, value in c.items():
                total[key] += value
            bf1_values.append(boundary_f1(logits, (mask >= 0.5).float(), threshold=args.threshold))

    metrics = class_metrics_from_confusion(total)
    result = {
        "config": args.config,
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection_metric": ckpt.get("selection_metric"),
        "checkpoint_selection_value": ckpt.get("selection_value"),
        "topo_indices": data_cfg.get("topo_indices"),
        "params": count_parameters(model),
        "threshold": args.threshold,
        "topo_corruption": args.topo_corruption,
        "degradation_details": degradation_details,
        "corruption_seed": args.corruption_seed,
        "metrics_by_class": metrics,
        "mIoU": metrics["mean"]["iou"],
        "boundary_f1": float(sum(bf1_values) / max(1, len(bf1_values))),
        "gate_statistics": finalize_gate_statistics(gate_accumulators),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EVAL] config={args.config} checkpoint={args.checkpoint}", flush=True)
    print_table(metrics)
    print(f"boundary_f1 {result['boundary_f1']:.6f}", flush=True)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
