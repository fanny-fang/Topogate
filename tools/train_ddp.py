from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from icassp_topogate.config import load_config
from icassp_topogate.data.dataset import PatchManifestDataset
from icassp_topogate.data.transforms import TrainTransform
from icassp_topogate.metrics.segmentation import class_metrics_from_confusion, confusion_from_logits
from icassp_topogate.models.factory import build_model
from icassp_topogate.train.terrain_corruption import (
    binary_kl_to_rgb,
    corrupt_topography,
    reliability_supervision_loss,
)
from icassp_topogate.train.losses import (
    gate_l2_loss,
    region_router_losses,
    segmentation_loss,
)
from icassp_topogate.utils.seed import set_seed


class DistributedEvalSampler(Sampler[int]):
    """Shard validation data without padding or duplicate samples."""

    def __init__(self, dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        if len(self.dataset) <= self.rank:
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.num_replicas + 1


def init_distributed(backend: str) -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_confusion(confusion: dict[str, float], device: torch.device, distributed: bool) -> dict[str, float]:
    values = torch.tensor(
        [confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {k: float(v) for k, v in zip(("tp", "fp", "fn", "tn"), values.tolist())}


def reduce_loss(loss_sum: float, sample_count: int, device: torch.device, distributed: bool) -> float:
    values = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    denom = max(float(values[1].item()), 1.0)
    return float(values[0].item() / denom)


def evaluate(model, loader, device: torch.device, distributed: bool, ignore_index: int | None, threshold: float):
    model.eval()
    total = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            topo = batch.get("topo")
            if topo is not None:
                topo = topo.to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(rgb, topo)["logits"]
            c = confusion_from_logits(logits, mask, threshold=threshold, ignore_index=ignore_index)
            for key, value in c.items():
                total[key] += value
    total = reduce_confusion(total, device, distributed)
    return class_metrics_from_confusion(total)


def selection_value(metrics: dict[str, dict[str, float]], metric_name: str) -> float:
    if "_" not in metric_name:
        raise ValueError(f"Selection metric should look like landslide_dice or mean_iou, got {metric_name}")
    group, key = metric_name.split("_", 1)
    if group not in metrics or key not in metrics[group]:
        raise KeyError(f"Metric {metric_name} not found in validation metrics")
    return float(metrics[group][key])


def print_metrics_table(
    epoch: int,
    train_loss: float,
    metrics: dict[str, dict[str, float]],
    metric_name: str,
    metric_value: float,
    train_aux: dict[str, float] | None = None,
):
    print(f"[VAL] epoch={epoch:03d} train_loss={train_loss:.6f} selection={metric_name}:{metric_value:.6f}", flush=True)
    if train_aux:
        route = {key: value for key, value in train_aux.items() if key.startswith("route_")}
        gradients = {key: value for key, value in train_aux.items() if key.startswith("grad_")}
        corruptions = {key: value for key, value in train_aux.items() if key.startswith("corrupt_")}
        fusion = {
            key: value
            for key, value in train_aux.items()
            if key not in route and key not in gradients and key not in corruptions
        }
        for label, values in (("ROUTE", route), ("GRAD", gradients), ("CORRUPT", corruptions)):
            if values:
                text = " ".join(f"{key}={value:.6f}" for key, value in sorted(values.items()))
                print(f"[{label}] epoch={epoch:03d} {text}", flush=True)
        for stage in ("s1_", "s2_", "s3_", "s4_"):
            values = {key: value for key, value in fusion.items() if key.startswith(stage)}
            if values:
                text = " ".join(f"{key}={value:.6f}" for key, value in sorted(values.items()))
                print(f"[FUSION] epoch={epoch:03d} stage={stage[:2]} {text}", flush=True)
        other = {key: value for key, value in fusion.items() if not key.startswith(("s1_", "s2_", "s3_", "s4_"))}
        if other:
            text = " ".join(f"{key}={value:.6f}" for key, value in sorted(other.items()))
            print(f"[AUX] epoch={epoch:03d} {text}", flush=True)
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


def set_model_epoch_state(model, epoch: int, max_epochs: int) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "set_epoch_state"):
        raw_model.set_epoch_state(epoch, max_epochs)


def _is_rc_controller_parameter(name: str) -> bool:
    return name.startswith(
        (
            "fuser.delta_projection.",
            "fuser.reliability_reduce.",
            "fuser.spatial_reliability.",
            "fuser.channel_reliability.",
            "fuser.lambda_logits",
        )
    )


def _is_rc_maf_parameter(name: str) -> bool:
    return name.startswith(
        (
            "fuser.topo_proj.",
            "fuser.spatial_attn.",
            "fuser.channel_attn.",
            "fuser.fuse.",
        )
    )


def _is_region_router_parameter(name: str) -> bool:
    return name.startswith(
        (
            "fuser.maf_projection.",
            "fuser.topogate_projection.",
            "fuser.maf_aux_head.",
            "fuser.topogate_aux_head.",
            "fuser.terrain_embedding.",
            "fuser.router.",
            "fuser.global_alpha",
        )
    )


def _is_region_expert_parameter(name: str) -> bool:
    return name.startswith(("fuser.maf.", "fuser.topogate."))


def _is_region_s34_expert_parameter(name: str) -> bool:
    maf_modules = ("topo_proj", "spatial_attn", "channel_attn", "fuse")
    if name.startswith("fuser.maf."):
        return any(
            name.startswith(f"fuser.maf.{module}.{index}.")
            for module in maf_modules
            for index in (2, 3)
        )
    return name.startswith(("fuser.topogate.blocks.2.", "fuser.topogate.blocks.3."))


def build_optimizer(model, train_cfg: dict, stage: str = "full"):
    base_lr = train_cfg.get("lr", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    multipliers = {
        "backbone": train_cfg.get("backbone_lr_mult", 1.0),
        "topo_stem": train_cfg.get("topo_lr_mult", 1.0),
        "fuser": train_cfg.get("fuser_lr_mult", 1.0),
        "decoder": train_cfg.get("decoder_lr_mult", 1.0),
    }
    groups: dict[float, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean_name = name.removeprefix("module.")
        if stage == "controller" and not _is_rc_controller_parameter(clean_name):
            continue
        if stage == "region_router" and not (
            _is_region_router_parameter(clean_name) or clean_name.startswith("decoder.")
        ):
            continue
        if stage == "region_finetune" and not (
            _is_region_router_parameter(clean_name)
            or _is_region_s34_expert_parameter(clean_name)
            or clean_name.startswith("decoder.")
        ):
            continue
        module_key = clean_name.split(".", 1)[0]
        if _is_region_router_parameter(clean_name):
            multiplier = train_cfg.get("region_router_lr_mult", multipliers["fuser"])
        elif _is_region_expert_parameter(clean_name):
            multiplier = train_cfg.get("region_expert_lr_mult", 0.1)
        elif clean_name.startswith("decoder.") and stage == "region_router":
            multiplier = train_cfg.get("region_warmup_decoder_lr_mult", 1.0)
        elif clean_name.startswith("decoder.") and stage == "region_finetune":
            multiplier = train_cfg.get("region_decoder_lr_mult", 0.1)
        elif _is_rc_controller_parameter(clean_name):
            multiplier = train_cfg.get("fuser_controller_lr_mult", multipliers["fuser"])
        elif _is_rc_maf_parameter(clean_name):
            multiplier = train_cfg.get("fuser_maf_lr_mult", multipliers["fuser"])
        else:
            multiplier = multipliers.get(module_key, 1.0)
        lr = base_lr * multiplier
        if lr not in groups:
            groups[lr] = {"params": [], "lr": lr, "weight_decay": weight_decay}
        groups[lr]["params"].append(param)
    if not groups:
        raise RuntimeError(f"No trainable parameters were selected for stage={stage}.")
    return torch.optim.AdamW(list(groups.values()), lr=base_lr, weight_decay=weight_decay)


def rebuild_optimizer_preserving_state(optimizer, model, train_cfg: dict, stage: str):
    rebuilt = build_optimizer(model, train_cfg, stage=stage)
    retained_ids = {id(param) for group in rebuilt.param_groups for param in group["params"]}
    preserved = 0
    for param, state in optimizer.state.items():
        if id(param) in retained_ids:
            rebuilt.state[param] = state
            preserved += 1
    return rebuilt, preserved


def load_initial_checkpoint(model, checkpoint_path: str, rank: int) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if is_main_process(rank):
        print(
            f"[INIT] checkpoint={checkpoint_path} loaded={len(compatible)} "
            f"missing={len(missing)} ignored={len(source) - len(compatible)} unexpected={len(unexpected)}",
            flush=True,
        )


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_source_state(
    source: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    source = {
        key.removeprefix("module."): value
        for key, value in source.items()
    }
    source_uses_stages = any(key.startswith("backbone.model.stages.") for key in source)
    target_uses_encoder = any(key.startswith("backbone.model.encoder.") for key in target)
    if not (source_uses_stages and target_uses_encoder):
        return source

    block_prefix_map = {
        "layernorm_before.": "layer_norm_1.",
        "attention.q_proj.": "attention.self.query.",
        "attention.k_proj.": "attention.self.key.",
        "attention.v_proj.": "attention.self.value.",
        "attention.o_proj.": "attention.output.dense.",
        "attention.sequence_reduction.sequence_reduction.": "attention.self.sr.",
        "attention.sequence_reduction.layer_norm.": "attention.self.layer_norm.",
        "layernorm_after.": "layer_norm_2.",
        "mlp.fc1.": "mlp.dense1.",
        "mlp.fc2.": "mlp.dense2.",
    }
    converted = {}
    pattern = re.compile(r"^backbone\.model\.stages\.(\d+)\.(.+)$")
    for key, value in source.items():
        match = pattern.match(key)
        if match is None:
            converted[key] = value
            continue
        stage = int(match.group(1))
        suffix = match.group(2)
        if suffix.startswith("patch_embeddings."):
            new_key = (
                f"backbone.model.encoder.patch_embeddings.{stage}."
                f"{suffix.removeprefix('patch_embeddings.')}"
            )
        elif suffix.startswith("layer_norm."):
            new_key = (
                f"backbone.model.encoder.layer_norm.{stage}."
                f"{suffix.removeprefix('layer_norm.')}"
            )
        else:
            block_match = re.match(r"^blocks\.(\d+)\.(.+)$", suffix)
            if block_match is None:
                raise KeyError(f"Unsupported SegFormer checkpoint key: {key}")
            block = int(block_match.group(1))
            block_suffix = block_match.group(2)
            mapped_suffix = block_suffix
            for source_prefix, target_prefix in block_prefix_map.items():
                if block_suffix.startswith(source_prefix):
                    mapped_suffix = (
                        target_prefix + block_suffix.removeprefix(source_prefix)
                    )
                    break
            new_key = (
                f"backbone.model.encoder.block.{stage}.{block}.{mapped_suffix}"
            )
        converted[new_key] = value
    return converted


def load_dual_parent_checkpoints(
    model,
    maf_checkpoint_path: str,
    topogate_checkpoint_path: str,
    cfg: dict,
    rank: int,
) -> None:
    maf_checkpoint = torch.load(maf_checkpoint_path, map_location="cpu")
    topogate_checkpoint = torch.load(topogate_checkpoint_path, map_location="cpu")
    maf_cfg = maf_checkpoint.get("cfg", {})
    topogate_cfg = topogate_checkpoint.get("cfg", {})
    expected_layers = list(cfg.get("fusion", {}).get("inject_layers", []))
    checks = (
        ("MAF", maf_cfg, "maf_attention"),
        ("TopoGate", topogate_cfg, "topogate"),
    )
    for label, parent_cfg, expected_type in checks:
        fusion_cfg = parent_cfg.get("fusion", {})
        if fusion_cfg.get("type") != expected_type:
            raise ValueError(
                f"{label} checkpoint fusion type is {fusion_cfg.get('type')}, "
                f"expected {expected_type}."
            )
        if list(fusion_cfg.get("inject_layers", [])) != expected_layers:
            raise ValueError(
                f"{label} inject_layers={fusion_cfg.get('inject_layers')} does not "
                f"match target {expected_layers}."
            )
        if list(parent_cfg.get("data", {}).get("topo_indices", [])) != [0, 1, 2]:
            raise ValueError(f"{label} checkpoint is not TOPO3.")

    target = model.state_dict()
    maf_source = _normalize_source_state(
        maf_checkpoint.get("model", maf_checkpoint),
        target,
    )
    topogate_source = _normalize_source_state(
        topogate_checkpoint.get("model", topogate_checkpoint),
        target,
    )
    updates: dict[str, torch.Tensor] = {}

    def map_tensor(source_key: str, target_key: str, value: torch.Tensor) -> None:
        if target_key not in target:
            raise KeyError(f"Parent tensor has no target: {source_key} -> {target_key}")
        if target[target_key].shape != value.shape:
            raise ValueError(
                f"Shape mismatch {source_key} -> {target_key}: "
                f"{value.shape} vs {target[target_key].shape}"
            )
        updates[target_key] = value

    common_prefixes = ("backbone.", "topo_stem.", "decoder.")
    common_count = 0
    maf_fuser_count = 0
    topogate_fuser_count = 0
    for key, value in maf_source.items():
        if key.startswith(common_prefixes):
            map_tensor(key, key, value)
            common_count += 1
        elif key.startswith("fuser."):
            map_tensor(key, f"fuser.maf.{key.removeprefix('fuser.')}", value)
            maf_fuser_count += 1
    for key, value in topogate_source.items():
        if key.startswith("fuser."):
            map_tensor(
                key,
                f"fuser.topogate.{key.removeprefix('fuser.')}",
                value,
            )
            topogate_fuser_count += 1
    if min(common_count, maf_fuser_count, topogate_fuser_count) == 0:
        raise RuntimeError(
            "Dual-parent initialization mapped an empty required component: "
            f"common={common_count}, maf={maf_fuser_count}, topogate={topogate_fuser_count}."
        )

    missing, unexpected = model.load_state_dict(updates, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected dual-parent keys: {unexpected[:8]}")
    required_missing = [
        key
        for key in target
        if key.startswith(
            (
                "backbone.",
                "topo_stem.",
                "decoder.",
                "fuser.maf.",
                "fuser.topogate.",
            )
        )
        and key in missing
    ]
    if required_missing:
        raise RuntimeError(
            f"Dual-parent initialization missed required tensors: {required_missing[:8]}"
        )
    cfg["_initialization"] = {
        "shared_source": "maf",
        "maf_checkpoint": str(Path(maf_checkpoint_path).resolve()),
        "maf_checkpoint_sha256": _sha256_file(maf_checkpoint_path),
        "maf_epoch": maf_checkpoint.get("epoch"),
        "topogate_checkpoint": str(Path(topogate_checkpoint_path).resolve()),
        "topogate_checkpoint_sha256": _sha256_file(topogate_checkpoint_path),
        "topogate_epoch": topogate_checkpoint.get("epoch"),
        "mapped_common_tensors": common_count,
        "mapped_maf_fuser_tensors": maf_fuser_count,
        "mapped_topogate_fuser_tensors": topogate_fuser_count,
    }
    if is_main_process(rank):
        print(
            "[DUAL_INIT] shared=MAF "
            f"common={common_count} maf_fuser={maf_fuser_count} "
            f"topogate_fuser={topogate_fuser_count} "
            f"new_tensors={len(missing)}",
            flush=True,
        )


def set_controller_warmup_mode(model) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.backbone.eval()
    if raw_model.topo_stem is not None:
        raw_model.topo_stem.eval()
    raw_model.decoder.eval()
    fuser = raw_model.fuser
    for name in ("topo_proj", "spatial_attn", "channel_attn", "fuse"):
        module = getattr(fuser, name, None)
        if module is not None:
            module.eval()


def set_region_stage_mode(model) -> None:
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.backbone.eval()
    if raw_model.topo_stem is not None:
        raw_model.topo_stem.eval()
    raw_model.fuser.maf.eval()
    raw_model.fuser.topogate.eval()


def set_stage_trainability(model, stage: str) -> None:
    for name, parameter in model.named_parameters():
        clean_name = name.removeprefix("module.")
        if stage == "controller":
            trainable = _is_rc_controller_parameter(clean_name)
        elif stage == "region_router":
            trainable = (
                _is_region_router_parameter(clean_name)
                or clean_name.startswith("decoder.")
            )
        elif stage == "region_finetune":
            trainable = (
                _is_region_router_parameter(clean_name)
                or _is_region_s34_expert_parameter(clean_name)
                or clean_name.startswith("decoder.")
            )
        else:
            trainable = True
        parameter.requires_grad_(trainable)


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def diagnostic_gradient_norms(model) -> dict[str, float]:
    raw_model = model.module if hasattr(model, "module") else model
    groups = {}
    if raw_model.topo_stem is not None:
        groups["topostem"] = list(raw_model.topo_stem.parameters())
    if hasattr(raw_model.fuser, "diagnostic_parameter_groups"):
        groups.update(raw_model.fuser.diagnostic_parameter_groups())
    return {f"grad_{name}": _gradient_norm(parameters) for name, parameters in groups.items()}


def response_statistics(
    response_maps: list[dict[str, torch.Tensor]],
    mask: torch.Tensor,
    ignore_index: int | None,
) -> dict[str, float]:
    stats = {}
    for item in response_maps:
        stage = int(item["stage"].item())
        target = torch.nn.functional.interpolate(mask.float(), size=item["before"].shape[-2:], mode="nearest")
        valid = torch.ones_like(target, dtype=torch.bool)
        if ignore_index is not None:
            valid = target != float(ignore_index)
        positive = valid & (target > 0.5)
        negative = valid & ~positive
        for name in ("before", "after"):
            value = item[name]
            if positive.any():
                stats[f"s{stage}_{name}_landslide"] = float(value[positive].mean().cpu())
            if negative.any():
                stats[f"s{stage}_{name}_background"] = float(value[negative].mean().cpu())
    return stats


def save_best_checkpoint(
    model,
    cfg: dict,
    work_dir: Path,
    epoch: int,
    metrics: dict[str, dict[str, float]],
    metric_name: str,
    metric_value: float,
) -> Path:
    raw_model = model.module if hasattr(model, "module") else model
    safe_metric = metric_name.replace("/", "_")
    best_path = work_dir / f"best_epoch_{epoch:03d}_{safe_metric}_{metric_value:.4f}.pt"
    checkpoint = {
        "epoch": epoch,
        "cfg": cfg,
        "metrics_by_class": metrics,
        "selection_metric": metric_name,
        "selection_value": metric_value,
        "model": raw_model.state_dict(),
    }
    torch.save(checkpoint, best_path)
    shutil.copyfile(best_path, work_dir / "best.pt")
    
    for old_path in work_dir.glob("best_epoch_*.pt"):
        if old_path != best_path:
            old_path.unlink()
    with (work_dir / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "epoch": epoch,
                "selection_metric": metric_name,
                "selection_value": metric_value,
                "best_checkpoint": best_path.name,
                "metrics_by_class": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--selection-metric", default="landslide_dice")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--maf-checkpoint", default=None)
    parser.add_argument("--topogate-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    distributed, rank, world_size, local_rank = init_distributed(args.backend)
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    ignore_index = loss_cfg.get("ignore_index", 255)
    train_ds = PatchManifestDataset(args.train_manifest, transform=TrainTransform(), topo_indices=data_cfg.get("topo_indices"))
    val_ds = PatchManifestDataset(args.val_manifest, topo_indices=data_cfg.get("topo_indices"))

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedEvalSampler(val_ds, num_replicas=world_size, rank=rank) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    dual_parent_requested = bool(args.maf_checkpoint or args.topogate_checkpoint)
    if dual_parent_requested:
        if not (args.maf_checkpoint and args.topogate_checkpoint):
            raise ValueError(
                "--maf-checkpoint and --topogate-checkpoint must be supplied together."
            )
        if args.init_checkpoint:
            raise ValueError(
                "--init-checkpoint cannot be combined with dual-parent initialization."
            )
        if cfg.get("fusion", {}).get("type") != "maf_topogate_region_soft":
            raise ValueError(
                "Dual-parent initialization is only valid for maf_topogate_region_soft."
            )
    build_cfg = copy.deepcopy(cfg)
    if dual_parent_requested:
        # Both checkpoints contain complete weights. Building offline avoids an
        # unnecessary Hugging Face request without changing the architecture.
        build_cfg["backbone"]["pretrained"] = False
    model = build_model(build_cfg).to(device)
    model.cfg = cfg
    if dual_parent_requested:
        load_dual_parent_checkpoints(
            model,
            args.maf_checkpoint,
            args.topogate_checkpoint,
            cfg,
            rank,
        )
    elif args.init_checkpoint:
        load_initial_checkpoint(model, args.init_checkpoint, rank)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    if is_main_process(rank):
        print(
            {
                "model": cfg["name"],
                "world_size": world_size,
                "per_gpu_batch_size": args.batch_size,
                "effective_batch_size": args.batch_size * world_size,
                "train_samples": len(train_ds),
                "val_samples": len(val_ds),
                "topo_indices": data_cfg.get("topo_indices"),
                "selection_metric": args.selection_metric,
            },
            flush=True,
        )
    if args.dry_run:
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    train_cfg = cfg.get("train", {})
    is_region_fusion = cfg.get("fusion", {}).get("type") == "maf_topogate_region_soft"
    controller_warmup_epochs = int(train_cfg.get("controller_warmup_epochs", 0))
    region_warmup_epochs = int(train_cfg.get("region_warmup_epochs", 0))
    if is_region_fusion:
        optimization_stage = (
            "region_router" if region_warmup_epochs > 0 else "region_finetune"
        )
    else:
        optimization_stage = "controller" if controller_warmup_epochs > 0 else "full"
    set_stage_trainability(model, optimization_stage)
    optimizer = build_optimizer(model, train_cfg, stage=optimization_stage)
    work_dir = Path(args.work_dir)
    metrics_file = work_dir / "metrics.jsonl"
    if is_main_process(rank):
        work_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    best_value = -1.0
    for epoch in range(1, args.epochs + 1):
        if is_region_fusion:
            next_stage = (
                "region_router"
                if epoch <= region_warmup_epochs
                else "region_finetune"
            )
        else:
            next_stage = "controller" if epoch <= controller_warmup_epochs else "full"
        if next_stage != optimization_stage:
            optimization_stage = next_stage
            set_stage_trainability(model, optimization_stage)
            optimizer, preserved_states = rebuild_optimizer_preserving_state(
                optimizer,
                model,
                train_cfg,
                stage=optimization_stage,
            )
            if is_main_process(rank):
                print(
                    f"[TRAIN_STAGE] epoch={epoch:03d} stage={optimization_stage} "
                    f"optimizer_rebuilt=1 preserved_states={preserved_states}",
                    flush=True,
                )
        set_model_epoch_state(model, epoch, args.epochs)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        if is_region_fusion:
            set_region_stage_mode(model)
        elif optimization_stage == "controller":
            set_controller_warmup_mode(model)
        loss_sum = 0.0
        sample_count = 0
        aux_sums: dict[str, float] = {}
        for batch in train_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            topo = batch.get("topo")
            if topo is not None:
                topo = topo.to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            reliability_target = None
            corruption_cfg = cfg.get("topo_corruption", {})
            corruption_stats = {}
            if topo is not None and corruption_cfg.get("enabled", False):
                topo, reliability_target, corruption_stats = corrupt_topography(topo, corruption_cfg)
            out = model(rgb, topo)
            loss = segmentation_loss(
                out["logits"],
                mask,
                bce_weight=loss_cfg.get("bce", 0.5),
                dice_weight=loss_cfg.get("dice", 0.5),
                focal_weight=loss_cfg.get("focal", 0.0),
                tversky_weight=loss_cfg.get("tversky", 0.0),
                focal_tversky_weight=loss_cfg.get("focal_tversky", 0.0),
                boundary_weight=loss_cfg.get("boundary", 0.0),
                focal_alpha=loss_cfg.get("focal_alpha", 0.25),
                focal_gamma=loss_cfg.get("focal_gamma", 2.0),
                tversky_alpha=loss_cfg.get("tversky_alpha", 0.3),
                tversky_beta=loss_cfg.get("tversky_beta", 0.7),
                focal_tversky_gamma=loss_cfg.get("focal_tversky_gamma", 1.33),
                ignore_index=ignore_index,
            )
            route_payload = out.get("route_payload", {})
            if route_payload.get("logits") is not None:
                expert_logits = out.get("route_expert_logits", {})
                if set(expert_logits) != {"maf", "topogate"}:
                    raise RuntimeError(
                        "Region route supervision requires complete MAF and "
                        "TopoGate auxiliary logits during training."
                    )
                route_terms = region_router_losses(
                    route_payload["logits"],
                    route_payload["a"],
                    expert_logits["maf"],
                    expert_logits["topogate"],
                    mask,
                    region_size=int(
                        cfg.get("fusion", {}).get("region_size", 32)
                    ),
                    ignore_index=ignore_index,
                )
                loss = (
                    loss
                    + float(loss_cfg.get("region_route", 0.0))
                    * route_terms["route"]
                    + float(loss_cfg.get("region_tv", 0.0))
                    * route_terms["tv"]
                    + float(loss_cfg.get("region_deviation", 0.0))
                    * route_terms["deviation"]
                )
                for key, value in route_terms.items():
                    log_key = (
                        "route_loss"
                        if key == "route"
                        else key
                        if key.startswith("route_")
                        else f"route_{key}"
                    )
                    aux_sums[log_key] = (
                        aux_sums.get(log_key, 0.0)
                        + float(value.detach().cpu()) * int(mask.shape[0])
                    )
            elif route_payload.get("a") is not None:
                deviation = route_payload["a"].abs().mean()
                loss = (
                    loss
                    + float(loss_cfg.get("region_deviation", 0.0)) * deviation
                )
                aux_sums["route_deviation"] = (
                    aux_sums.get("route_deviation", 0.0)
                    + float(deviation.detach().cpu()) * int(mask.shape[0])
                )
            if out.get("gates"):
                loss = loss + loss_cfg.get("gate_l2", 0.0) * gate_l2_loss(out["gates"]).to(device)
            aux_losses = out.get("aux_losses", {})
            if aux_losses:
                loss = loss + loss_cfg.get("route_balance", 0.0) * aux_losses.get("route_balance", loss.new_zeros(()))
                for key in ("route_balance", "route_entropy", "route_mean_max", "route_temperature"):
                    value = aux_losses.get(key)
                    if value is not None:
                        aux_sums[key] = aux_sums.get(key, 0.0) + float(value.detach().cpu()) * int(mask.shape[0])
            if reliability_target is not None:
                reliability_loss = reliability_supervision_loss(out.get("gates", []), reliability_target)
                reliability_weight = float(corruption_cfg.get("reliability_weight", 0.0))
                loss = loss + reliability_weight * reliability_loss
                aux_sums["reliability_loss"] = aux_sums.get("reliability_loss", 0.0) + float(
                    reliability_loss.detach().cpu()
                ) * int(mask.shape[0])
                consistency_weight = float(corruption_cfg.get("consistency_weight", 0.0))
                if consistency_weight > 0 and out.get("rgb_logits") is not None:
                    consistency_loss = binary_kl_to_rgb(out["logits"], out["rgb_logits"], reliability_target)
                    loss = loss + consistency_weight * consistency_loss
                    aux_sums["consistency_loss"] = aux_sums.get("consistency_loss", 0.0) + float(
                        consistency_loss.detach().cpu()
                    ) * int(mask.shape[0])
                for key, value in corruption_stats.items():
                    aux_sums[key] = aux_sums.get(key, 0.0) + float(value.detach().cpu()) * int(mask.shape[0])
            for key, value in out.get("fusion_diagnostics", {}).items():
                aux_sums[key] = aux_sums.get(key, 0.0) + float(value.detach().cpu()) * int(mask.shape[0])
            for key, value in response_statistics(out.get("response_maps", []), mask, ignore_index).items():
                aux_sums[key] = aux_sums.get(key, 0.0) + value * int(mask.shape[0])
            model.zero_grad(set_to_none=True)
            loss.backward()
            for key, value in diagnostic_gradient_norms(model).items():
                aux_sums[key] = aux_sums.get(key, 0.0) + value * int(mask.shape[0])
            optimizer.step()
            bs = int(mask.shape[0])
            loss_sum += float(loss.detach().cpu()) * bs
            sample_count += bs

        train_loss = reduce_loss(loss_sum, sample_count, device, distributed)
        train_aux = {key: reduce_loss(value, sample_count, device, distributed) for key, value in aux_sums.items()}
        metrics = evaluate(model, val_loader, device, distributed, ignore_index=ignore_index, threshold=args.threshold)
        current_value = selection_value(metrics, args.selection_metric)
        if is_main_process(rank):
            print_metrics_table(epoch, train_loss, metrics, args.selection_metric, current_value, train_aux=train_aux)
            with metrics_file.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_aux": train_aux,
                            "selection_metric": args.selection_metric,
                            "selection_value": current_value,
                            "metrics_by_class": metrics,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if current_value > best_value:
                best_value = current_value
                best_path = save_best_checkpoint(model, cfg, work_dir, epoch, metrics, args.selection_metric, current_value)
                print(f"[BEST] epoch={epoch:03d} checkpoint={best_path.name}", flush=True)
        if distributed:
            dist.barrier()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
