from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from icassp_topogate.config import load_config
from icassp_topogate.models.factory import build_model


def count_params(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile paper models with PyTorch operator FLOP counting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.out) if args.out else None
    if output_path and output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing profile: {output_path}")

    cfg = load_config(args.config)
    build_cfg = copy.deepcopy(cfg)
    build_cfg["backbone"]["pretrained"] = False
    device = torch.device(args.device)
    model = build_model(build_cfg).to(device).eval()
    rgb = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    topo_channels = len(cfg.get("topo_channels") or [])
    topo = None
    if cfg["fusion"]["type"] != "rgb":
        topo = torch.randn(args.batch_size, topo_channels, args.height, args.width, device=device)

    with torch.inference_mode():
        with FlopCounterMode(display=False) as flop_counter:
            output = model(rgb, topo)
        total_flops = int(flop_counter.get_total_flops())

        for _ in range(args.warmup):
            model(rgb, topo)
        synchronize(device)
        elapsed_seconds = None
        if args.repeats > 0:
            started = time.perf_counter()
            for _ in range(args.repeats):
                model(rgb, topo)
            synchronize(device)
            elapsed_seconds = time.perf_counter() - started

    parameters = count_params(model)
    report = {
        "model": cfg["name"],
        "config": str(Path(args.config).resolve()),
        "input": {"batch_size": args.batch_size, "rgb_channels": 3, "topo_channels": topo_channels, "height": args.height, "width": args.width},
        "parameters": {
            **parameters,
            "total_M": parameters["total"] / 1e6,
            "trainable_M": parameters["trainable"] / 1e6,
        },
        "flops": {"total": total_flops, "GFLOPs": total_flops / 1e9},
        "output_shape": list(output["logits"].shape),
        "num_gates": len(output.get("gates") or []),
        "device": str(device),
        "latency_ms_mean": None if elapsed_seconds is None else 1000.0 * elapsed_seconds / args.repeats,
        "profiling_definition": "One FP32 forward pass; multiply-add is counted according to torch.utils.flop_counter.",
    }
    if device.type == "cuda":
        report["peak_memory_MiB"] = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
