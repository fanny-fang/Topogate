from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache and verify a HuggingFace SegFormer backbone before DDP starts."
    )
    parser.add_argument("--model", default="nvidia/mit-b1")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--offline-only", action="store_true")
    return parser.parse_args()


def load_model(model_name: str, cache_dir: Path, local_files_only: bool):
    from transformers import SegformerModel

    return SegformerModel.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
        use_safetensors=False,
    )


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded = False
    try:
        model = load_model(args.model, cache_dir, local_files_only=True)
    except (OSError, RuntimeError):
        if args.offline_only:
            raise
        model = load_model(args.model, cache_dir, local_files_only=False)
        downloaded = True

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()

    # A second offline load proves that later DDP ranks will not need network access.
    verified = load_model(args.model, cache_dir, local_files_only=True)
    del verified
    gc.collect()

    print(
        json.dumps(
            {
                "status": "ready",
                "model": args.model,
                "cache_dir": str(cache_dir),
                "downloaded": downloaded,
                "offline_verified": True,
                "parameter_count": parameter_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
