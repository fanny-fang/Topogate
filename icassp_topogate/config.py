from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import config: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "CONFIG"):
            raise AttributeError(f"{path} must define CONFIG")
        return dict(module.CONFIG)
    raise ValueError(f"Unsupported config format: {path.suffix}")
