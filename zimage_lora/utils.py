from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import time

import torch


def dump_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_yaml_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now_ts() -> int:
    return int(time.time())


def save_state_dict(path: str | Path, state: Dict[str, torch.Tensor]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
