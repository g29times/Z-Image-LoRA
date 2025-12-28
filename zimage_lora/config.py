from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import yaml


def _as_int(v: Any, name: str) -> int:
    if isinstance(v, bool):
        raise ValueError(f"{name} must be int")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError(f"{name} must be int")
        try:
            return int(float(s))
        except Exception as e:
            raise ValueError(f"{name} must be int") from e
    raise ValueError(f"{name} must be int")


def _as_float(v: Any, name: str) -> float:
    if isinstance(v, bool):
        raise ValueError(f"{name} must be float")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError(f"{name} must be float")
        try:
            return float(s)
        except Exception as e:
            raise ValueError(f"{name} must be float") from e
    raise ValueError(f"{name} must be float")


@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.raw.get("dataset", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})

    @property
    def job(self) -> Dict[str, Any]:
        return self.raw.get("job", {})

    @property
    def output(self) -> Dict[str, Any]:
        return self.raw.get("output", {})


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    cfg = Config(raw=data)
    validate_config(cfg)
    return cfg


def ensure_job_id(cfg: Config) -> str:
    job = cfg.raw.setdefault("job", {})
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        job_id = uuid.uuid4().hex
        job["id"] = job_id
    return job_id


def validate_config(cfg: Config) -> None:
    ds = cfg.dataset
    if ds.get("type") != "image_only_with_caption":
        raise ValueError("dataset.type must be image_only_with_caption")

    data_dir = ds.get("data_dir")
    if not data_dir:
        raise ValueError("dataset.data_dir is required")

    cap = ds.get("caption", {})
    if cap.get("strategy") not in {"filename_or_txt"}:
        raise ValueError("dataset.caption.strategy must be filename_or_txt")

    trig = cap.get("trigger_token")
    if not isinstance(trig, str) or not trig:
        raise ValueError("dataset.caption.trigger_token must be a non-empty string")

    tr = cfg.training
    if tr.get("backend") != "lightning":
        raise ValueError("training.backend must be lightning")

    # Normalize numeric fields (YAML may parse scientific notation as string depending on quoting)
    bs = _as_int(tr.get("batch_size", 1), "training.batch_size")
    if bs <= 0:
        raise ValueError("training.batch_size must be positive int")
    tr["batch_size"] = bs

    ga = _as_int(tr.get("gradient_accumulation", 1), "training.gradient_accumulation")
    if ga <= 0:
        raise ValueError("training.gradient_accumulation must be positive int")
    tr["gradient_accumulation"] = ga

    ms = _as_int(tr.get("max_steps", 2000), "training.max_steps")
    if ms <= 0:
        raise ValueError("training.max_steps must be positive int")
    tr["max_steps"] = ms

    ws = _as_int(tr.get("warmup_steps", 0), "training.warmup_steps")
    if ws < 0:
        raise ValueError("training.warmup_steps must be >= 0")
    tr["warmup_steps"] = ws

    nw = _as_int(tr.get("num_workers", 0), "training.num_workers")
    if nw < 0:
        raise ValueError("training.num_workers must be >= 0")
    tr["num_workers"] = nw

    lr = _as_float(tr.get("learning_rate", 1e-4), "training.learning_rate")
    if lr <= 0:
        raise ValueError("training.learning_rate must be positive")
    tr["learning_rate"] = lr

    out = cfg.output
    reg = out.get("registry", {})
    if reg.get("type") != "local":
        raise ValueError("output.registry.type must be local")
    if not reg.get("path"):
        raise ValueError("output.registry.path is required")


def resolve_artifact_dir(cfg: Config, job_id: str) -> Path:
    out = cfg.output
    reg_path = Path(out.get("registry", {}).get("path"))
    style_name = str(out.get("style_name") or "style")
    pattern = str(out.get("naming", {}).get("pattern") or "{job_id}_{style_name}_v0")
    run_name = pattern.format(job_id=job_id, style_name=style_name)
    return reg_path / run_name
