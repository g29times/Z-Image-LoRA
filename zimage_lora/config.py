from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

import yaml


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

    bs = tr.get("batch_size", 1)
    if not isinstance(bs, int) or bs <= 0:
        raise ValueError("training.batch_size must be positive int")

    lr = tr.get("learning_rate", 1e-4)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError("training.learning_rate must be positive")

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
