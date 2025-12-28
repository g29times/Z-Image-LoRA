from __future__ import annotations

from typing import Any, Dict

from zimage_lora.model import LoraTrainingModule


def create_lightning_module(cfg: Dict[str, Any]):
    model_cfg = cfg.get("model", {})
    impl = str(model_cfg.get("impl") or "tiny").lower()

    if impl in {"tiny", "dummy"}:
        return LoraTrainingModule(cfg)

    if impl in {"zimage_turbo", "zimage", "zimage-turbo"}:
        from zimage_lora.models.zimage_turbo import ZImageTurboLightningModule

        return ZImageTurboLightningModule(cfg)

    raise ValueError(f"Unknown model.impl: {impl}")
