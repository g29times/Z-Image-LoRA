from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class TinyImageModel(nn.Module):
    def __init__(self, in_ch: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, in_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LoraTrainingModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.training_cfg = cfg.get("training", {})

        # v0: tiny/dummy 冒烟实现，用于先验证数据管线、训练循环、artifact 输出。
        # 真实基模实现通过 model.impl 选择（例如 zimage_turbo），见 zimage_lora/models/。
        self.model = TinyImageModel()

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x = batch["pixel_values"]
        y = self.model(x)
        loss = F.mse_loss(y, x)

        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = float(self.training_cfg.get("learning_rate", 1e-4))
        opt_type = str((self.training_cfg.get("optimizer", {}) or {}).get("type", "adamw")).lower()

        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        sched = str(self.training_cfg.get("scheduler", "cosine")).lower()
        if sched == "cosine":
            max_steps = int(self.training_cfg.get("max_steps", 2000))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

        return optimizer

    def export_lora_weights(self) -> Dict[str, torch.Tensor]:
        # v0: 仅供 tiny/dummy 实现的占位导出。
        # 真实 LoRA 导出由具体模型模块提供（例如 save_lora()）。
        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        return state
