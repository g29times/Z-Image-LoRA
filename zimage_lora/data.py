from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pytorch_lightning as pl


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _read_caption_for_image(img_path: Path, strategy: str) -> str:
    if strategy != "filename_or_txt":
        raise ValueError("Unsupported caption strategy")

    txt_path = img_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8").strip()

    return img_path.stem


class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        resolution: int,
        center_crop: bool,
        caption_strategy: str,
        trigger_token: str,
        repeats: int,
        augment: Dict[str, Any],
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        images = [p for p in self.data_dir.rglob("*") if p.is_file() and _is_image(p)]
        if not images:
            raise ValueError(f"No images found under: {self.data_dir}")

        self.images: List[Path] = images * max(1, int(repeats))
        self.caption_strategy = caption_strategy
        self.trigger_token = trigger_token

        tfms: List[Any] = []
        tfms.append(T.Resize(resolution, interpolation=T.InterpolationMode.BICUBIC))
        if center_crop:
            tfms.append(T.CenterCrop(resolution))
        else:
            tfms.append(T.RandomCrop(resolution))

        if bool(augment.get("horizontal_flip", False)):
            tfms.append(T.RandomHorizontalFlip(p=0.5))

        if bool(augment.get("color_jitter", False)):
            tfms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))

        tfms.append(T.ToTensor())
        tfms.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.transforms = T.Compose(tfms)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transforms(img)

        caption = _read_caption_for_image(img_path, self.caption_strategy)
        if self.trigger_token not in caption:
            caption = f"{self.trigger_token} {caption}".strip()

        return {
            "pixel_values": pixel_values,
            "caption": caption,
            "path": str(img_path),
        }


class ImageCaptionDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any], training_cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.training_cfg = training_cfg
        self._ds: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        cap = self.cfg.get("caption", {})
        self._ds = ImageCaptionDataset(
            data_dir=self.cfg["data_dir"],
            resolution=int(self.cfg.get("resolution", 1024)),
            center_crop=bool(self.cfg.get("center_crop", True)),
            caption_strategy=str(cap.get("strategy", "filename_or_txt")),
            trigger_token=str(cap.get("trigger_token", "<zstyle>")),
            repeats=int(self.cfg.get("repeats", 1)),
            augment=dict(self.cfg.get("augment", {}) or {}),
        )

    def train_dataloader(self) -> DataLoader:
        if self._ds is None:
            raise RuntimeError("DataModule not set up")
        return DataLoader(
            self._ds,
            batch_size=int(self.training_cfg.get("batch_size", 1)),
            shuffle=bool(self.cfg.get("shuffle", True)),
            num_workers=int(self.training_cfg.get("num_workers", 4)),
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate,
        )

    @staticmethod
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = torch.stack([x["pixel_values"] for x in batch], dim=0)
        captions = [x["caption"] for x in batch]
        paths = [x["path"] for x in batch]
        return {"pixel_values": pixel_values, "captions": captions, "paths": paths}
