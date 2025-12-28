from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from zimage_lora.config import load_config, ensure_job_id, resolve_artifact_dir
from zimage_lora.data import ImageCaptionDataModule
from zimage_lora.models.registry import create_lightning_module
from zimage_lora.utils import dump_json, dump_yaml_text, now_ts, save_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(Path("configs") / "train_v0.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    job_id = ensure_job_id(cfg)
    artifact_dir = resolve_artifact_dir(cfg, job_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # 保存 config
    raw_text = Path(args.config).read_text(encoding="utf-8")
    if bool((cfg.output.get("save", {}) or {}).get("config", True)):
        dump_yaml_text(artifact_dir / "config.yaml", raw_text)

    # Lightning
    training_cfg = cfg.training
    seed = int(training_cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(artifact_dir / "checkpoints"),
            save_top_k=1,
            monitor=None,
            save_last=True,
            every_n_train_steps=int(training_cfg.get("log_every_n_steps", 10)),
        )
    )

    precision = str(training_cfg.get("mixed_precision", "fp16")).lower()
    if precision in {"fp16", "16", "16-mixed"}:
        pl_precision = "16-mixed"
    elif precision in {"bf16", "bf16-mixed"}:
        pl_precision = "bf16-mixed"
    else:
        pl_precision = "32-true"

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        precision=pl_precision,
        max_steps=int(training_cfg.get("max_steps", 2000)),
        accumulate_grad_batches=int(training_cfg.get("gradient_accumulation", 1)),
        log_every_n_steps=int(training_cfg.get("log_every_n_steps", 10)),
        val_check_interval=training_cfg.get("val_check_interval", 0),
        enable_checkpointing=True,
        default_root_dir=str(artifact_dir),
        callbacks=callbacks,
    )

    dm = ImageCaptionDataModule(cfg.dataset, training_cfg)
    model = create_lightning_module(cfg.raw)

    trainer.fit(model, datamodule=dm)

    # 保存 artifacts
    if bool((cfg.output.get("save", {}) or {}).get("training_meta", True)):
        dump_json(
            artifact_dir / "training_meta.json",
            {
                "job_id": job_id,
                "ended_at": now_ts(),
                "global_step": int(trainer.global_step),
            },
        )

    if bool((cfg.output.get("save", {}) or {}).get("lora_weights", True)):
        # 优先保存真实 LoRA adapter；否则走占位导出
        lora_dir = artifact_dir / "lora"
        if hasattr(model, "save_lora"):
            lora_dir.mkdir(parents=True, exist_ok=True)
            model.save_lora(str(lora_dir))
        else:
            state = model.export_lora_weights()
            save_state_dict(artifact_dir / "lora_weights.pt", state)


if __name__ == "__main__":
    main()
