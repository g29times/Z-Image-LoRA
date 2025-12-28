from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class ZImageTurboLightningModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.get("model", {})
        self.base_cfg = (self.model_cfg.get("base", {}) or {})
        self.lora_cfg = (self.model_cfg.get("lora", {}) or {})
        self.training_cfg = cfg.get("training", {})

        self.pipe = None
        self.transformer = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None
        self.noise_scheduler = None

        self._init_pipeline_and_lora()

    def _init_pipeline_and_lora(self) -> None:
        from diffusers import ZImagePipeline
        from diffusers import DDPMScheduler

        name = str(self.base_cfg.get("name", "Tongyi-MAI/Z-Image-Turbo"))
        precision = str(self.base_cfg.get("precision", "bf16")).lower()
        if precision in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif precision in {"fp16", "float16"}:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Loading can be CPU-RAM heavy; default to low_cpu_mem_usage=True unless explicitly disabled.
        low_cpu_mem_usage = bool(self.base_cfg.get("low_cpu_mem_usage", True))
        self.pipe = ZImagePipeline.from_pretrained(
            name,
            dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        # Optional attention backend & offload (helps on low VRAM / low RAM machines)
        attn_backend = self.base_cfg.get("attention_backend")
        if attn_backend and hasattr(self.pipe, "transformer") and hasattr(self.pipe.transformer, "set_attention_backend"):
            self.pipe.transformer.set_attention_backend(attn_backend)

        cpu_offload = bool(self.base_cfg.get("cpu_offload", False))
        if cpu_offload and hasattr(self.pipe, "enable_model_cpu_offload"):
            self.pipe.enable_model_cpu_offload()
        else:
            # Keep pipeline on GPU for training
            self.pipe.to("cuda")

        # components
        self.transformer = getattr(self.pipe, "transformer", None)
        self.vae = getattr(self.pipe, "vae", None)
        self.text_encoder = getattr(self.pipe, "text_encoder", None)
        self.tokenizer = getattr(self.pipe, "tokenizer", None)
        self.scheduler = getattr(self.pipe, "scheduler", None)

        if self.transformer is None or self.vae is None or self.text_encoder is None or self.tokenizer is None or self.scheduler is None:
            raise RuntimeError("ZImagePipeline is missing required components (transformer/vae/text_encoder/tokenizer/scheduler)")

        # Training needs a scheduler that supports add_noise(). Some inference schedulers (e.g.
        # FlowMatchEulerDiscreteScheduler) don't implement it.
        if hasattr(self.scheduler, "add_noise"):
            self.noise_scheduler = self.scheduler
        else:
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        # LoRA injection (PEFT) - only for transformer by default
        if bool(self.lora_cfg.get("enable", True)):
            try:
                from peft import LoraConfig, get_peft_model
            except Exception as e:
                raise RuntimeError("peft is required for LoRA training") from e

            target_modules = self.lora_cfg.get("target_modules") or ["to_q", "to_k", "to_v", "to_out"]

            # DESIGN.md uses dotted names like attention.to_q. In Z-Image, these sometimes point to
            # container modules (e.g. ModuleList([Linear, Dropout])), which PEFT can't target.
            # Here we expand requested targets into leaf module names that are PEFT-supported.
            supported_types = (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention)
            try:
                from transformers.pytorch_utils import Conv1D  # type: ignore

                supported_types = supported_types + (Conv1D,)
            except Exception:
                pass

            requested: list[str] = [str(m) for m in target_modules]
            expanded: list[str] = []
            for name, module in self.transformer.named_modules():
                if not isinstance(module, supported_types):
                    continue
                for req in requested:
                    req = req.strip()
                    if not req:
                        continue
                    # Match either full dotted path prefix or last-token fragment.
                    last = req.split(".")[-1]
                    if name == req or name.startswith(req + ".") or name.endswith("." + last) or name == last:
                        expanded.append(name)
                        break

            # If we couldn't expand (model structure differs), fall back to last-token fragments.
            if expanded:
                cleaned = sorted(set(expanded))
            else:
                cleaned = [m.split(".")[-1] for m in requested]

            lora = LoraConfig(
                r=int(self.lora_cfg.get("rank", 16)),
                lora_alpha=int(self.lora_cfg.get("alpha", 16)),
                lora_dropout=float(self.lora_cfg.get("dropout", 0.0)),
                target_modules=cleaned,
                bias="none",
            )
            self.transformer = get_peft_model(self.transformer, lora)

        # Freeze everything except trainable params (LoRA)
        # Note: Diffusers pipelines are not guaranteed to behave like nn.Module for .parameters().
        for component in (self.vae, self.text_encoder, self.transformer):
            if component is None:
                continue
            for p in component.parameters():
                p.requires_grad = False

        # PEFT marks LoRA weights trainable; explicitly enforce:
        for n, p in self.transformer.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Register modules so Lightning sees parameters
        self.pipe.transformer = self.transformer

    def _encode_prompt(self, captions: list[str]) -> torch.Tensor:
        # Prefer pipeline helper if present
        if hasattr(self.pipe, "encode_prompt"):
            # Different diffusers versions/pipelines expose different signatures.
            # Try a few patterns, then fall back to manual tokenization.
            for kwargs in (
                {"device": self.device, "num_images_per_prompt": 1, "do_classifier_free_guidance": False},
                {"device": self.device, "do_classifier_free_guidance": False},
                {"device": self.device},
                {},
            ):
                try:
                    out = self.pipe.encode_prompt(captions, **kwargs)
                    if isinstance(out, tuple):
                        return out[0]
                    return out
                except TypeError:
                    continue

        tok = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tok = {k: v.to(self.device) for k, v in tok.items()}
        enc = self.text_encoder(**tok)
        if isinstance(enc, tuple):
            return enc[0]
        if hasattr(enc, "last_hidden_state"):
            return enc.last_hidden_state
        return enc

    @torch.no_grad()
    def _encode_images_to_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [-1,1]
        vae_out = self.vae.encode(pixel_values)
        if hasattr(vae_out, "latent_dist"):
            latents = vae_out.latent_dist.sample()
        elif isinstance(vae_out, tuple):
            latents = vae_out[0]
        else:
            latents = vae_out

        # Common SD convention
        scale = getattr(self.vae.config, "scaling_factor", 0.18215)
        return latents * scale

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        pixel_values = batch["pixel_values"].to(self.device)
        captions = batch["captions"]

        # 1) prompt embeds
        prompt_embeds = self._encode_prompt(captions)

        # 2) latents
        latents = self._encode_images_to_latents(pixel_values)

        # 3) sample noise + timestep
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        num_train_timesteps = int(getattr(getattr(self.noise_scheduler, "config", None), "num_train_timesteps", 1000))
        timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=self.device, dtype=torch.long)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 4) predict noise (transformer forward signature varies across implementations)
        fwd = self.transformer.forward
        sig = None
        try:
            sig = inspect.signature(fwd)
        except Exception:
            sig = None

        kwargs: Dict[str, Any] = {}
        if sig is not None:
            params = sig.parameters
            if "encoder_hidden_states" in params:
                kwargs["encoder_hidden_states"] = prompt_embeds
            elif "prompt_embeds" in params:
                kwargs["prompt_embeds"] = prompt_embeds
            elif "context" in params:
                kwargs["context"] = prompt_embeds
            elif "text_embeds" in params:
                kwargs["text_embeds"] = prompt_embeds

        # Try common call patterns
        try:
            model_out = self.transformer(noisy_latents, timesteps, **kwargs)
        except TypeError:
            # Some models use different arg names for timestep
            if sig is not None and "timestep" in sig.parameters:
                model_out = self.transformer(noisy_latents, timestep=timesteps, **kwargs)
            elif sig is not None and "timesteps" in sig.parameters:
                model_out = self.transformer(noisy_latents, timesteps=timesteps, **kwargs)
            else:
                model_out = self.transformer(noisy_latents, timesteps)
        if isinstance(model_out, tuple):
            pred = model_out[0]
        elif hasattr(model_out, "sample"):
            pred = model_out.sample
        else:
            pred = model_out

        loss = F.mse_loss(pred.float(), noise.float())
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr = float(self.training_cfg.get("learning_rate", 1e-4))

        trainable = [p for p in self.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters found (LoRA injection may have failed)")

        optimizer = torch.optim.AdamW(trainable, lr=lr)
        return optimizer

    def save_lora(self, out_dir: str) -> None:
        # Prefer PEFT save
        if hasattr(self.transformer, "save_pretrained"):
            self.transformer.save_pretrained(out_dir)
            return

        # Fallback: torch.save state dict
        from pathlib import Path

        out_path = str(Path(out_dir) / "adapter.pt")
        torch.save(self.transformer.state_dict(), out_path)
