"""Public RDPO high-noise fine-tuning entry for SCoPE.

Training reproduces the released recipe: only the high-noise expert is
optimized, timesteps are sampled from ``[0.9, 1.0)``, and the data mixture is a
``ConcatDataset`` of the four native loaders in :mod:`scope.data`. The run is
described by a single YAML config (see configs/train_rdpo_high_only.yaml).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from scope.config import SCOPE_MODEL_ID, InferenceConfig
from scope.data import build_training_dataset
from scope.patch import enable_scope_grad
from scope.weights import load_pipeline, resolve_model_dir

_TRAINABLE_KEYWORDS = ["plucker_pe", "self_attn", "norm3", "ffn"]

# PIL first frames cannot be default-collated, so they are kept as a list.
_LIST_KEYS = ("first_frame_pil",)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Training config must be a mapping: {path}")
    return config


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        output[key] = values if key in _LIST_KEYS else default_collate(values)
    return output


class SCoPEFineTuner(pl.LightningModule):
    """Fine-tune the released SCoPE model in the RDPO high-noise regime."""

    def __init__(
        self,
        model_path: str,
        learning_rate: float,
        lr_plucker: float,
        lr_self_attn: float,
        lr_ffn: float,
        weight_decay: float,
        height: int,
        width: int,
        num_frames: int,
        plucker_eps: float,
        log_scale_aug_prob: float,
        log_scale_aug_range: tuple[float, float],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.pipe = None

    def setup(self, stage: str | None = None) -> None:
        if self.pipe is not None:
            return
        inference_config = InferenceConfig(
            height=self.hparams.height,
            width=self.hparams.width,
            num_frames=self.hparams.num_frames,
        )
        model_dir = resolve_model_dir(self.hparams.model_path)
        self.pipe = load_pipeline(model_dir, inference_config)
        for block in self.pipe.dit.blocks:
            encoding = block.self_attn.plucker_pe
            encoding.plucker_eps = self.hparams.plucker_eps
            encoding.log_scale_aug_prob = self.hparams.log_scale_aug_prob
            encoding.log_scale_aug_range = tuple(self.hparams.log_scale_aug_range)
        self.pipe.i2v_vae_condition_mode = "official_zero"
        enable_scope_grad(self.pipe, _TRAINABLE_KEYWORDS, expert="high_noise_model")
        object.__setattr__(self, "dit", self.pipe.dit)
        object.__setattr__(self, "dit2", self.pipe.dit2)
        self.pipe.scheduler.set_timesteps(
            self.pipe.scheduler.num_train_timesteps,
            training=True,
        )

    def training_step(self, batch: dict[str, Any], batch_index: int) -> torch.Tensor:
        del batch_index
        if len(batch["first_frame_pil"]) != 1:
            raise ValueError("SCoPE A14B training currently requires batch_size=1 per GPU")
        self.pipe.device = self.device

        self.pipe.load_models_to_device(["vae"])
        video = batch["video"].to(device=self.device, dtype=self.pipe.torch_dtype)
        with torch.inference_mode():
            latents = self.pipe.vae.single_encode(video, self.device)
        latents = latents.to(device=self.device, dtype=self.pipe.torch_dtype).detach()

        self.pipe.load_models_to_device(["text_encoder"])
        with torch.inference_mode():
            context = self.pipe.prompter.encode_prompt(
                batch["caption"], positive=True, device=self.device
            )

        conditioning = self.pipe.build_i2v_conditioning(
            input_image=batch["first_frame_pil"][0],
            num_frames=self.hparams.num_frames,
            height=self.hparams.height,
            width=self.hparams.width,
        )
        if conditioning is None:
            raise RuntimeError("The selected model does not expose Wan2.2 I2V conditioning")

        camera = {
            "pose": batch["pose"].to(device=self.device, dtype=self.pipe.torch_dtype),
            "x_fov": batch["x_fov"].to(device=self.device, dtype=self.pipe.torch_dtype),
            "xi": batch["xi"].to(device=self.device, dtype=self.pipe.torch_dtype),
        }
        self.pipe.load_models_to_device(["dit", "dit2"])
        loss = self.pipe.training_loss(
            input_latents=latents,
            noise=torch.randn_like(latents),
            context=context,
            height=self.hparams.height,
            width=self.hparams.width,
            camera_control_panshot=camera,
            y=conditioning,
            first_frame_latents=conditioning[:, 4:, 0:1].clone(),
            min_timestep_boundary=0.9,
            max_timestep_boundary=1.0,
            switch_DiT_boundary=0.9,
            use_gradient_checkpointing=True,
            use_gradient_checkpointing_offload=False,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        grouped_parameters: dict[str, list[torch.nn.Parameter]] = {
            "plucker_pe": [],
            "self_attn": [],
            "ffn_norm3": [],
        }
        for name, parameter in self.pipe.dit.named_parameters():
            if not parameter.requires_grad:
                continue
            if "plucker_pe" in name:
                grouped_parameters["plucker_pe"].append(parameter)
            elif "self_attn" in name:
                grouped_parameters["self_attn"].append(parameter)
            elif "ffn" in name or "norm3" in name:
                grouped_parameters["ffn_norm3"].append(parameter)
            else:
                raise RuntimeError(
                    f"Trainable parameter is not assigned to an optimizer group: {name}"
                )

        if not any(grouped_parameters.values()):
            raise RuntimeError("No trainable parameters found in the high-noise expert")
        learning_rates = {
            "plucker_pe": self.hparams.lr_plucker,
            "self_attn": self.hparams.lr_self_attn,
            "ffn_norm3": self.hparams.lr_ffn,
        }
        parameter_groups = [
            {"params": parameters, "lr": learning_rates[name], "name": name}
            for name, parameters in grouped_parameters.items()
            if parameters
        ]
        return torch.optim.AdamW(
            parameter_groups,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_rdpo_high_only.yaml"),
        help="Training YAML (see configs/train_rdpo_high_only.yaml).",
    )
    parser.add_argument("--model_path", default=None, help="Override config model_path.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Override config output_dir.")
    parser.add_argument("--num_gpus", type=int, default=None, help="Override trainer.num_gpus.")
    parser.add_argument("--max_steps", type=int, default=None, help="Override trainer.max_steps.")
    parser.add_argument("--resume_from_checkpoint", type=Path, default=None)
    return parser


def build_strategy(num_gpus: int) -> str | Any:
    if num_gpus <= 1:
        return "auto"
    from pytorch_lightning.strategies import FSDPStrategy
    from torch.distributed.fsdp import ShardingStrategy

    from diffsynth.models.wan_video_dit import DiTBlock

    return FSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy={DiTBlock},
        state_dict_type="sharded",
        use_orig_params=True,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    data_config = config["data"]
    model_config = config.get("model", {})
    trainer_config = config.get("trainer", {})
    optimizer_config = config.get("optimizer", {})

    model_path = args.model_path or config.get("model_path", SCOPE_MODEL_ID)
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/training"))
    num_gpus = args.num_gpus or int(trainer_config.get("num_gpus", 8))
    max_steps = args.max_steps or int(trainer_config.get("max_steps", 10_000))

    pl.seed_everything(int(config.get("seed", 42)), workers=True)

    dataset = build_training_dataset(data_config)
    num_workers = int(data_config.get("num_workers", 4))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="scope-{step:06d}",
        every_n_train_steps=int(trainer_config.get("save_every_n_steps", 1_000)),
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    model = SCoPEFineTuner(
        model_path=model_path,
        learning_rate=float(optimizer_config.get("learning_rate", 2e-5)),
        lr_plucker=float(optimizer_config.get("lr_plucker", 2e-5)),
        lr_self_attn=float(optimizer_config.get("lr_self_attn", 5e-6)),
        lr_ffn=float(optimizer_config.get("lr_ffn", 2e-6)),
        weight_decay=float(optimizer_config.get("weight_decay", 1e-2)),
        height=int(data_config.get("height", 480)),
        width=int(data_config.get("width", 832)),
        num_frames=int(data_config.get("num_frames", 81)),
        plucker_eps=float(model_config.get("plucker_eps", 1e-3)),
        log_scale_aug_prob=float(model_config.get("log_scale_aug_prob", 0.3)),
        log_scale_aug_range=tuple(model_config.get("log_scale_aug_range", (-1.2, 1.6))),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        strategy=build_strategy(num_gpus),
        precision=str(trainer_config.get("precision", "bf16-mixed")),
        max_steps=max_steps,
        gradient_clip_val=float(trainer_config.get("gradient_clip_val", 1.0)),
        default_root_dir=output_dir,
        callbacks=[checkpoint],
        log_every_n_steps=int(trainer_config.get("log_every_n_steps", 10)),
    )
    trainer.fit(model, train_dataloaders=loader, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
