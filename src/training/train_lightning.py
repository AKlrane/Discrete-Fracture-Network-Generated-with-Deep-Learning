import argparse
import math
import sys
from itertools import chain
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:  # pragma: no cover - compatibility with older installs.
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import Callback, ModelCheckpoint
        from pytorch_lightning.loggers import CSVLogger
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Lightning training requires the 'lightning' package. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dfn_dataset import DFNDataset
from src.models.flow_matching import weights_init as flow_weights_init
from src.models.wae import Decoder, Encoder, LatentDiscriminator
from src.models.wae import weights_init as wae_weights_init
from src.models.vqvae import weights_init as vqvae_weights_init
from src.models.wgan_gp import Critic, Generator
from src.models.wgan_gp import weights_init as wgan_weights_init
from src.training.train_flow_matching import create_model as create_flow_model
from src.training.train_flow_matching import create_optimizer as create_flow_optimizer
from src.training.train_flow_matching import flow_matching_loss, sample_flow
from src.training.train_wae import mmd_imq
from src.training.train_vqvae import create_model as create_vqvae_model
from src.training.train_vqvae import create_optimizer as create_vqvae_optimizer
from src.training.train_vqvae import save_vqvae_samples, vqvae_loss
from src.training.train_wgan_gp import gradient_penalty
from src.utils.device import lightning_accelerator, select_device
from src.utils.image_utils import save_image_grid
from src.utils.seed import set_seed

torch.set_float32_matmul_precision("medium")

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_lightning_path(outputs_cfg: dict[str, Any], key: str, fallback_key: str) -> Path:
    if key in outputs_cfg:
        return resolve_path(outputs_cfg[key])
    return resolve_path(outputs_cfg[fallback_key]) / "lightning"


class SampleGridCallback(Callback):
    def __init__(self, sample_dir: Path, sample_interval: int) -> None:
        super().__init__()
        self.sample_dir = sample_dir
        self.sample_interval = sample_interval

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if trainer.global_step > 0 and trainer.global_step % self.sample_interval == 0:
            pl_module.save_sample_grid(self.sample_dir, trainer.global_step)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.global_step > 0 and trainer.global_step % self.sample_interval != 0:
            pl_module.save_sample_grid(self.sample_dir, trainer.global_step)


class WGANLightningModule(pl.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.training_cfg = config["training"]
        self.model_cfg = config["model"]
        self.latent_dim = int(self.model_cfg["latent_dim"])
        self.base_channels = int(self.model_cfg["base_channels"])
        self.generator = Generator(latent_dim=self.latent_dim, base_channels=self.base_channels)
        self.critic = Critic(base_channels=self.base_channels)
        self.generator.apply(wgan_weights_init)
        self.critic.apply(wgan_weights_init)
        self.fixed_noise = torch.randn(int(self.training_cfg["num_sample_images"]), self.latent_dim)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        betas = (float(self.training_cfg["beta1"]), float(self.training_cfg["beta2"]))
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=float(self.training_cfg["lr"]),
            betas=betas,
        )
        optimizer_c = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.training_cfg["lr"]),
            betas=betas,
        )
        return [optimizer_g, optimizer_c]

    def training_step(self, real_images: torch.Tensor, batch_idx: int) -> None:
        optimizer_g, optimizer_c = self.optimizers()
        batch_size = real_images.size(0)

        for _ in range(int(self.training_cfg["critic_steps"])):
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.generator(z).detach()
            real_score = self.critic(real_images)
            fake_score = self.critic(fake_images)
            gp = gradient_penalty(self.critic, real_images, fake_images, self.device)
            critic_loss = (
                fake_score.mean()
                - real_score.mean()
                + float(self.training_cfg["lambda_gp"]) * gp
            )
            optimizer_c.zero_grad()
            self.manual_backward(critic_loss)
            optimizer_c.step()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z)
        fake_score_for_g = self.critic(fake_images)
        generator_loss = -fake_score_for_g.mean()
        optimizer_g.zero_grad()
        self.manual_backward(generator_loss)
        optimizer_g.step()

        self.log_dict(
            {
                "critic_loss": critic_loss.detach(),
                "generator_loss": generator_loss.detach(),
                "gradient_penalty": gp.detach(),
                "real_score_mean": real_score.detach().mean(),
                "fake_score_mean": fake_score.detach().mean(),
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

    def save_sample_grid(self, sample_dir: Path, step: int) -> None:
        self.generator.eval()
        with torch.no_grad():
            samples = self.generator(self.fixed_noise.to(self.device))
        nrow = int(math.sqrt(int(self.training_cfg["num_sample_images"])))
        save_image_grid(samples, sample_dir / f"step_{step:07d}.png", nrow=nrow)
        self.generator.train()


class WAELightningModule(pl.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.training_cfg = config["training"]
        self.model_cfg = config["model"]
        self.regularizer_cfg = config["regularizer"]
        self.regularizer_type = str(self.regularizer_cfg["type"]).lower()
        if self.regularizer_type not in {"mmd", "gan"}:
            raise ValueError("regularizer.type must be either 'mmd' or 'gan'")

        self.latent_dim = int(self.model_cfg["latent_dim"])
        self.base_channels = int(self.model_cfg["base_channels"])
        self.encoder = Encoder(latent_dim=self.latent_dim, base_channels=self.base_channels)
        self.decoder = Decoder(latent_dim=self.latent_dim, base_channels=self.base_channels)
        self.encoder.apply(wae_weights_init)
        self.decoder.apply(wae_weights_init)
        self.latent_discriminator: LatentDiscriminator | None = None
        if self.regularizer_type == "gan":
            hidden_dim = int(self.regularizer_cfg.get("discriminator_hidden_dim", 256))
            self.latent_discriminator = LatentDiscriminator(
                latent_dim=self.latent_dim,
                hidden_dim=hidden_dim,
            )
            self.latent_discriminator.apply(wae_weights_init)
        self.fixed_noise = torch.randn(int(self.training_cfg["num_sample_images"]), self.latent_dim)

    def configure_optimizers(self) -> torch.optim.Optimizer | list[torch.optim.Optimizer]:
        betas = (float(self.training_cfg["beta1"]), float(self.training_cfg["beta2"]))
        optimizer_autoencoder = torch.optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=float(self.training_cfg["lr"]),
            betas=betas,
        )
        if self.latent_discriminator is None:
            return optimizer_autoencoder
        optimizer_discriminator = torch.optim.Adam(
            self.latent_discriminator.parameters(),
            lr=float(self.regularizer_cfg.get("discriminator_lr", self.training_cfg["lr"])),
            betas=betas,
        )
        return [optimizer_autoencoder, optimizer_discriminator]

    def training_step(self, real_images: torch.Tensor, batch_idx: int) -> None:
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            optimizer_autoencoder, optimizer_discriminator = optimizers
        else:
            optimizer_autoencoder = optimizers
            optimizer_discriminator = None

        batch_size = real_images.size(0)
        discriminator_loss = torch.tensor(float("nan"), device=self.device)
        encoded_score_mean = torch.tensor(float("nan"), device=self.device)
        prior_score_mean = torch.tensor(float("nan"), device=self.device)

        if self.regularizer_type == "gan":
            assert self.latent_discriminator is not None
            assert optimizer_discriminator is not None
            for _ in range(int(self.regularizer_cfg.get("discriminator_steps", 1))):
                with torch.no_grad():
                    encoded_detached = self.encoder(real_images).detach()
                prior_z = torch.randn(batch_size, self.latent_dim, device=self.device)
                encoded_logits = self.latent_discriminator(encoded_detached)
                prior_logits = self.latent_discriminator(prior_z)
                discriminator_loss = 0.5 * (
                    F.binary_cross_entropy_with_logits(prior_logits, torch.ones_like(prior_logits))
                    + F.binary_cross_entropy_with_logits(encoded_logits, torch.zeros_like(encoded_logits))
                )
                optimizer_discriminator.zero_grad()
                self.manual_backward(discriminator_loss)
                optimizer_discriminator.step()
                encoded_score_mean = encoded_logits.detach().mean()
                prior_score_mean = prior_logits.detach().mean()

        encoded = self.encoder(real_images)
        reconstructed = self.decoder(encoded)
        reconstruction_loss = F.l1_loss(reconstructed, real_images)
        lambda_recon = float(self.regularizer_cfg.get("lambda_recon", 1.0))

        if self.regularizer_type == "mmd":
            prior_z = torch.randn_like(encoded)
            imq_scales = [
                float(scale)
                for scale in self.regularizer_cfg.get("imq_scales", [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
            ]
            mmd_loss = mmd_imq(encoded, prior_z, imq_scales)
            adversarial_loss = torch.tensor(float("nan"), device=self.device)
            total_loss = lambda_recon * reconstruction_loss + float(self.regularizer_cfg.get("lambda_mmd", 10.0)) * mmd_loss
        else:
            assert self.latent_discriminator is not None
            mmd_loss = torch.tensor(float("nan"), device=self.device)
            encoded_logits_for_autoencoder = self.latent_discriminator(encoded)
            adversarial_loss = F.binary_cross_entropy_with_logits(
                encoded_logits_for_autoencoder,
                torch.ones_like(encoded_logits_for_autoencoder),
            )
            total_loss = lambda_recon * reconstruction_loss + float(self.regularizer_cfg.get("lambda_adv", 1.0)) * adversarial_loss

        optimizer_autoencoder.zero_grad()
        self.manual_backward(total_loss)
        optimizer_autoencoder.step()

        self.log_dict(
            {
                "total_loss": total_loss.detach(),
                "reconstruction_loss": reconstruction_loss.detach(),
                "mmd_loss": mmd_loss.detach(),
                "adversarial_loss": adversarial_loss.detach(),
                "discriminator_loss": discriminator_loss.detach(),
                "encoded_score_mean": encoded_score_mean,
                "prior_score_mean": prior_score_mean,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

    def save_sample_grid(self, sample_dir: Path, step: int) -> None:
        self.decoder.eval()
        with torch.no_grad():
            samples = self.decoder(self.fixed_noise.to(self.device))
        nrow = int(math.sqrt(int(self.training_cfg["num_sample_images"])))
        save_image_grid(samples, sample_dir / f"step_{step:07d}.png", nrow=nrow)
        self.decoder.train()


class FlowMatchingLightningModule(pl.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.training_cfg = config["training"]
        self.data_cfg = config["data"]
        self.sampler_cfg = config.get("sampler", {})
        self.model_cfg = config["model"]
        self.model = create_flow_model(config)
        self.model.apply(flow_weights_init)
        image_channels = int(self.model_cfg.get("image_channels", 1))
        image_size = int(self.data_cfg["image_size"])
        self.fixed_noise = torch.randn(
            int(self.training_cfg["num_sample_images"]),
            image_channels,
            image_size,
            image_size,
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return create_flow_optimizer(self.model, self.training_cfg)

    def training_step(self, real_images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss, metrics = flow_matching_loss(self.model, real_images)
        self.log_dict(
            {
                "loss": loss.detach(),
                "t_mean": metrics["t_mean"],
                "target_velocity_norm": metrics["target_velocity_norm"],
                "predicted_velocity_norm": metrics["predicted_velocity_norm"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def save_sample_grid(self, sample_dir: Path, step: int) -> None:
        self.model.eval()
        with torch.no_grad():
            samples = sample_flow(
                self.model,
                self.fixed_noise.to(self.device),
                solver=str(self.sampler_cfg.get("solver", "euler")),
                num_steps=int(self.sampler_cfg.get("num_steps", 50)),
            )
        nrow = int(math.sqrt(int(self.training_cfg["num_sample_images"])))
        save_image_grid(samples, sample_dir / f"step_{step:07d}.png", nrow=nrow)
        self.model.train()


class VQVAELightningModule(pl.LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.training_cfg = config["training"]
        self.data_cfg = config["data"]
        self.loss_cfg = config.get("loss", {})
        self.sampling_cfg = config.get("sampling", {})
        self.model_cfg = config["model"]
        self.model = create_vqvae_model(config)
        self.model.apply(vqvae_weights_init)

        dataset = DFNDataset(
            image_dir=resolve_path(self.data_cfg["image_dir"]),
            image_size=int(self.data_cfg["image_size"]),
        )
        num_sample_images = min(int(self.training_cfg["num_sample_images"]), len(dataset))
        self.fixed_images = torch.stack([dataset[index] for index in range(num_sample_images)])
        self.latent_size = int(self.data_cfg["image_size"]) // int(
            self.model_cfg.get("latent_downsample_factor", 8)
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return create_vqvae_optimizer(self.model, self.training_cfg)

    def training_step(self, real_images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        total_loss, metrics = vqvae_loss(self.model, real_images, self.loss_cfg)
        self.log_dict(
            {
                "total_loss": metrics["total_loss"],
                "reconstruction_loss": metrics["reconstruction_loss"],
                "vq_loss": metrics["vq_loss"],
                "perplexity": metrics["perplexity"],
                "code_usage": metrics["code_usage"],
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return total_loss

    def save_sample_grid(self, sample_dir: Path, step: int) -> None:
        save_vqvae_samples(
            self.model,
            self.fixed_images.to(self.device),
            sample_dir,
            step,
            self.fixed_images.size(0),
            self.latent_size,
            save_random=bool(self.sampling_cfg.get("save_random_samples", True)),
        )


def create_model(config: dict[str, Any], model_type: str) -> pl.LightningModule:
    if model_type == "auto":
        if config.get("method") in {"flow_matching", "vqvae"}:
            model_type = str(config["method"])
        else:
            model_type = "wae" if "regularizer" in config else "wgan_gp"
    if model_type == "wgan_gp":
        return WGANLightningModule(config)
    if model_type == "wae":
        return WAELightningModule(config)
    if model_type == "flow_matching":
        return FlowMatchingLightningModule(config)
    if model_type == "vqvae":
        return VQVAELightningModule(config)
    raise ValueError("--model must be one of: auto, wgan_gp, wae, flow_matching, vqvae")


def create_dataloader(config: dict[str, Any]) -> DataLoader:
    training_cfg = config["training"]
    data_cfg = config["data"]
    dataset = DFNDataset(
        image_dir=resolve_path(data_cfg["image_dir"]),
        image_size=int(data_cfg["image_size"]),
    )
    return DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        pin_memory=select_device(str(training_cfg.get("device", "auto"))).type == "cuda",
        drop_last=True,
    )


def train_lightning(
    config: dict[str, Any],
    model_type: str,
    resume: str | Path | None = None,
    max_steps: int | None = None,
) -> None:
    training_cfg = config["training"]
    outputs_cfg = config["outputs"]
    set_seed(int(training_cfg["seed"]))

    sample_dir = resolve_lightning_path(outputs_cfg, "lightning_sample_dir", "sample_dir")
    checkpoint_dir = resolve_lightning_path(outputs_cfg, "lightning_checkpoint_dir", "checkpoint_dir")
    log_dir = resolve_lightning_path(outputs_cfg, "lightning_log_dir", "log_dir")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    accelerator, devices = lightning_accelerator(str(training_cfg.get("device", "auto")))
    precision = str(training_cfg.get("precision", "32-true"))
    model = create_model(config, model_type)
    dataloader = create_dataloader(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:04d}-{step:07d}",
        every_n_epochs=int(training_cfg["checkpoint_interval"]),
        save_top_k=-1,
        save_last=True,
    )
    sample_callback = SampleGridCallback(
        sample_dir=sample_dir,
        sample_interval=int(training_cfg["sample_interval"]),
    )
    logger = CSVLogger(save_dir=log_dir.parent, name=log_dir.name, version="")
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=int(training_cfg["num_epochs"]),
        max_steps=max_steps or -1,
        precision=precision,
        logger=logger,
        callbacks=[checkpoint_callback, sample_callback],
        enable_checkpointing=True,
        log_every_n_steps=int(training_cfg.get("log_interval", 50)),
    )
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=resume)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DFN generators with Lightning.")
    parser.add_argument("--config", type=Path, default=Path("configs/wgan_gp_128.yaml"))
    parser.add_argument(
        "--model",
        choices=("auto", "wgan_gp", "wae", "flow_matching", "vqvae"),
        default="auto",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_lightning(
        load_config(resolve_path(args.config)),
        model_type=args.model,
        resume=args.resume,
        max_steps=args.max_steps,
    )
