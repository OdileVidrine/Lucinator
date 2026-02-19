import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import List

import torch
from torch import nn
from torch import set_num_interop_threads, set_num_threads
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm


ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, image_size: int):
        self.root_dir = root_dir
        self.paths = self._collect_paths(root_dir)
        if not self.paths:
            raise ValueError(f"No images found in {root_dir}")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    @staticmethod
    def _collect_paths(root_dir: str) -> List[str]:
        paths = []
        for ext in ALLOWED_EXTS:
            paths.extend(glob.glob(os.path.join(root_dir, f"**/*{ext}"), recursive=True))
            paths.extend(glob.glob(os.path.join(root_dir, f"**/*{ext.upper()}"), recursive=True))
        return sorted(set(paths))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            return self.transform(img)


class Generator(nn.Module):
    def __init__(self, latent_dim: int, feature_maps: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    beta1: float
    latent_dim: int
    feature_maps: int
    seed: int
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, gen: nn.Module, disc: nn.Module, g_opt, d_opt, epoch: int, cfg: TrainConfig):
    torch.save(
        {
            "generator": gen.state_dict(),
            "discriminator": disc.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
            "epoch": epoch,
            "config": cfg.__dict__,
        },
        path,
    )


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    samples_dir = os.path.join(cfg.output_dir, "samples")
    checkpoints_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    dataset = ImageFolderDataset(cfg.data_dir, cfg.image_size)
    num_workers = os.cpu_count() or 1
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    if device.type == "cpu":
        # Always use all CPU cores when running on CPU.
        threads = os.cpu_count() or 1
        set_num_threads(threads)
        set_num_interop_threads(max(1, threads // 2))

    gen = Generator(cfg.latent_dim, cfg.feature_maps, channels=3).to(device)
    disc = Discriminator(cfg.feature_maps, channels=3).to(device)

    g_opt = torch.optim.Adam(gen.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(64, cfg.latent_dim, 1, 1, device=device)

    for epoch in range(1, cfg.epochs + 1):
        gen.train()
        disc.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for real in pbar:
            real = real.to(device)
            batch_size = real.size(0)

            # Train Discriminator
            disc.zero_grad(set_to_none=True)
            # Targets for the discriminator: "real" should be 1, "fake" should be 0.
            real_labels = torch.ones(batch_size, device=device)
            fake_labels = torch.zeros(batch_size, device=device)

            # 1) Score real images against the "real" target.
            real_logits = disc(real)
            d_real_loss = criterion(real_logits, real_labels)

            # 2) Score generated images against the "fake" target.
            noise = torch.randn(batch_size, cfg.latent_dim, 1, 1, device=device)
            fake = gen(noise)
            fake_logits = disc(fake.detach())
            d_fake_loss = criterion(fake_logits, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()

            # Train Generator
            gen.zero_grad(set_to_none=True)
            # Generator wants the discriminator to predict "real" for its fakes,
            # so we use real_labels as the target here (label flipping).
            noise = torch.randn(batch_size, cfg.latent_dim, 1, 1, device=device)
            fake = gen(noise)
            fake_logits = disc(fake)
            g_loss = criterion(fake_logits, real_labels)
            g_loss.backward()
            g_opt.step()

            pbar.set_postfix({"d_loss": f"{d_loss.item():.3f}", "g_loss": f"{g_loss.item():.3f}"})

        gen.eval()
        with torch.no_grad():
            sample = gen(fixed_noise)
            save_image(
                sample,
                os.path.join(samples_dir, f"epoch_{epoch:03d}.png"),
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
            )

        if epoch % 5 == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(checkpoints_dir, f"ckpt_epoch_{epoch:03d}.pt")
            save_checkpoint(ckpt_path, gen, disc, g_opt, d_opt, epoch, cfg)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a simple DCGAN from scratch.")
    parser.add_argument("--data-dir", default="example_data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--feature-maps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        latent_dim=args.latent_dim,
        feature_maps=args.feature_maps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    train(parse_args())
