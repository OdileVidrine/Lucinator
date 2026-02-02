import argparse
import os

import torch
from torchvision.utils import save_image

from train import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a trained DCGAN checkpoint.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--out", default="outputs/samples/generated.png")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--feature-maps", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    gen = Generator(args.latent_dim, args.feature_maps, channels=3).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    gen.load_state_dict(ckpt["generator"])
    gen.eval()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with torch.no_grad():
        noise = torch.randn(args.count, args.latent_dim, 1, 1, device=device)
        fake = gen(noise)
        save_image(fake, args.out, nrow=8, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
