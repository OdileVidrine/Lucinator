import argparse
import math
import os
import re
from typing import List, Tuple

from PIL import Image


EPOCH_RE = re.compile(r"epoch_(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-epoch sample images into a single grid."
    )
    parser.add_argument("--samples-dir", default="outputs/samples")
    parser.add_argument("--out", default="outputs/samples/epoch_grid.png")
    parser.add_argument(
        "--cols",
        type=int,
        default=0,
        help="Number of columns (0 = auto square-ish).",
    )
    parser.add_argument(
        "--tile-index",
        type=int,
        default=0,
        help="Which tile to pick from each epoch grid (0 = top-left).",
    )
    return parser.parse_args()


def _epoch_key(path: str) -> int:
    name = os.path.basename(path)
    match = EPOCH_RE.search(name)
    return int(match.group(1)) if match else -1


def _infer_grid_params(w: int, h: int, tiles_x: int, tiles_y: int) -> Tuple[int, int, int]:
    # Infer padding and tile size from a torchvision make_grid output.
    # make_grid uses padding on all sides: total = tiles * tile + padding * (tiles + 1)
    for padding in range(0, 11):
        tile_w = (w - padding * (tiles_x + 1)) / tiles_x
        tile_h = (h - padding * (tiles_y + 1)) / tiles_y
        if tile_w.is_integer() and tile_h.is_integer():
            return int(tile_w), int(tile_h), padding
    raise SystemExit(f"Could not infer grid params for {w}x{h}")


def load_images(paths: List[str], tile_index: int) -> List[Image.Image]:
    images = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        tiles_x = 8
        tiles_y = 8
        tile_w, tile_h, padding = _infer_grid_params(w, h, tiles_x, tiles_y)
        total_tiles = tiles_x * tiles_y
        if tile_index < 0 or tile_index >= total_tiles:
            raise SystemExit(f"tile-index {tile_index} out of range for {path} ({total_tiles} tiles)")
        r = tile_index // tiles_x
        c = tile_index % tiles_x
        left = padding + c * (tile_w + padding)
        top = padding + r * (tile_h + padding)
        crop = img.crop((left, top, left + tile_w, top + tile_h))
        images.append(crop)
    return images


def grid_size(n: int, cols: int) -> Tuple[int, int]:
    if cols <= 0:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def main() -> None:
    args = parse_args()
    paths = [p for p in os.listdir(args.samples_dir) if EPOCH_RE.search(p)]
    paths = [os.path.join(args.samples_dir, p) for p in paths]
    paths.sort(key=_epoch_key)
    if not paths:
        raise SystemExit(f"No epoch_XXX.png files found in {args.samples_dir}")

    images = load_images(paths, args.tile_index)
    w, h = images[0].size
    rows, cols = grid_size(len(images), args.cols)

    grid = Image.new("RGB", (cols * w, rows * h), (0, 0, 0))
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        grid.paste(img, (c * w, r * h))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    grid.save(args.out)
    print(f"Saved {args.out} ({len(images)} images, {rows}x{cols})")


if __name__ == "__main__":
    main()
