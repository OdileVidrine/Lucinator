# Lucinator GAN (from scratch)

This is a minimal PyTorch DCGAN that trains **from scratch** on the images in `example_data/`.

## Setup (uv)
```bash
uv venv
uv sync
```

If you have a GPU, install the appropriate PyTorch build for CUDA instead of the default CPU build:
```bash
uv pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```
(Adjust the CUDA version to match your system.)

## Train
```bash
uv run python train.py \
  --data-dir example_data \
  --image-size 64 \
  --batch-size 16 \
  --epochs 50
```

Outputs:
- `outputs/samples/epoch_XXX.png` (progress samples)
- `outputs/checkpoints/ckpt_epoch_XXX.pt`

## Generate
```bash
uv run python sample.py --ckpt outputs/checkpoints/ckpt_epoch_050.pt
```

## Notes
- This uses a small DCGAN architecture. With a limited dataset, results may be rough.
- You can try `--image-size 128` and a smaller batch size if you have enough GPU memory.
