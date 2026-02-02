import os
from PIL import Image, ImageOps

root = 'example_data'
size = (64, 64)
allowed = {'.jpg', '.jpeg', '.png'}
count = 0

for dirpath, _, filenames in os.walk(root):
    for name in filenames:
        ext = os.path.splitext(name)[1].lower()
        if ext not in allowed:
            continue
        path = os.path.join(dirpath, name)
        with Image.open(path) as img:
            img = img.convert('RGB')
            # center-crop to square, then resize to 64x64
            out = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            out.save(path)
        count += 1

print(f"Resized {count} images to 64x64 in {root}")
