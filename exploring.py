import os
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import numpy as np

# Correct path based on your actual folder structure
base_dir = r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\archive (1)\chest_xray"

# Safety check so you get a clear error instead of a cryptic traceback
if not os.path.exists(base_dir):
    raise FileNotFoundError(
        f"base_dir does not exist: {base_dir}\n"
        "Update the base_dir variable to point to your actual dataset folder "
        "(it should contain train/val/test subfolders)."
    )

# 1. Check class balance
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        path = os.path.join(base_dir, split, cls)
        if os.path.exists(path):
            count = len(os.listdir(path))
            print(f"{split}/{cls}: {count} images")
        else:
            print(f"{split}/{cls}: folder not found at {path}")

# 2. Check image sizes/modes (are they all grayscale? consistent size?)
normal_train_path = os.path.join(base_dir, "train", "NORMAL")
sample_files = os.listdir(normal_train_path)[:20]
for f in sample_files:
    img = Image.open(os.path.join(normal_train_path, f))
    print(f, img.size, img.mode)

# 3. Visualize sample images from each class
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i, cls in enumerate(["NORMAL", "PNEUMONIA"]):
    folder = os.path.join(base_dir, "train", cls)
    files = os.listdir(folder)[:4]
    for j, f in enumerate(files):
        img = Image.open(os.path.join(folder, f))
        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].set_title(cls)
        axes[i, j].axis("off")
plt.tight_layout()
plt.show()

# 4. Check for corrupted files
corrupted = []
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        folder = os.path.join(base_dir, split, cls)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            fpath = os.path.join(folder, f)
            try:
                img = Image.open(fpath)
                img.verify()
            except (UnidentifiedImageError, OSError):
                corrupted.append(fpath)

print(f"Found {len(corrupted)} corrupted files")
print(corrupted[:10])