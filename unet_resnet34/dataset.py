"""
dataset.py
==========
PyTorch Dataset for SpaceNet AOI_2_Vegas building segmentation.

Each __getitem__ returns:
    image : FloatTensor  (3, IMG_SIZE, IMG_SIZE)  – ImageNet-normalised
    mask  : FloatTensor  (1, IMG_SIZE, IMG_SIZE)  – binary {0.0, 1.0}

Training augmentations (albumentations):
    RandomCrop, HorizontalFlip, VerticalFlip, RandomRotate90,
    RandomBrightnessContrast, GaussNoise, GridDistortion

Validation:
    CenterCrop only + normalise
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.RandomCrop(height=img_size, width=img_size, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2),
        A.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.CenterCrop(height=img_size, width=img_size, p=1.0),
        A.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class SpaceNetDataset(Dataset):
    """
    Parameters
    ----------
    rgb_dir   : directory of PS-RGB GeoTIFFs
    mask_dir  : directory of binary PNG masks (from preprocess.py)
    ids       : list of string image IDs  e.g. ['1', '42', …]
    transform : albumentations Compose pipeline; None → auto from `augment`
    augment   : used only when transform is None
    """

    def __init__(
        self,
        rgb_dir:   Path | str,
        mask_dir:  Path | str,
        ids:       List[str],
        transform: Optional[A.Compose] = None,
        augment:   bool = False,
    ):
        self.rgb_dir  = Path(rgb_dir)
        self.mask_dir = Path(mask_dir)
        self.ids      = ids
        self.transform = (
            transform if transform is not None
            else (get_train_transforms() if augment else get_val_transforms())
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_image(self, img_id: str) -> np.ndarray:
        """Load PS-RGB TIF → uint8 (H, W, 3)."""
        path = self.rgb_dir / f"SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{img_id}.tif"
        try:
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read([1, 2, 3])          # (3, H, W)
                arr = np.transpose(arr, (1, 2, 0)) # → (H, W, 3)
                if arr.dtype != np.uint8:
                    lo, hi = arr.min(), arr.max()
                    arr = ((arr.astype(np.float32) - lo) / max(hi - lo, 1) * 255
                           ).astype(np.uint8)
            return arr
        except Exception:
            # Fallback: PIL (works for 3-band TIFF saved as RGB)
            return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _load_mask(self, img_id: str) -> np.ndarray:
        """Load binary PNG mask → uint8 (H, W)."""
        path = self.mask_dir / f"mask_img{img_id}.png"
        if not path.exists():
            return np.zeros((config.IMAGE_H, config.IMAGE_W), dtype=np.uint8)
        return np.array(Image.open(path).convert("L"), dtype=np.uint8)

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.ids[idx]
        image  = self._load_image(img_id)   # (H, W, 3)  uint8
        mask   = self._load_mask(img_id)    # (H, W)     uint8

        out    = self.transform(image=image, mask=mask)
        img_t  = out["image"].float()               # (3, H, W)
        msk_t  = out["mask"].float().unsqueeze(0)   # (1, H, W)
        msk_t  = (msk_t > 127.0).float()           # binarise

        return img_t, msk_t


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_ids(split_file: Path | str) -> List[str]:
    return [l.strip() for l in Path(split_file).read_text().splitlines() if l.strip()]


def make_loaders(
    rgb_dir:   Path | str = config.RGB_DIR,
    mask_dir:  Path | str = config.MASK_DIR,
    split_dir: Path | str = config.SPLIT_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders from the txt split files."""
    train_ids = load_ids(Path(split_dir) / "train.txt")
    val_ids   = load_ids(Path(split_dir) / "val.txt")

    train_ds = SpaceNetDataset(rgb_dir, mask_dir, train_ids, augment=True)
    val_ds   = SpaceNetDataset(rgb_dir, mask_dir, val_ids,   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )
    return train_loader, val_loader
