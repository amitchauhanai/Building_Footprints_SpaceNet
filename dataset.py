"""
dataset.py  –  SpaceNet AOI_2_Vegas building segmentation dataset
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALB_AVAILABLE = True
except ImportError:
    ALB_AVAILABLE = False

DEFAULT_IMG_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def build_train_transforms(img_size: int = DEFAULT_IMG_SIZE):
    assert ALB_AVAILABLE, "pip install albumentations"
    return A.Compose([
        A.RandomCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def build_val_transforms(img_size: int = DEFAULT_IMG_SIZE):
    assert ALB_AVAILABLE, "pip install albumentations"
    return A.Compose([
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class _MinimalTransform:
    def __init__(self, img_size: int):
        self.size = img_size
        self.mean = np.array(MEAN, dtype=np.float32)
        self.std  = np.array(STD,  dtype=np.float32)

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        img = np.array(Image.fromarray(image).resize((self.size, self.size), Image.BILINEAR))
        msk = np.array(Image.fromarray(mask).resize((self.size, self.size), Image.NEAREST))
        img = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        msk = msk.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img.transpose(2, 0, 1))
        msk_t = torch.from_numpy(msk).unsqueeze(0)
        return img_t, msk_t


class SpaceNetVegasDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, id_list: List[str],
                 transform=None, img_size: int = DEFAULT_IMG_SIZE, augment: bool = False):
        self.rgb_dir  = Path(rgb_dir)
        self.mask_dir = Path(mask_dir)
        self.ids      = id_list

        if transform is not None:
            self._tfm     = transform
            self._use_alb = ALB_AVAILABLE and hasattr(transform, "additional_targets")
        elif ALB_AVAILABLE:
            self._tfm     = build_train_transforms(img_size) if augment else build_val_transforms(img_size)
            self._use_alb = True
        else:
            self._tfm     = _MinimalTransform(img_size)
            self._use_alb = False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id    = self.ids[idx]
        rgb_path  = self.rgb_dir  / f"SN2_buildings_train_AOI_2_Vegas_PS-RGB_img{img_id}.tif"
        mask_path = self.mask_dir / f"mask_img{img_id}.png"

        image = self._load_rgb(rgb_path)
        mask  = (np.array(Image.open(mask_path).convert("L"))
                 if mask_path.exists()
                 else np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8))

        if self._use_alb:
            out   = self._tfm(image=image, mask=mask)
            img_t = out["image"].float()
            msk_t = out["mask"].float().unsqueeze(0)
        else:
            img_t, msk_t = self._tfm(image, mask)

        return img_t, (msk_t > 0.5).float()

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        try:
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read()[:3]
                arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    lo, hi = arr.min(), arr.max()
                    arr = ((arr.astype(np.float32) - lo) / (hi - lo + 1e-6) * 255).astype(np.uint8)
            return arr
        except Exception:
            pass
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_split_ids(split_file: str) -> List[str]:
    return [l.strip() for l in Path(split_file).read_text().splitlines() if l.strip()]


def make_datasets(rgb_dir, mask_dir, data_dir, img_size=DEFAULT_IMG_SIZE):
    base = Path(data_dir)
    train_ds = SpaceNetVegasDataset(rgb_dir, mask_dir,
                                    load_split_ids(str(base / "train.txt")),
                                    img_size=img_size, augment=True)
    val_ds   = SpaceNetVegasDataset(rgb_dir, mask_dir,
                                    load_split_ids(str(base / "val.txt")),
                                    img_size=img_size, augment=False)
    return train_ds, val_ds
