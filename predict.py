"""
predict.py  –  Inference with a trained UNetResNet34

Usage:
    # single image
    python predict.py --checkpoint checkpoints/best.pth --input ../PS-RGB/SN2_..._img1.tif

    # whole directory
    python predict.py --checkpoint checkpoints/best.pth \
                      --input ../PS-RGB --pattern "*PS-RGB*.tif" \
                      --out_dir outputs/predictions
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import DEFAULT_IMG_SIZE, MEAN, STD
from model   import build_model


def load_rgb(path: Path) -> np.ndarray:
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read()[:3]
            arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                lo, hi = arr.min(), arr.max()
                arr = ((arr.astype(np.float32)-lo)/(hi-lo+1e-6)*255).astype(np.uint8)
        return arr
    except Exception:
        pass
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def preprocess(image: np.ndarray, img_size: int) -> torch.Tensor:
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    img  = np.array(Image.fromarray(image).resize((img_size, img_size), Image.BILINEAR),
                    dtype=np.float32)
    img  = (img / 255.0 - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def postprocess(logit, orig_h, orig_w, thr):
    prob = torch.sigmoid(logit.squeeze())
    prob = F.interpolate(prob.unsqueeze(0).unsqueeze(0),
                         size=(orig_h, orig_w), mode="bilinear",
                         align_corners=False).squeeze()
    prob_np  = prob.cpu().numpy()
    bin_mask = (prob_np > thr).astype(np.uint8) * 255
    prob_u8  = (prob_np * 255).clip(0, 255).astype(np.uint8)
    return bin_mask, prob_u8


def make_overlay(image, binary_mask, color=(220, 30, 30), alpha=0.45):
    out = image.astype(np.float32).copy(); m = binary_mask > 128
    for c, v in enumerate(color): out[m, c] = out[m, c]*(1-alpha) + v*alpha
    return out.clip(0, 255).astype(np.uint8)


class Predictor:
    def __init__(self, checkpoint, img_size=DEFAULT_IMG_SIZE, threshold=0.5, device="auto"):
        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
        self.device    = device
        self.img_size  = img_size
        self.threshold = threshold
        self.model     = build_model(pretrained=False, device=device)
        ck = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        print(f"Loaded {checkpoint}  epoch={ck.get('epoch','?')}  "
              f"best_iou={ck.get('best_val_iou',0):.4f}  device={device}")

    @torch.no_grad()
    def predict(self, image_path: Path):
        image  = load_rgb(image_path)
        orig_h, orig_w = image.shape[:2]
        t      = preprocess(image, self.img_size).to(self.device)
        logit  = self.model(t)
        binary, prob = postprocess(logit, orig_h, orig_w, self.threshold)
        return image, binary, prob, make_overlay(image, binary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input",      required=True)
    p.add_argument("--out_dir",    default="outputs/predictions")
    p.add_argument("--pattern",    default="*.tif")
    p.add_argument("--img_size",   type=int,   default=DEFAULT_IMG_SIZE)
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--save_prob",  action="store_true")
    p.add_argument("--device",     default="auto")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pred    = Predictor(args.checkpoint, args.img_size, args.threshold, args.device)
    paths   = (sorted(Path(args.input).glob(args.pattern))
               if Path(args.input).is_dir() else [Path(args.input)])

    for i, p in enumerate(paths):
        _, binary, prob, overlay = pred.predict(p)
        Image.fromarray(binary,  mode="L").save(out_dir / f"{p.stem}_mask.png")
        Image.fromarray(overlay).save(out_dir          / f"{p.stem}_overlay.png")
        if args.save_prob:
            Image.fromarray(prob, mode="L").save(out_dir / f"{p.stem}_prob.png")
        print(f"  [{i+1}/{len(paths)}] {p.name}")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
