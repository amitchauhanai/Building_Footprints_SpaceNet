"""
evaluate.py  –  Evaluate a checkpoint on the validation split.

Usage:
    python evaluate.py \
        --checkpoint checkpoints/best.pth \
        --rgb_dir    ../PS-RGB \
        --mask_dir   data/masks \
        --val_txt    data/val.txt \
        --out_dir    outputs \
        --vis_n      20
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from dataset import SpaceNetVegasDataset, load_split_ids, DEFAULT_IMG_SIZE, MEAN, STD
from model   import build_model


def pixel_metrics(pred_bin, gt_bin):
    tp = np.logical_and(pred_bin,  gt_bin).sum()
    tn = np.logical_and(~pred_bin, ~gt_bin).sum()
    fp = np.logical_and(pred_bin,  ~gt_bin).sum()
    fn = np.logical_and(~pred_bin,  gt_bin).sum()
    return dict(
        iou      = float(tp / (tp + fp + fn + 1e-6)),
        f1       = float(2*tp / (2*tp + fp + fn + 1e-6)),
        precision= float(tp / (tp + fp + 1e-6)),
        recall   = float(tp / (tp + fn + 1e-6)),
        accuracy = float((tp + tn) / (tp + tn + fp + fn + 1e-6)),
    )


def denorm(tensor):
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


def make_vis(image_t, gt_t, logit_t, thr=0.5):
    img  = denorm(image_t)
    gt   = (gt_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
    pred = ((torch.sigmoid(logit_t).squeeze().cpu().numpy() > thr) * 255).astype(np.uint8)

    def overlay(base, mask, color):
        out = base.copy(); m = mask > 128
        for c, v in enumerate(color): out[m, c] = v
        return out

    H, W = img.shape[:2]
    canvas = np.zeros((H, W * 3, 3), dtype=np.uint8)
    canvas[:, :W]      = img
    canvas[:, W:2*W]   = overlay(img.copy(), gt,   (0, 200, 0))
    canvas[:, 2*W:3*W] = overlay(img.copy(), pred, (200, 0, 0))
    return Image.fromarray(canvas)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rgb_dir",    required=True)
    p.add_argument("--mask_dir",   required=True)
    p.add_argument("--val_txt",    required=True)
    p.add_argument("--out_dir",    default="outputs")
    p.add_argument("--img_size",   type=int,   default=DEFAULT_IMG_SIZE)
    p.add_argument("--batch",      type=int,   default=8)
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--vis_n",      type=int,   default=20)
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    if args.vis_n > 0: vis_dir.mkdir(exist_ok=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

    model = build_model(pretrained=False, device=device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded epoch {ckpt.get('epoch','?')}  best_iou={ckpt.get('best_val_iou',0):.4f}")

    val_ids    = load_split_ids(args.val_txt)
    val_ds     = SpaceNetVegasDataset(args.rgb_dir, args.mask_dir, val_ids,
                                      img_size=args.img_size, augment=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    print(f"Evaluating {len(val_ds)} samples…")

    results   = []
    vis_count = 0
    sample_i  = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)

            for i in range(imgs.size(0)):
                img_id   = val_ids[sample_i]
                pred_bin = (torch.sigmoid(logits[i]) > args.threshold).cpu().numpy().squeeze().astype(bool)
                gt_bin   = masks[i].cpu().numpy().squeeze().astype(bool)
                m        = pixel_metrics(pred_bin, gt_bin)
                m["img_id"] = img_id
                results.append(m)

                if vis_count < args.vis_n:
                    make_vis(imgs[i], masks[i], logits[i], args.threshold).save(
                        vis_dir / f"img{img_id}_vis.png")
                    vis_count += 1
                sample_i += 1

    keys    = ["iou", "f1", "precision", "recall", "accuracy"]
    summary = {k: float(np.mean([r[k] for r in results])) for k in keys}

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k:12s}: {v:.4f}")

    out_path = out_dir / "eval_results.json"
    out_path.write_text(json.dumps({"summary": summary, "per_image": results}, indent=2))
    print(f"\nResults → {out_path}")
    if args.vis_n > 0: print(f"Vis     → {vis_dir}/  ({vis_count} saved)")


if __name__ == "__main__":
    main()
