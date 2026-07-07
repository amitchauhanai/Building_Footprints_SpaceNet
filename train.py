"""
train.py  –  Train UNetResNet34 on SpaceNet AOI_2_Vegas
Supports: CUDA (with AMP), Apple MPS, CPU

Usage:
    python train.py \
        --rgb_dir   ../dataset/PS-RGB \
        --mask_dir  data/masks \
        --data_dir  data \
        --out_dir   . \
        --epochs 50 --batch 8 --img_size 512
"""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False

from dataset import make_datasets
from model import build_model


# ── Losses ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.w   = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dce = DiceLoss()

    def forward(self, logits, targets):
        return self.w * self.bce(logits, targets) + (1 - self.w) * self.dce(logits, targets)


# ── Metrics ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def iou_f1(logits, targets, thr=0.5):
    p  = (torch.sigmoid(logits) > thr).float()
    tp = (p * targets).sum().item()
    fp = (p * (1 - targets)).sum().item()
    fn = ((1 - p) * targets).sum().item()
    iou = tp / (tp + fp + fn + 1e-6)
    f1  = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return iou, f1


# ── One epoch ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    tot_loss = tot_iou = tot_f1 = 0
    n = 0
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:                         # CUDA + AMP
            from torch.cuda.amp import autocast
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:                                          # MPS / CPU
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        iou, f1 = iou_f1(logits, masks)
        tot_loss += loss.item()
        tot_iou  += iou
        tot_f1   += f1
        n += 1

    return tot_loss / n, tot_iou / n, tot_f1 / n


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss = tot_iou = tot_f1 = 0
    n = 0
    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        iou, f1 = iou_f1(logits, masks)
        tot_loss += loss.item()
        tot_iou  += iou
        tot_f1   += f1
        n += 1
    return tot_loss / n, tot_iou / n, tot_f1 / n


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_dir",       required=True)
    p.add_argument("--mask_dir",      required=True)
    p.add_argument("--data_dir",      required=True)
    p.add_argument("--out_dir",       default=".")
    p.add_argument("--img_size",      type=int,   default=512)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch",         type=int,   default=8)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--bce_weight",    type=float, default=0.5)
    p.add_argument("--workers",       type=int,   default=0)
    p.add_argument("--no_amp",        action="store_true")
    p.add_argument("--no_pretrain",   action="store_true")
    p.add_argument("--freeze_epochs", type=int,   default=0)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--resume",        default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out  = Path(args.out_dir)
    ckpt_dir = out / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = out / "logs";        log_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device : {device}")

    # MPS does not support pin_memory
    pin_mem = (device == "cuda")
    workers = 0 if device in ("mps", "cpu") else args.workers

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds = make_datasets(
        args.rgb_dir, args.mask_dir, args.data_dir, args.img_size
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=workers, pin_memory=pin_mem, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=workers, pin_memory=pin_mem,
    )
    print(f"Train  : {len(train_ds)}  |  Val : {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model(pretrained=not args.no_pretrain, device=device)
    criterion = BCEDiceLoss(bce_weight=args.bce_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    # AMP only on CUDA
    scaler = (torch.cuda.amp.GradScaler()
              if (device == "cuda" and not args.no_amp) else None)

    start_epoch  = 0
    best_val_iou = 0.0
    history      = {"train_loss": [], "val_loss": [], "val_iou": [], "val_f1": []}
    no_improve   = 0

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch  = ck["epoch"] + 1
        best_val_iou = ck.get("best_val_iou", 0.0)
        history      = ck.get("history", history)
        print(f"Resumed from epoch {start_epoch}  best IoU={best_val_iou:.4f}")

    writer = None  # disabled – TensorBoard subprocesses can conflict with MPS
    # writer = SummaryWriter(log_dir=str(log_dir)) if TB_AVAILABLE else None

    print("Starting training loop...")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        if epoch < args.freeze_epochs:
            model.freeze_encoder()
        elif epoch == args.freeze_epochs and epoch > 0:
            model.unfreeze_encoder()
            print(f"[epoch {epoch}] encoder unfrozen")

        tl, ti, tf = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl, vi, vf = val_epoch(model, val_loader, criterion, device)

        scheduler.step(epoch + 1)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:03d}/{args.epochs}  {elapsed:.0f}s  "
            f"| train  loss={tl:.4f}  iou={ti:.4f}  "
            f"| val    loss={vl:.4f}  iou={vi:.4f}  f1={vf:.4f}"
        )

        if writer:
            writer.add_scalars("Loss", {"train": tl, "val": vl}, epoch)
            writer.add_scalars("IoU",  {"train": ti, "val": vi}, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_iou"].append(vi)
        history["val_f1"].append(vf)

        state = dict(
            epoch=epoch, model=model.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            best_val_iou=best_val_iou, history=history, args=vars(args),
        )

        if vi > best_val_iou:
            best_val_iou = vi
            no_improve   = 0
            torch.save({**state, "best_val_iou": best_val_iou}, ckpt_dir / "best.pth")
            print(f"  ✓ new best  iou={best_val_iou:.4f}  → checkpoints/best.pth")
        else:
            no_improve += 1

        torch.save(state, ckpt_dir / "last.pth")

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stop: no improvement for {args.patience} epochs.")
            break

    (log_dir / "history.json").write_text(json.dumps(history, indent=2))
    if writer:
        writer.close()
    print(f"\nDone.  Best val IoU={best_val_iou:.4f}  → {ckpt_dir/'best.pth'}")


if __name__ == "__main__":
    main()
