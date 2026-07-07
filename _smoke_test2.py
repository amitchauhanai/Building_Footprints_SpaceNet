"""Test full training loop: 2 batches train + 2 batches val."""
import time, sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import make_datasets
from model import build_model
from train import BCEDiceLoss, train_epoch, val_epoch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Tiny dataset slice
train_ds, val_ds = make_datasets("../dataset/PS-RGB", "data/masks", "data", 512)

# Use only 4 samples each to be fast
from torch.utils.data import Subset
train_sub = Subset(train_ds, list(range(4)))
val_sub   = Subset(val_ds,   list(range(4)))

train_loader = DataLoader(train_sub, batch_size=2, shuffle=False, num_workers=0)
val_loader   = DataLoader(val_sub,   batch_size=2, shuffle=False, num_workers=0)

print("Loading model with pretrained=True ...")
t0 = time.time()
model = build_model(pretrained=True, device=device)
print(f"  loaded in {time.time()-t0:.1f}s")

criterion = BCEDiceLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("Running 1 train epoch (4 samples)...")
t0 = time.time()
tl, ti, tf = train_epoch(model, train_loader, criterion, optimizer, None, device)
print(f"  train  loss={tl:.4f} iou={ti:.4f} f1={tf:.4f}  ({time.time()-t0:.1f}s)")

print("Running val epoch (4 samples)...")
t0 = time.time()
vl, vi, vf = val_epoch(model, val_loader, criterion, device)
print(f"  val    loss={vl:.4f} iou={vi:.4f} f1={vf:.4f}  ({time.time()-t0:.1f}s)")

print("\nSmoke test 2 PASSED.")
