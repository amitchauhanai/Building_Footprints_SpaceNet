"""Test one training step with the full train_epoch function using increasing batch sizes."""
import sys, time, torch, torch.nn as nn
sys.path.insert(0, '.')
from dataset import make_datasets
from model import build_model
from train import BCEDiceLoss
from torch.utils.data import DataLoader, Subset

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

train_ds, _ = make_datasets("../dataset/PS-RGB", "data/masks", "data", 512)

model     = build_model(pretrained=True, device=device)
criterion = BCEDiceLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# test with increasing number of samples to find where it hangs
for n_samples in [8, 32, 64, 128]:
    sub     = Subset(train_ds, list(range(n_samples)))
    loader  = DataLoader(sub, batch_size=8, shuffle=False, num_workers=0)
    model.train()
    print(f"\nTesting {n_samples} samples ({len(loader)} batches)...")
    t0 = time.time()
    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(f"  batch {i+1}/{len(loader)}  loss={loss.item():.4f}  {time.time()-t0:.1f}s")

print("\nAll sizes OK.")
