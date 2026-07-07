"""Quick smoke test: one forward + backward pass, then one batch from the real dataset."""
import time, torch, sys
sys.path.insert(0, '.')

print("1. Importing model...")
from model import build_model

print("2. Building model on MPS...")
t0 = time.time()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model  = build_model(pretrained=False, device=device)
print(f"   done in {time.time()-t0:.1f}s  device={device}")

print("3. Forward pass (2,3,512,512)...")
t0 = time.time()
x  = torch.randn(2, 3, 512, 512, device=device)
y  = model(x)
print(f"   out shape: {y.shape}  ({time.time()-t0:.1f}s)")

print("4. Backward pass...")
t0 = time.time()
loss = y.mean()
loss.backward()
print(f"   done in {time.time()-t0:.1f}s")

print("5. Loading one batch from dataset...")
from dataset import make_datasets
from torch.utils.data import DataLoader
t0 = time.time()
train_ds, _ = make_datasets("../dataset/PS-RGB", "data/masks", "data", 512)
loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
imgs, masks = next(iter(loader))
print(f"   batch: imgs={imgs.shape} masks={masks.shape}  ({time.time()-t0:.1f}s)")

print("\nAll checks passed.")
