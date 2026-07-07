"""Debug: iterate the full training DataLoader without a model to find which batch hangs."""
import sys, time
sys.path.insert(0, '.')
from dataset import make_datasets
from torch.utils.data import DataLoader

train_ds, val_ds = make_datasets("../dataset/PS-RGB", "data/masks", "data", 512)
print(f"Train size: {len(train_ds)}  Val size: {len(val_ds)}")

loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)

t0 = time.time()
for i, (imgs, masks) in enumerate(loader):
    if (i+1) % 50 == 0 or (i+1) == 1:
        print(f"  batch {i+1}/{len(loader)}  imgs={imgs.shape}  {time.time()-t0:.1f}s")
    if i >= 10:   # just test first 10 batches
        print("First 10 batches OK.")
        break

print("Done.")
