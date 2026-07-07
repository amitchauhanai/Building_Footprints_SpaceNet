"""
preprocess.py
=============
Read PolygonWKT_Pix entries from the SpaceNet solutions CSV and rasterise
them into binary PNG masks (255 = building, 0 = background).

No rasterio / GDAL required – polygon burn uses opencv.

Usage
-----
    python preprocess.py            # uses paths from config.py
    python preprocess.py --dry_run  # print stats only, no files written
"""

import argparse
import csv
import re
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config


# ──────────────────────────────────────────────────────────────────────────────
# WKT helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_wkt_polygon(wkt: str) -> list[np.ndarray]:
    """
    Parse a POLYGON ((...)) or MULTIPOLYGON WKT string that contains 3-D
    coordinates (x y z, …) and return a list of numpy int32 arrays shaped
    (N, 1, 2) ready for cv2.fillPoly.

    Returns an empty list when the geometry is EMPTY or unparseable.
    """
    wkt = wkt.strip()
    if not wkt or wkt.upper() in ("POLYGON EMPTY", "MULTIPOLYGON EMPTY", ""):
        return []

    rings = []
    # Extract every parenthesised coordinate list, e.g. "(1.0 2.0 0, 3.0 4.0 0)"
    for ring_str in re.findall(r"\(([^()]+)\)", wkt):
        pts = []
        for coord in ring_str.split(","):
            nums = coord.split()
            if len(nums) >= 2:
                try:
                    pts.append([float(nums[0]), float(nums[1])])
                except ValueError:
                    continue
        if len(pts) >= 3:
            arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
            rings.append(arr)
    return rings


def burn_polygons(
    rings_list: list[list[np.ndarray]],
    height: int,
    width: int,
) -> np.ndarray:
    """Burn a list of ring-lists into a (H, W) uint8 mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for rings in rings_list:
        if rings:
            cv2.fillPoly(mask, rings, color=255)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate building masks from solutions CSV")
    p.add_argument("--csv",        default=str(config.SOLUTIONS_CSV), help="Path to solutions CSV")
    p.add_argument("--mask_dir",   default=str(config.MASK_DIR),      help="Output mask directory")
    p.add_argument("--split_dir",  default=str(config.SPLIT_DIR),     help="Where to write train/val txt")
    p.add_argument("--img_h",      type=int, default=config.IMAGE_H)
    p.add_argument("--img_w",      type=int, default=config.IMAGE_W)
    p.add_argument("--train_split",type=float, default=config.TRAIN_SPLIT)
    p.add_argument("--seed",       type=int,   default=config.RANDOM_SEED)
    p.add_argument("--dry_run",    action="store_true", help="Print stats without writing files")
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mask_dir  = Path(args.mask_dir)
    split_dir = Path(args.split_dir)
    if not args.dry_run:
        mask_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Read CSV and group polygons by image id ──────────────────────────
    # ImageId format: AOI_2_Vegas_imgN
    # We extract the numeric suffix as the canonical id.
    polygons_by_id: dict[str, list] = defaultdict(list)

    print(f"Reading {args.csv} …")
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id_full = row["ImageId"].strip()          # e.g. AOI_2_Vegas_img42
            wkt_pix     = row["PolygonWKT_Pix"].strip()

            # extract numeric id
            m = re.search(r"img(\d+)$", img_id_full, re.IGNORECASE)
            if not m:
                continue
            img_id = m.group(1)
            rings  = parse_wkt_polygon(wkt_pix)
            if rings:
                polygons_by_id[img_id].append(rings)

    all_ids = sorted(polygons_by_id.keys(), key=lambda x: int(x))
    print(f"Found {len(all_ids)} images with building annotations.")

    # Also collect images that exist in RGB dir but have NO buildings
    rgb_dir = config.RGB_DIR
    for tif in sorted(rgb_dir.glob("*PS-RGB_img*.tif")):
        m = re.search(r"img(\d+)\.tif$", tif.name)
        if m and m.group(1) not in polygons_by_id:
            polygons_by_id[m.group(1)] = []   # empty → all-background mask

    all_ids = sorted(polygons_by_id.keys(), key=lambda x: int(x))
    print(f"Total images (incl. background-only): {len(all_ids)}")

    if args.dry_run:
        print("Dry run – no files written.")
        return

    # ── 2. Rasterise masks ──────────────────────────────────────────────────
    n_written = 0
    n_skipped = 0

    for img_id in all_ids:
        out_path = mask_dir / f"mask_img{img_id}.png"

        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue

        mask = burn_polygons(
            polygons_by_id[img_id],
            height=args.img_h,
            width=args.img_w,
        )
        Image.fromarray(mask, mode="L").save(out_path)
        n_written += 1

        if n_written % 200 == 0:
            print(f"  {n_written}/{len(all_ids)} masks written …")

    print(f"Done. Written={n_written}  Skipped={n_skipped}")

    # ── 3. Train / val split ────────────────────────────────────────────────
    random.seed(args.seed)
    shuffled = all_ids.copy()
    random.shuffle(shuffled)

    n_train = int(len(shuffled) * args.train_split)
    train_ids = shuffled[:n_train]
    val_ids   = shuffled[n_train:]

    (split_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (split_dir / "val.txt").write_text("\n".join(val_ids)   + "\n")

    print(f"Train: {len(train_ids)}  |  Val: {len(val_ids)}")
    print(f"Splits written to {split_dir}/{{train,val}}.txt")


if __name__ == "__main__":
    main()
