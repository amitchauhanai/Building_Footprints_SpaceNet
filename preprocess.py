"""
preprocess.py
=============
Convert SpaceNet AOI_2_Vegas GeoJSON building labels into binary PNG masks
that are pixel-aligned with the corresponding PS-RGB GeoTIFF images.

Each output mask is a single-channel uint8 PNG where:
  255 = building footprint
    0 = background

Usage
-----
    python preprocess.py \
        --rgb_dir  /path/to/AOI_2_Vegas/PS-RGB \
        --geojson_dir /path/to/AOI_2_Vegas/geojson_buildings \
        --mask_dir /path/to/building_detection/data/masks \
        --split 0.85          # fraction for train (rest = val)
        --seed 42

Outputs
-------
    data/masks/          -> one PNG per image, same stem as the TIF
    data/train.txt       -> newline-separated image IDs for training
    data/val.txt         -> newline-separated image IDs for validation
"""

import os
import re
import json
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.features import rasterize
    from shapely.geometry import shape, mapping
    from shapely.ops import transform as shp_transform
    import pyproj
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_id(filename: str) -> str:
    """Extract the numeric image ID from a filename like *_imgN.tif / *_imgN.geojson."""
    m = re.search(r'img(\d+)', filename)
    return m.group(1) if m else None


def geojson_to_mask_rasterio(geojson_path: Path, tif_path: Path) -> np.ndarray:
    """
    Burn GeoJSON building polygons into a binary mask using rasterio.
    The mask is spatially aligned with the source TIF.
    Returns a uint8 numpy array (H, W).
    """
    with rasterio.open(tif_path) as src:
        height = src.height
        width = src.width
        transform = src.transform
        crs = src.crs

    with open(geojson_path) as f:
        gj = json.load(f)

    if not gj.get("features"):
        return np.zeros((height, width), dtype=np.uint8)

    # Reproject geometries if needed (GeoJSONs are typically WGS84 / EPSG:4326)
    shapes = []
    for feat in gj["features"]:
        geom = shape(feat["geometry"])
        if crs and str(crs).upper() not in ("EPSG:4326", "WGS84"):
            project = pyproj.Transformer.from_crs(
                "EPSG:4326", crs.to_epsg(), always_xy=True
            ).transform
            geom = shp_transform(project, geom)
        shapes.append((mapping(geom), 1))

    mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    return (mask * 255).astype(np.uint8)


def geojson_to_mask_pixel(geojson_path: Path, tif_path: Path) -> np.ndarray:
    """
    Fallback: burn polygons using pixel coordinates extracted from the TIF's
    geotransform manually (no rasterio dependency beyond reading the transform
    parameters via GDAL through Pillow/TIFF tags).
    Requires opencv-python (cv2).
    """
    if not CV2_AVAILABLE:
        raise RuntimeError(
            "Neither rasterio nor opencv-python is available. "
            "Install at least one: pip install rasterio  OR  pip install opencv-python"
        )

    # Read image size via PIL
    with Image.open(tif_path) as img:
        width, height = img.size

    # Try to get geotransform from TIFF tags
    try:
        import tifffile
        with tifffile.TiffFile(str(tif_path)) as tif:
            tags = {t.name: t.value for t in tif.pages[0].tags.values()}
        model_tiepoint = tags.get("ModelTiepointTag")   # [i,j,k, x,y,z]
        model_pixel    = tags.get("ModelPixelScaleTag")  # [sx, sy, sz]
        if model_tiepoint is None or model_pixel is None:
            raise ValueError("No geotransform tags found")
        # Affine: x = origin_x + col * sx
        #         y = origin_y - row * sy
        tp = model_tiepoint
        origin_x = tp[3]
        origin_y = tp[4]
        sx = model_pixel[0]
        sy = model_pixel[1]
    except Exception:
        # Last resort: assume full image extent is [0,1]x[0,1] – won't be
        # geographically correct but allows shapes to be in image space.
        origin_x, origin_y = 0.0, 1.0
        sx = 1.0 / width
        sy = 1.0 / height

    def geo_to_pixel(x, y):
        col = (x - origin_x) / sx
        row = (origin_y - y) / sy
        return col, row

    with open(geojson_path) as f:
        gj = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)
    if not gj.get("features"):
        return mask

    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            rings = geom["coordinates"]
        elif geom["type"] == "MultiPolygon":
            rings = [r for poly in geom["coordinates"] for r in poly]
        else:
            continue

        for ring in rings:
            pts = np.array([geo_to_pixel(c[0], c[1]) for c in ring],
                           dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


def build_mask(geojson_path: Path, tif_path: Path) -> np.ndarray:
    """Try rasterio first, fall back to pixel-based method."""
    if RASTERIO_AVAILABLE:
        return geojson_to_mask_rasterio(geojson_path, tif_path)
    return geojson_to_mask_pixel(geojson_path, tif_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate building masks from GeoJSON labels")
    p.add_argument("--rgb_dir",     required=True,  help="Path to PS-RGB TIF directory")
    p.add_argument("--geojson_dir", required=True,  help="Path to geojson_buildings directory")
    p.add_argument("--mask_dir",    required=True,  help="Output directory for PNG masks")
    p.add_argument("--split_dir",   default=None,   help="Output dir for train.txt/val.txt (default: parent of mask_dir)")
    p.add_argument("--split",       type=float, default=0.85, help="Train fraction (default 0.85)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--skip_existing", action="store_true", help="Skip images where mask already exists")
    return p.parse_args()


def main():
    args = parse_args()

    rgb_dir     = Path(args.rgb_dir)
    geojson_dir = Path(args.geojson_dir)
    mask_dir    = Path(args.mask_dir)
    split_dir   = Path(args.split_dir) if args.split_dir else mask_dir.parent

    mask_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    # Collect all PS-RGB TIF files
    tif_files = sorted(rgb_dir.glob("*PS-RGB_img*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No PS-RGB TIF files found in {rgb_dir}")

    print(f"Found {len(tif_files)} PS-RGB images.")

    processed_ids = []
    errors = []

    for idx, tif_path in enumerate(tif_files):
        img_id = get_image_id(tif_path.name)
        if img_id is None:
            print(f"  [SKIP] Cannot parse image ID from: {tif_path.name}")
            continue

        mask_path = mask_dir / f"mask_img{img_id}.png"

        if args.skip_existing and mask_path.exists():
            processed_ids.append(img_id)
            continue

        # Find matching GeoJSON
        geojson_path = geojson_dir / f"SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{img_id}.geojson"
        if not geojson_path.exists():
            # No annotation → all-background mask
            with Image.open(tif_path) as img:
                w, h = img.size
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            try:
                mask = build_mask(geojson_path, tif_path)
            except Exception as e:
                print(f"  [ERROR] img{img_id}: {e}")
                errors.append(img_id)
                continue

        Image.fromarray(mask, mode="L").save(mask_path)
        processed_ids.append(img_id)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(tif_files):
            pct = 100 * (idx + 1) / len(tif_files)
            print(f"  [{idx+1}/{len(tif_files)}] {pct:.0f}%  last: img{img_id}")

    # Train / val split
    random.seed(args.seed)
    random.shuffle(processed_ids)
    n_train = int(len(processed_ids) * args.split)
    train_ids = processed_ids[:n_train]
    val_ids   = processed_ids[n_train:]

    (split_dir / "train.txt").write_text("\n".join(train_ids) + "\n")
    (split_dir / "val.txt").write_text("\n".join(val_ids) + "\n")

    print(f"\nDone.")
    print(f"  Total processed : {len(processed_ids)}")
    print(f"  Train           : {len(train_ids)}")
    print(f"  Val             : {len(val_ids)}")
    print(f"  Errors skipped  : {len(errors)}")
    print(f"  Masks saved to  : {mask_dir}")
    print(f"  Splits saved to : {split_dir}")


if __name__ == "__main__":
    main()
