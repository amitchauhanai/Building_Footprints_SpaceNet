import sys
print("Python:", sys.version)
pkgs = ["torch", "torchvision", "rasterio", "albumentations",
        "shapely", "pyproj", "cv2", "PIL", "numpy", "tifffile", "tensorboard"]
for p in pkgs:
    try:
        mod = __import__(p)
        ver = getattr(mod, "__version__", "?")
        print(f"  OK      {p:<20} {ver}")
    except ImportError:
        print(f"  MISSING {p}")
