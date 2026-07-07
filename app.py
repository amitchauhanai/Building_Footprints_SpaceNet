"""
app.py  –  FastAPI inference server for UNetResNet34 building segmentation
==========================================================================

Endpoints
---------
GET  /                    Health check + model info
GET  /health              Liveness probe (k8s / Docker)
POST /predict             Upload a GeoTIFF or PNG → returns binary mask PNG
POST /predict/overlay     Upload a GeoTIFF or PNG → returns RGB overlay PNG
POST /predict/full        Upload a GeoTIFF or PNG → returns JSON with base64
                          encoded mask, overlay, and probability map

Environment variables
---------------------
CHECKPOINT_PATH   path to best.pth          default: checkpoints/best.pth
IMG_SIZE          network input size (px)    default: 512
THRESHOLD         sigmoid threshold          default: 0.5
DEVICE            cuda | mps | cpu | auto    default: auto
"""

import base64
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel

from dataset import MEAN, STD
from model import build_model

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Config from environment ────────────────────────────────────────────────
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "checkpoints/best.pth")
IMG_SIZE        = int(os.getenv("IMG_SIZE",   "512"))
THRESHOLD       = float(os.getenv("THRESHOLD", "0.5"))
DEVICE_ENV      = os.getenv("DEVICE",         "auto")


# ── Model singleton ────────────────────────────────────────────────────────
class ModelState:
    model     = None
    device    = None
    epoch     = None
    best_iou  = None
    load_time = None


_state = ModelState()


def get_device(preference: str) -> str:
    if preference == "auto":
        if torch.cuda.is_available():    return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return preference


def load_model() -> None:
    """Load the checkpoint once at startup and cache in _state."""
    ckpt_path = Path(CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path.resolve()}. "
            "Set CHECKPOINT_PATH env var or place best.pth in checkpoints/."
        )

    device = get_device(DEVICE_ENV)
    log.info(f"Loading checkpoint from {ckpt_path}  on device={device}")

    t0   = time.time()
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(pretrained=False, device=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _state.model     = model
    _state.device    = device
    _state.epoch     = ckpt.get("epoch", "?")
    _state.best_iou  = ckpt.get("best_val_iou", 0.0)
    _state.load_time = time.time() - t0

    log.info(
        f"Model ready  epoch={_state.epoch}  "
        f"best_val_iou={_state.best_iou:.4f}  "
        f"loaded in {_state.load_time:.2f}s"
    )


# ── App lifecycle ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Building Segmentation API",
    description=(
        "U-Net + ResNet-34 binary building segmentation "
        "trained on SpaceNet AOI_2_Vegas (WorldView-3 PS-RGB, 30 cm/px)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Image preprocessing / postprocessing ──────────────────────────────────

def read_image_bytes(data: bytes) -> np.ndarray:
    """
    Accept GeoTIFF (via rasterio) or any PIL-supported format.
    Returns (H, W, 3) uint8 numpy array.
    """
    # Try rasterio first (handles GeoTIFF multi-band)
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(data) as mem_file:
            with mem_file.open() as src:
                arr = src.read()[:3]                          # (3, H, W)
                arr = np.transpose(arr, (1, 2, 0))            # (H, W, 3)
                if arr.dtype != np.uint8:
                    lo, hi = arr.min(), arr.max()
                    arr = (
                        (arr.astype(np.float32) - lo) / (hi - lo + 1e-6) * 255
                    ).astype(np.uint8)
        return arr
    except Exception:
        pass
    # Fallback: PIL
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def preprocess_image(image: np.ndarray, img_size: int) -> torch.Tensor:
    """Resize → ImageNet normalise → (1, 3, H, W) tensor."""
    mean = np.array(MEAN, dtype=np.float32)
    std  = np.array(STD,  dtype=np.float32)
    img  = np.array(
        Image.fromarray(image).resize((img_size, img_size), Image.BILINEAR),
        dtype=np.float32,
    )
    img  = (img / 255.0 - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)


def run_inference(
    image_arr: np.ndarray,
    img_size: int   = IMG_SIZE,
    threshold: float = THRESHOLD,
):
    """
    Returns:
        binary_mask  (H, W) uint8  {0, 255}
        prob_map     (H, W) uint8  {0–255}
        overlay      (H, W, 3) uint8
    """
    orig_h, orig_w = image_arr.shape[:2]
    tensor = preprocess_image(image_arr, img_size).to(_state.device)

    with torch.no_grad():
        logit = _state.model(tensor)                          # (1,1,h,w)

    prob = torch.sigmoid(logit.squeeze())
    prob = F.interpolate(
        prob.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    prob_np  = prob.cpu().numpy()
    bin_mask = (prob_np > threshold).astype(np.uint8) * 255
    prob_u8  = (prob_np * 255).clip(0, 255).astype(np.uint8)

    # Red building overlay
    overlay  = image_arr.astype(np.float32).copy()
    m        = bin_mask > 128
    overlay[m, 0] = overlay[m, 0] * 0.55 + 220 * 0.45
    overlay[m, 1] = overlay[m, 1] * 0.55 +  30 * 0.45
    overlay[m, 2] = overlay[m, 2] * 0.55 +  30 * 0.45
    overlay = overlay.clip(0, 255).astype(np.uint8)

    return bin_mask, prob_u8, overlay


def array_to_png_bytes(arr: np.ndarray, mode: str = "L") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def array_to_b64(arr: np.ndarray, mode: str) -> str:
    return base64.b64encode(array_to_png_bytes(arr, mode)).decode()


# ── Pydantic response schemas ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:    str
    device:    str
    epoch:     object
    best_iou:  float
    img_size:  int
    threshold: float


class PredictFullResponse(BaseModel):
    width:        int
    height:       int
    threshold:    float
    building_pct: float           # % of pixels classified as building
    mask_png_b64:    str          # base64-encoded binary mask PNG
    overlay_png_b64: str          # base64-encoded overlay PNG
    prob_png_b64:    str          # base64-encoded probability map PNG
    inference_ms:    float


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Liveness / readiness probe — returns model metadata."""
    if _state.model is None:
        raise HTTPException(503, "Model not loaded yet.")
    return HealthResponse(
        status    = "ok",
        device    = _state.device,
        epoch     = _state.epoch,
        best_iou  = round(_state.best_iou, 4),
        img_size  = IMG_SIZE,
        threshold = THRESHOLD,
    )


@app.post(
    "/predict",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    tags=["Inference"],
    summary="Returns binary mask PNG (building=255, background=0)",
)
async def predict_mask(
    file:      UploadFile = File(..., description="GeoTIFF or PNG satellite image"),
    threshold: float      = Query(THRESHOLD, ge=0.0, le=1.0),
    img_size:  int        = Query(IMG_SIZE,  ge=128, le=1024),
):
    if _state.model is None:
        raise HTTPException(503, "Model not loaded.")

    t0 = time.time()
    data      = await file.read()
    image_arr = read_image_bytes(data)
    binary, _, _ = run_inference(image_arr, img_size, threshold)
    elapsed   = (time.time() - t0) * 1000

    log.info(f"predict  {file.filename}  {image_arr.shape[:2]}  {elapsed:.0f}ms")
    return Response(
        content      = array_to_png_bytes(binary, "L"),
        media_type   = "image/png",
        headers      = {"X-Inference-Ms": f"{elapsed:.1f}"},
    )


@app.post(
    "/predict/overlay",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    tags=["Inference"],
    summary="Returns RGB overlay PNG with building footprints in red",
)
async def predict_overlay(
    file:      UploadFile = File(...),
    threshold: float      = Query(THRESHOLD, ge=0.0, le=1.0),
    img_size:  int        = Query(IMG_SIZE,  ge=128, le=1024),
):
    if _state.model is None:
        raise HTTPException(503, "Model not loaded.")

    t0 = time.time()
    data      = await file.read()
    image_arr = read_image_bytes(data)
    _, _, overlay = run_inference(image_arr, img_size, threshold)
    elapsed   = (time.time() - t0) * 1000

    log.info(f"overlay  {file.filename}  {image_arr.shape[:2]}  {elapsed:.0f}ms")
    return Response(
        content    = array_to_png_bytes(overlay, "RGB"),
        media_type = "image/png",
        headers    = {"X-Inference-Ms": f"{elapsed:.1f}"},
    )


@app.post(
    "/predict/full",
    response_model=PredictFullResponse,
    tags=["Inference"],
    summary="Returns JSON with base64-encoded mask, overlay, and probability map",
)
async def predict_full(
    file:      UploadFile = File(...),
    threshold: float      = Query(THRESHOLD, ge=0.0, le=1.0),
    img_size:  int        = Query(IMG_SIZE,  ge=128, le=1024),
):
    if _state.model is None:
        raise HTTPException(503, "Model not loaded.")

    t0 = time.time()
    data      = await file.read()
    image_arr = read_image_bytes(data)
    binary, prob, overlay = run_inference(image_arr, img_size, threshold)
    elapsed   = (time.time() - t0) * 1000

    h, w = image_arr.shape[:2]
    building_pct = float((binary > 128).sum()) / (h * w) * 100

    log.info(
        f"full  {file.filename}  {image_arr.shape[:2]}  "
        f"building={building_pct:.1f}%  {elapsed:.0f}ms"
    )
    return PredictFullResponse(
        width            = w,
        height           = h,
        threshold        = threshold,
        building_pct     = round(building_pct, 2),
        mask_png_b64     = array_to_b64(binary,  "L"),
        overlay_png_b64  = array_to_b64(overlay, "RGB"),
        prob_png_b64     = array_to_b64(prob,    "L"),
        inference_ms     = round(elapsed, 1),
    )


# ── Entrypoint ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
