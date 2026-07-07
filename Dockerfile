# ──────────────────────────────────────────────────────────────────────────
# Stage 1: builder  – install Python deps in an isolated layer
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

# System deps needed to compile rasterio / GDAL wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin \
        libgdal-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy only requirements so Docker caches this layer unless deps change
COPY requirements-api.txt .

# Install everything into a prefix so we can COPY it cleanly to the runtime stage
RUN pip install --no-cache-dir --prefix=/runtime -r requirements-api.txt


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: runtime  – lean final image
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="amitchauhanai"
LABEL description="Building segmentation inference API — U-Net + ResNet-34 / SpaceNet AOI_2_Vegas"

# Runtime system libs (GDAL, OpenCV headless, libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin \
        libgdal-dev \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /runtime /usr/local

WORKDIR /app

# Copy source files
COPY model.py       .
COPY dataset.py     .
COPY app.py         .

# Checkpoints are mounted at runtime (see docker-compose.yml)
# but we create the directory so the container starts without errors
RUN mkdir -p checkpoints outputs/predictions

# ── Environment defaults (override with -e or docker-compose) ──────────────
ENV CHECKPOINT_PATH=checkpoints/best.pth
ENV IMG_SIZE=512
ENV THRESHOLD=0.5
ENV DEVICE=cpu
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Non-root user for security
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --no-create-home appuser
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

# Tini-style entrypoint with exec form for proper signal handling
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", \
            "--host", "0.0.0.0", \
            "--port", "8000", \
            "--workers", "1", \
            "--log-level", "info"]
