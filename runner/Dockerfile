# illama-runner: Lightweight Inference Server
# Uses OpenVINO GenAI for LLM inference with OpenAI-compatible API
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Image version, baked into the runtime so /v1/models, /, /api/version, and
# the FastAPI metadata all reflect the build that produced this image.
# setup.sh reads VERSION at the project root and passes it via --build-arg.
ARG ILLAMA_VERSION=2.6.0
ENV ILLAMA_VERSION=${ILLAMA_VERSION}
LABEL org.opencontainers.image.title="illama-runner"
LABEL org.opencontainers.image.version="${ILLAMA_VERSION}"

# Install system dependencies + Intel GPU compute runtime
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    curl \
    wget \
    gpg-agent \
    software-properties-common \
    # Intel GPU compute dependencies
    ocl-icd-libopencl1 \
    clinfo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Intel graphics PPA and install compute runtime
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
    tee /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y \
    intel-opencl-icd \
    level-zero \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the pinned runtime stack into a dedicated venv so we keep the
# system Python untouched (Ubuntu 24.04 marks it externally-managed) and
# avoid the `--break-system-packages` warning. Putting the venv on PATH
# means the existing `CMD ["python3", ...]` and the HEALTHCHECK keep
# working without changes.
#
# pillow + requests + numpy are needed for VLM image inputs:
#   - pillow decodes the bytes (PNG/JPEG/WebP) the client sends as base64 data URLs
#     or HTTP(S) URLs into PIL.Image.
#   - requests fetches https:// URLs with a hard timeout and size cap.
#   - numpy is a transitive dep of openvino but pinning here avoids surprise upgrades.
#
# OpenVINO, openvino-genai, and openvino-tokenizers are one build.
# Both wheels MUST come from the same release to avoid binary ABI mismatches.
ENV VIRTUAL_ENV=/opt/illama-venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Stable OpenVINO 2026.4.1 (PyPI, 2026-10-01). These three MUST move together:
# they are one build, and mixing release families breaks the binary ABI.
# Validated on Arc Pro B50: EAGLE-3, continuous batching, and VLM inference.
# Bump deliberately, together, and re-benchmark after.
RUN pip install --no-cache-dir \
    "openvino==2026.4.1" \
    "openvino-genai==2026.4.1.0" \
    "openvino-tokenizers==2026.4.1.0" \
    "fastapi==0.136.1" \
    "uvicorn[standard]==0.46.0" \
    "pydantic==2.13.3" \
    "pillow==11.0.0" \
    "requests==2.32.3" \
    "numpy==2.1.3"

# Copy server code. We use a glob so the helper modules (_helpers.py,
# _schemas.py) ship alongside server.py - server.py imports them at module
# load. .dockerignore in the build context filters out __pycache__ and other
# noise so the final image stays small.
WORKDIR /app
COPY *.py /app/

# Environment defaults
ENV ILLAMA_MODELS=/models
ENV ILLAMA_PORT=8000
ENV ILLAMA_DEVICE=GPU
ENV ILLAMA_IDLE_TIMEOUT=60
ENV ILLAMA_CACHE_DIR=/cache

# Expose port
EXPOSE 8000

# Allow GPU model cache reuse between restarts and reloads.
# The server passes CACHE_DIR=$ILLAMA_CACHE_DIR to the OpenVINO pipeline at load time.
RUN mkdir -p /cache

# Report unhealthy status. Docker restart policy alone restarts exited processes,
# not an unhealthy container whose process is still running.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://localhost:${ILLAMA_PORT:-8000}/health" || exit 1

# Run server
CMD ["python3", "/app/server.py"]
