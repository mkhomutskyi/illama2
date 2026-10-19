# illama-builder: Ephemeral Model Converter
# Converts HuggingFace models to OpenVINO IR format (INT4)
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive

# Image version, set from the project-root VERSION file via --build-arg.
ARG ILLAMA_VERSION=2.6.0
ENV ILLAMA_VERSION=${ILLAMA_VERSION}
LABEL org.opencontainers.image.title="illama-builder"
LABEL org.opencontainers.image.version="${ILLAMA_VERSION}"

# Install build tools
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-venv \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the pinned conversion stack into a dedicated venv so optimum-cli
# is on PATH without using --break-system-packages on the externally-
# managed Ubuntu 24.04 system Python.
ENV VIRTUAL_ENV=/opt/illama-venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# optimum-intel 2.2.0 carries Eagle3 / DFlash export support and allows
# transformers>=4.51,<5.6. Pin OpenVINO to the same 2026.4.1 build as the
# runner. openvino-genai is an inference runtime; export does not need it.
# `datasets` is required for data-aware INT4 compression (AWQ / scale
# estimation). einops / sentencepiece / protobuf are imported by
# trust-remote-code modeling files (including AngelSlim eagle3 drafts).
# transformers stays inside the optimum-intel window. 4.57.6 is the pin
# used for the validated exports; do not float to 5.x outside that window.
RUN pip install --no-cache-dir \
    "openvino==2026.4.1" \
    "openvino-tokenizers==2026.4.1.0" \
    "optimum-intel[openvino]==2.2.0" \
    "transformers==4.57.6" \
    "datasets" \
    "einops" \
    "sentencepiece" \
    "protobuf" \
    "accelerate==1.13.0"

# Default entrypoint for model export. PATH above puts the venv first, so
# `optimum-cli` resolves to /opt/illama-venv/bin/optimum-cli.
ENTRYPOINT ["optimum-cli", "export", "openvino"]
