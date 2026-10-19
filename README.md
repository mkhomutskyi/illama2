# Illama 2

An Ollama-like CLI experience for running LLMs with OpenVINO on Intel hardware.

Run local models through OpenAI-compatible and Ollama-compatible APIs, with a
separate model converter and a lightweight inference container. Apache-2.0;
model weights retain their own licenses.

Current source release: **2.6.0**, including the October 18 benchmark/setup
updates and October 19 publication maintenance. Stable runtime and optional
nightly profiles remain separately pinned; see [CHANGELOG.md](CHANGELOG.md).

| Start here | Contents |
|---|---|
| [Arc Pro B50 quickstart](docs/arc-b50-quickstart.md) | Build, select the correct GPU, configure, infer, and roll back |
| [October 18, 2026 measurements](docs/benchmarks/2026-10-18-b50.md) | Three models, exact nightly versions, reproducible protocol and limitations |
| [ComfyUI on B50](docs/comfyui-b50.md) | Independent Intel XPU image, existing-model tests and VRAM limits |
| [Security policy](SECURITY.md) | Credential handling, trusted-network scope and reporting |

The default build uses **OpenVINO 2026.4.1**. The measured **2026.5 nightly** is
an explicit [optional profile](runner/Dockerfile.nightly), not a silent change
to the stable builder or runner. Verified GPU inference does not prove that
every model architecture can be exported or that every context fits in VRAM.

## Architecture and release history

- Current architecture: decoupled `builder` + `runner` + `illama` CLI, with no active OVMS dependency.
- Runtime pins: OpenVINO **2026.4.1**, GenAI **2026.4.1.0**, optimum-intel **2.2.0**. Arc Pro B50 notes: [docs/b50-model-notes.md](docs/b50-model-notes.md).
- Completed in the current refactor:
  - `config/server.json` is the active server config surface
  - tokenizer-aware chat prompt handling
  - real SSE streaming for `stream=true`
  - exact pinned dependency versions in both Dockerfiles
  - CLI health polling and stale OVMS cleanup
  - OpenAI-style error objects and request limit knobs
- Runtime validation completed on the Ubuntu host:
  - rebuilt images successfully
  - verified `illama serve`, `illama ps`, `/illama/health`, and `/v1` paths
  - verified OpenWebUI-compatible OpenAI API behavior
  - confirmed existing downloaded models are reused from `/opt/illama/models`
- 2026-05 hardening pass:
  - GPU model cache wired through to the OpenVINO pipeline via `CACHE_DIR=$ILLAMA_CACHE_DIR` (the previous `OV_GPU_CACHE_MODEL=1` env var was a no-op)
  - dependency bumps: openvino 2026.1.0, openvino-genai 2026.1.0.0, transformers 4.57.6, accelerate 1.13.0, fastapi 0.136.1, uvicorn 0.46.0, pydantic 2.13.3
  - timing-safe bearer-token comparison via `secrets.compare_digest`
  - bounded SSE event queue and tightened Prometheus label escaping
  - CLI no longer splices unescaped values into Python source or curl `-d` JSON bodies, and `serve` cleans up stale exited containers before starting
- 2026-05 v2.2 additions:
  - **Ollama-compatible API** (`/api/version`, `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/ps`) — drop-in for Continue.dev, LangChain `ChatOllama`, OpenWebUI's Ollama backend, etc.
  - OpenAI: `seed`, `response_format=json_object`, and `frequency_penalty → repetition_penalty` mapping
  - Stats persistence to `${ILLAMA_CACHE_DIR}/illama_stats.json` so counters survive restarts
  - Docker `HEALTHCHECK` reports an unhealthy runner; the restart policy handles process exits. Restarting an unhealthy but still-running process requires operator/orchestrator action.
  - Lazy `illama-builder` image — `setup.sh` only builds the runner; the conversion image is built on first non-OpenVINO `illama pull`
  - New CLI commands: `illama logs [N]`, `illama bench`, `illama prune builder`
  - Removed `bench.sh` in favour of `illama bench` → `scripts/benchmark_api.py`

## Deployment (Ubuntu + Docker)

Two supported entry points run **the same container** with the same volumes and env: pick whichever fits your ops model.

| Path | When to use it | Files |
|---|---|---|
| `illama serve` (bash CLI) | Single host, manual control, `illama` symlinked into `/usr/local/bin` | none beyond the install |
| `docker compose up -d` | Compose-based stacks, GitOps, CI/CD, pinning to a specific image tag | `deploy/docker-compose.yml`, `deploy/.env` |
| `systemctl start illama-runner` | Auto-start on boot, journalctl logs, layered on top of compose | `deploy/illama-runner.service` |

All three are interchangeable — `illama stop` + `docker compose up -d` (or vice-versa) works fine as long as the container name (`illama-service`) and host port match.

### Prerequisites

```bash
# 1. System packages
sudo apt update
sudo apt install -y docker.io docker-compose-plugin python3 curl

# 2. Docker group (so you don't need sudo for `docker ...`)
sudo usermod -aG docker "$USER"
# log out / log back in (or run `newgrp docker`) so it takes effect

# 3. Confirm Docker is usable
docker info >/dev/null && echo "docker OK"
docker compose version          # need v2 plugin (Ubuntu 22.04+ ships it)
```

For Intel GPU acceleration (any iGPU/dGPU with OpenVINO support):

```bash
# /dev/dri/* device nodes must exist
ls /dev/dri
# expected: card0  renderD128

# host user needs video + render group membership for /dev/dri access
sudo usermod -aG video,render "$USER"
# log out / log back in so it takes effect

# confirm
groups | grep -oE 'video|render'

# (optional) verify Intel compute runtime is installed on the host - the
# container ships its own copy, so this is just a sanity check
clinfo 2>/dev/null | grep -i 'intel.*gpu' || echo "(host clinfo not installed; container will provide its own)"
```

If you have **no GPU**, set `ILLAMA_DEVICE=CPU` (much slower; works for smoke tests).

### Fresh install

```bash
# 1. Clone the repo onto the host
git clone https://github.com/mkhomutskyi/illama2.git ~/illama2
cd ~/illama2

# 2. Run the installer. This:
#      - creates /opt/illama/{models,cache,config,deploy,...}
#      - copies the runner sources, server config, deploy artefacts
#      - builds illama-runner:latest + illama-runner:<version>
#      - drops a symlink at /usr/local/bin/illama
chmod +x setup.sh
./setup.sh

# 3. Pull a pre-converted OpenVINO model (~2-5 min depending on size)
illama pull OpenVINO/Phi-4-int4-ov

# 4. Pick ONE of the entry points below.
```

#### Entry point A — `illama serve` (bash CLI)

```bash
illama serve                      # starts illama-service container
illama ps                         # status + loaded model
illama run Phi-4-int4-ov -p "Hi"  # smoke-test inference
illama logs                       # tail server logs
illama stop                       # stop container
```

#### Entry point B — Docker Compose

`setup.sh` copies a ready-to-go compose file and an `.env` (seeded from `.env.example` on first install) to `/opt/illama/deploy/`.

```bash
cd /opt/illama/deploy

# 1. (optional) review/tune the env. Defaults match `illama serve`.
$EDITOR .env

# 2. Bring it up
docker compose up -d

# 3. Verify
docker compose ps
docker compose logs -f illama-runner   # Ctrl-C to detach
curl -s http://localhost:11434/illama/health | python3 -m json.tool

# 4. Stop / restart on config change
docker compose down
docker compose up -d
```

The compose file mirrors `illama serve` exactly: same image, same volumes (`/opt/illama/{models:ro,config:ro,cache}`), same `--device /dev/dri`, same `--group-add` for video/render, same healthcheck. See [`deploy/docker-compose.yml`](deploy/docker-compose.yml) for the canonical reference.

#### Entry point C — systemd (auto-start on boot)

Layered on top of compose; the unit just runs `docker compose up -d` on boot and `down` on stop.

```bash
sudo cp /opt/illama/deploy/illama-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now illama-runner

systemctl status illama-runner
journalctl -u illama-runner -f          # live logs
```

Switch back to manual control:

```bash
sudo systemctl disable --now illama-runner
illama serve
```

### Compose parameter reference

Every knob is defined in `deploy/.env.example` and consumed by `deploy/docker-compose.yml`. Copy the example to `deploy/.env` (setup.sh does this for you on first install) and edit:

| Variable | Default | Purpose |
|---|---|---|
| `ILLAMA_RUNNER_TAG` | `latest` | Image tag to launch. Pin to a version (`2.3.0`) for immutable deploys. |
| `ILLAMA_HOME` | `/opt/illama` | Base dir holding models, config, cache. |
| `ILLAMA_PORT` | `11434` | Host-side port; matches Ollama's default for OpenWebUI compatibility. |
| `ILLAMA_DEVICE` | `GPU` | `GPU` (Intel via OpenVINO) or `CPU`. |
| `HOST_VIDEO_GID` | `44` | GID of host `video` group; check with `getent group video`. |
| `HOST_RENDER_GID` | `110` | GID of host `render` group; check with `getent group render`. |
| `ILLAMA_IDLE_TIMEOUT` | `(falls back to config)` | When to unload model from VRAM. `300`, `300s`, `5m`, or `0` to disable. |
| `ILLAMA_API_KEY` | empty | Optional bearer-token auth. Generate with `openssl rand -hex 32`. |
| `ILLAMA_ALLOW_PRIVATE_IMAGE_URLS` | empty | `1` to disable VLM-image SSRF guard (only on trusted private networks). |
| `ILLAMA_VERSION` | `(baked at build)` | Override the version surfaced by `/api/version`. |

Container-internal settings live in `/opt/illama/config/server.json` (`max_new_tokens_chat`, `max_input_tokens`, `allow_stream`, etc.) — see [Configuration](#configuration). The compose file mounts `config/` read-only, so edit on the host and `docker compose restart illama-runner` to pick up changes.

### Smoke test

After any of the three entry points are up:

```bash
# 1. Health (process + model status)
curl -s http://localhost:11434/illama/health | python3 -m json.tool

# 2. Confirm CACHE_DIR is wired through (saves ~10s+ on subsequent loads)
docker logs illama-service 2>&1 | grep -E 'Cache dir|Pipeline properties'
#   Expected: "[INFO] Cache dir: /cache"
#             "[INFO] Pipeline properties: {'CACHE_DIR': '/cache'}"

# 3. OpenAI-compatible non-stream + stream chat
illama run Phi-4-int4-ov -p "Reply with: hello"
illama run Phi-4-int4-ov -p "Count from 1 to 5" --stream

# 4. Prometheus metrics
curl -s http://localhost:11434/metrics | head -20

# 5. (recommended) host-level diagnostics
illama doctor
```

If anything is off, `illama doctor` checks Docker, GPU device nodes, group membership, registry contents, and reachability of the runner — and prints concrete fixes.

### Upgrade an existing deployment

`setup.sh` is idempotent. Models in `/opt/illama/models` and the GPU compile cache in `/opt/illama/cache` are preserved. Live `config/server.json` is preserved too — if the shipped defaults change, the new file lands at `config/server.json.new` so you can diff and merge.

`setup.sh` copies this repo's Dockerfiles over `/opt/illama` and rebuilds the runner from those files. It does not bump libraries past the pins in the Dockerfiles, and it does not rebuild the builder image. To move OpenVINO or optimum-intel to a newer release, follow [How to update runtime libraries](#how-to-update-runtime-libraries).

```bash
# 1. Sync new sources
cd ~/illama2
git pull

# 2. Re-run the installer. Rebuilds runner image with the new VERSION tag.
./setup.sh

# 3. Restart whichever entry point you use
illama stop && illama serve
#   or
cd /opt/illama/deploy && docker compose up -d --force-recreate
#   or
sudo systemctl restart illama-runner

# 4. Verify
illama ps
illama doctor
```

### Rollback

`setup.sh` tags every built image as both `illama-runner:latest` and `illama-runner:<version>` so a rollback is one tag move:

```bash
# Pin :latest back to the previous version
docker tag illama-runner:2.1.0 illama-runner:latest
illama stop && illama serve

# Or via compose, point .env at the older tag
echo 'ILLAMA_RUNNER_TAG=2.1.0' >> /opt/illama/deploy/.env
cd /opt/illama/deploy && docker compose up -d
```

To revert to an older source tree as well:

```bash
cd ~/illama2
git checkout <previous-tag>
./setup.sh
```

### Uninstall

```bash
illama stop                                                       # or: docker compose down
sudo systemctl disable --now illama-runner 2>/dev/null || true
sudo rm -f /etc/systemd/system/illama-runner.service /usr/local/bin/illama
docker rmi illama-runner:latest illama-builder:latest 2>/dev/null || true
sudo rm -rf /opt/illama   # also wipes models + cache; back up first if needed
```

---

## Table of Contents

- [Architecture](#architecture)
- [Components](#components)
- [Directory Structure](#directory-structure)
- [Suggested Models](#suggested-models)
- [Quick Start](#quick-start)
- [Commands](#commands)
- [Cache Management](#cache-management)
- [Validated Models](#validated-models)
- [Configuration](#configuration)
- [Upgrade & Backup](#upgrade--backup)
- [Remote Server Update](#remote-server-update)
- [OpenWebUI Integration](#openwebui-integration)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [OpenVINO Dependencies](#openvino-dependencies)
- [How to update runtime libraries](#how-to-update-runtime-libraries)
- [Third-party stacks](#third-party-stacks)
- [License](#license)
- **B50 notes:** [docs/b50-model-notes.md](docs/b50-model-notes.md) — GPU passthrough gotchas, benchmarks, EAGLE-3

---

## Architecture

Illama uses a **decoupled architecture** with two main components:

```
                                    ┌─────────────────────────────────────┐
                                    │         Ubuntu Host                 │
                                    │                                     │
┌──────────────┐                    │  ┌─────────────────────────────┐   │
│   User CLI   │                    │  │      illama-runner          │   │
│              │    illama serve    │  │   (FastAPI + GenAI)         │   │
│  $ illama    │───────────────────▶│  │                             │   │
│              │                    │  │  ┌─────────────────────┐    │   │
└──────────────┘                    │  │  │  OpenVINO GenAI     │    │   │
       │                            │  │  │  (LLMPipeline)      │    │   │
       │                            │  │  └─────────────────────┘    │   │
       │ illama pull                │  │            │                │   │
       │                            │  │            ▼                │   │
       ▼                            │  │  ┌─────────────────────┐    │   │
┌──────────────┐                    │  │  │   Intel GPU         │    │   │
│illama-builder│                    │  │  │   /dev/dri          │    │   │
│  (Ephemeral) │                    │  │  └─────────────────────┘    │   │
│              │                    │  └─────────────────────────────┘   │
│  optimum-cli │                    │               ▲                    │
│  export      │                    │               │                    │
│  openvino    │──────────────────▶ │  ┌────────────┴────────────┐      │
└──────────────┘   Writes models    │  │   /opt/illama/models    │      │
                                    │  │   (Shared Volume)       │      │
                                    │  └─────────────────────────┘      │
                                    └────────────────────────────────────┘
```

### Data Flow

1. **Model Pull**: `illama pull` → spawns ephemeral `illama-builder` → downloads from HuggingFace → converts to OpenVINO IR (INT4) → writes to `/opt/illama/models/`
2. **Inference**: `illama serve` → starts `illama-runner` container → loads models on-demand → exposes OpenAI-compatible API
3. **Idle Management**: Built into server → unloads model from memory after configurable idle timeout

---

## Components

### 1. illama-builder (Ephemeral Container)

**Purpose**: Converts HuggingFace models to OpenVINO IR format with INT4 quantization.

| Property | Value |
|----------|-------|
| Base Image | `ubuntu:24.04` |
| Key Packages | `optimum-intel`, `transformers`, `nncf`, `accelerate` |
| Lifecycle | Starts → Converts → Exits (not persistent) |
| Entrypoint | `optimum-cli export openvino` |

**Why isolated?** The conversion dependencies (`torch`, `transformers`, `nncf`) are heavy and conflict-prone. By isolating them, breaking changes in HuggingFace never crash your running server.

### 2. illama-runner (Inference Server)

**Purpose**: Lightweight FastAPI server using OpenVINO GenAI for LLM inference.

| Property | Value |
|----------|-------|
| Image | `illama-runner:latest` (built locally) |
| Base | `ubuntu:24.04` + `openvino-genai` + `fastapi` |
| API Port | Configurable (default: `11434`) |
| OpenAI surface | `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/unload` |
| Ollama surface | `/api/version`, `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate` |
| Health / admin | `/health`, `/illama/health`, `/illama/stats`, `/metrics` |
| Streaming | SSE for `/v1/*`, NDJSON for `/api/*` |
| Lifecycle | Persistent service (restarts on reboot) |

**Features**:
- Single model loaded at a time (memory efficient)
- Built-in idle unload (no external watchdog needed)
- GPU acceleration via Intel OpenCL/Level Zero, with persistent OpenVINO GPU model cache at `/cache`
- OpenAI-compatible API with SSE streaming, plus a parallel Ollama-compatible NDJSON surface for clients like Continue.dev, LangChain `ChatOllama`, and OpenWebUI's Ollama backend
- Optional bearer-token auth (timing-safe comparison)
- Prometheus-style `/metrics` endpoint for lightweight monitoring
- Stats persisted to disk so counters survive restarts

### 3. illama CLI (Bash Script)

**Purpose**: Orchestrates builder and runner containers, manages the model registry, and provides ergonomic local commands. See the [Commands](#commands) section below for the full table; in summary the CLI handles `serve`/`stop`/`ps`/`logs`, `pull`/`list`/`rm`/`show`, `run` (interactive or single-shot, with optional streaming), `bench`, `prune`, and `version`.

---

## Directory Structure

After installation, the following directory structure is created:

```
/opt/illama/                          # ILLAMA_HOME (configurable)
├── builder/
│   └── Dockerfile                    # Builder image definition
├── runner/
│   ├── Dockerfile                    # Runner image definition
│   └── server.py                     # FastAPI server code
├── config/
│   └── server.json                   # Runner configuration
├── models/                           # MODEL REGISTRY
│   ├── .cache/                       # HuggingFace cache (during pull)
│   ├── Qwen2.5-7B-Instruct/          # Example model
│   │   ├── config.json               # Model configuration
│   │   ├── generation_config.json    # Generation parameters
│   │   ├── openvino_model.bin        # Model weights (large file)
│   │   ├── openvino_model.xml        # Model graph definition
│   │   ├── openvino_tokenizer.bin    # Tokenizer weights
│   │   ├── openvino_tokenizer.xml    # Tokenizer graph
│   │   ├── openvino_detokenizer.bin  # Detokenizer weights
│   │   ├── openvino_detokenizer.xml  # Detokenizer graph
│   │   ├── tokenizer.json            # Original tokenizer
│   │   ├── tokenizer_config.json     # Tokenizer settings
│   │   └── special_tokens_map.json   # Special tokens
│   └── TinyLlama-1.1B-Chat-v1.0/     # Another model
│       └── ...
├── cache/                            # OpenVINO GPU model compile cache + stats snapshot
│   └── illama_stats.json             # Persisted /illama/stats counters (survive restart)
└── logs/                             # Log files (optional)
```

### Model Folder Details

Each model folder in `/opt/illama/models/` contains:

| File | Purpose | Size (typical) |
|------|---------|----------------|
| `openvino_model.bin` | Quantized model weights | 3-8 GB |
| `openvino_model.xml` | Model graph (OpenVINO IR) | ~1 MB |
| `openvino_tokenizer.*` | Tokenizer for text → tokens | ~2 MB |
| `openvino_detokenizer.*` | Detokenizer for tokens → text | ~2 MB |
| `config.json` | Model architecture config | ~2 KB |
| `tokenizer.json` | Full tokenizer vocabulary | ~2 MB |
| `model.json` | Illama metadata and optional per-model overrides | small |

---

## Suggested Models

Models tested and recommended for use with Illama on Intel GPUs:

### By VRAM Size

| VRAM | Model | HuggingFace Repo | INT4 Size |
|------|-------|------------------|-----------|
| **8 GB** | TinyLlama 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~600 MB |
| **8 GB** | Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | ~2 GB |
| **8 GB** | Qwen2.5 3B | `Qwen/Qwen2.5-3B-Instruct` | ~2 GB |
| **12 GB** | Qwen2.5 7B | `Qwen/Qwen2.5-7B-Instruct` | ~4 GB |
| **12 GB** | Gemma 3 12B | `google/gemma-3-12b-it` | ~6 GB |
| **12 GB** | MiniCPM-o 2.6 | `openbmb/MiniCPM-o-2_6` | ~6 GB |
| **16 GB** | Llama 3.2 | `meta-llama/Llama-3.2-11B-Vision-Instruct` | ~6 GB |
| **16 GB** | Phi-4 14B | `microsoft/Phi-4` | ~8 GB |
| **16 GB** | OLMo 3 7B | `allenai/OLMo-2-1124-7B-Instruct` | ~8 GB |
| **16 GB** | GPT-OSS 20B | `openai/gpt-oss-20b` | ~10 GB |
| **24 GB** | Qwen3 30B (MoE) | `Qwen/Qwen3-30B-A3B-Instruct` | ~10 GB |
| **24 GB** | Qwen3 Coder 30B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | ~10 GB |

### By Use Case

| **Use Case** | **Recommended Model** | **HuggingFace Repo** |
|----------|-------------------|------------------|
| **General Chat** | Qwen2.5 7B | `Qwen/Qwen2.5-7B-Instruct` |
| **Coding** | Qwen3 Coder 30B | `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| **Fast Testing** | TinyLlama 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Reasoning** | Phi-4 14B | `microsoft/Phi-4` |
| **Multilingual** | Qwen3 30B | `Qwen/Qwen3-30B-A3B-Instruct` |
| **Vision/Audio** | MiniCPM-o 2.6 | `openbmb/MiniCPM-o-2_6` |
| **Vision** | Llama 3.2 Vision | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| **OpenAI Compatible** | GPT-OSS 20B | `openai/gpt-oss-20b` |
| **Fully Open Source** | OLMo 3 | `allenai/OLMo-2-1124-7B-Instruct` |

### Quick Pull Commands

`illama pull` accepts either the canonical HuggingFace repo id (`org/name`)
or a full HuggingFace URL — paste-from-browser works the same as paste-from-search:

```bash
# All four of these are equivalent:
illama pull OpenVINO/Phi-4-int4-ov
illama pull https://huggingface.co/OpenVINO/Phi-4-int4-ov
illama pull https://huggingface.co/OpenVINO/Phi-4-int4-ov/tree/main
illama pull hf://OpenVINO/Phi-4-int4-ov
```

A trailing `/tree/<ref>`, `/blob/<ref>/<path>`, `/resolve/<ref>/<path>`,
query string, or `#fragment` is stripped automatically; the runner only
ever sees the canonical `org/name`.

```bash
# Small & fast (for testing)
illama pull TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Llama 3.2 (small)
illama pull meta-llama/Llama-3.2-3B-Instruct

# Best all-around (7B)
illama pull Qwen/Qwen2.5-7B-Instruct

# Google Gemma 3
illama pull google/gemma-3-12b-it

# Microsoft Phi-4 (great reasoning)
illama pull microsoft/Phi-4

# OpenAI GPT-OSS (working in OpenVINO 2026)
illama pull openai/gpt-oss-20b

# AllenAI OLMo 3 (fully open source)
illama pull allenai/OLMo-2-1124-7B-Instruct

# Qwen3 MoE (best quality, needs 24GB)
illama pull Qwen/Qwen3-30B-A3B-Instruct

# Qwen3 Coder (for coding)
illama pull Qwen/Qwen3-Coder-30B-A3B-Instruct
```

### HuggingFace Authentication

Some models (Llama, Gemma, etc.) are "gated" and require:
1. Accepting the license on HuggingFace
2. Providing your HF token

**To authenticate:**

```bash
# Get your token from https://huggingface.co/settings/tokens

# Option 1: Set for current session
export HF_TOKEN="hf_your_token_here"
illama pull meta-llama/Llama-3.2-3B-Instruct

# Option 2: Add to ~/.bashrc for permanent use
echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

---

## Quick Start

### First-Time Setup (Optional)

Before running `setup.sh`, you can configure the default port and idle timeout:

```bash
# Set custom port (default is 11434)
export ILLAMA_PORT=11435

# Override idle timeout if you don't want the default 60s
export ILLAMA_IDLE_TIMEOUT=600   # 10 minutes

# Make permanent (add to ~/.bashrc)
echo 'export ILLAMA_PORT=11435' >> ~/.bashrc
echo 'export ILLAMA_IDLE_TIMEOUT=600' >> ~/.bashrc
source ~/.bashrc
```

### 1. Install

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Pull a Model

```bash
illama pull Qwen/Qwen2.5-7B-Instruct
```

### 3. Start Server

```bash
illama serve
```

### 4. Chat

```bash
illama run Qwen2.5-7B-Instruct
```

---

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `illama serve` | Start the inference server | `illama serve` |
| `illama stop` | Stop the server | `illama stop` |
| `illama ps` | Show server status & models | `illama ps` |
| `illama pull <repo>` | Download & convert a model | `illama pull Qwen/Qwen2.5-7B-Instruct` |
| `illama search [query] [--vlm]` | Browse the HuggingFace OpenVINO model index | `illama search gemma --vlm` |
| `illama list` | List installed models | `illama list` |
| `illama rm <model>` | Delete a model | `illama rm Qwen2.5-7B-Instruct` |
| `illama show <model>` | Show model info | `illama show Qwen2.5-7B-Instruct` |
| `illama run <model>` | Interactive chat | `illama run Qwen2.5-7B-Instruct` |
| `illama run <model> --stream` | Stream output live in the terminal | `illama run phi-4-int4-ov --stream` |
| `illama logs [N]` | Tail runner logs (default 100, follows) | `illama logs 200` |
| `illama bench [args...]` | Benchmark loaded/all models via the API | `illama bench --models Qwen2.5-7B-Instruct` |
| `illama prune builder` | Remove the conversion image (~5 GB) | `illama prune builder` |
| `illama gc [--dry-run] [--force]` | Reclaim disk: prune HF cache + broken model dirs | `illama gc --dry-run` |
| `illama doctor` | Run host / runtime health checks (no changes) | `illama doctor` |
| `illama version` | Show version info | `illama version` |
| `illama --help` | Show help | `illama --help` |
| `illama -v <cmd>` | Verbose mode | `illama -v serve` |

In streamed CLI mode, Illama now requests stream usage data and prints the same token/speed summary at the end of the streamed reply.

### Discovering models

`illama search` queries the public HuggingFace API (`/api/models?filter=openvino`), prints a numbered table of trending OpenVINO repos, and offers to one-step pull whichever entry you pick.

```bash
illama search                 # 20 trending OpenVINO repos
illama search gemma           # filtered by 'gemma'
illama search gemma --vlm     # only Vision Language Models (image-text-to-text)
illama search --sort downloads --limit 30
```

Output looks like:

```text
#   NAME                                               TYPE                  DLs   LIKES  UPDATED
1   OpenVINO/gemma-4-E4B-it-int4-ov                    image-text-to-text    976       2  13 days ago
2   OpenVINO/gemma-4-E4B-it-int8-ov                    image-text-to-text    339       2  13 days ago
...

Pull which one? [1-N / Enter to skip]:
```

Pre-converted `OpenVINO/...` repos are pulled via `huggingface_hub.snapshot_download` directly. Other repos are converted with the builder image (so they go through `optimum-cli export openvino --weight-format int4`).

If you set `HF_TOKEN` in your environment, the search request will pass it as a Bearer token so gated repos show up too.

---

## Cache Management

During model download and conversion, HuggingFace caches files in `/opt/illama/models/.cache/`. This can consume significant disk space.

### Check Cache Size

```bash
# Total cache size
du -sh /opt/illama/models/.cache/

# Cache breakdown by model
du -sh /opt/illama/models/.cache/huggingface/hub/*/
```

### Clean Cache

The cache is automatically cleaned after successful conversions. To manually clean:

```bash
# Clean all cache (safe after conversions are complete)
sudo rm -rf /opt/illama/models/.cache/*

# Check disk space freed
df -h /opt/illama/models/
```

### Check for Incomplete Models

Find models that failed conversion (missing OpenVINO files):

```bash
for d in /opt/illama/models/*/; do
  name=$(basename "$d")
  [[ "$name" == ".cache" ]] && continue
  if [[ ! -f "$d/openvino_model.xml" ]] && [[ ! -f "$d/openvino_language_model.xml" ]]; then
    echo "Incomplete: $name ($(du -sh "$d" | cut -f1))"
  fi
done
```

### View Models with Sizes

```bash
# List all models with disk usage
illama list

# Detailed view
du -sh /opt/illama/models/*/ | sort -h
```

### Rebuild Containers

If containers are missing or corrupted:

```bash
cd /opt/illama

# The version comes from the VERSION file at the repo root and is also
# tagged by setup.sh so you can roll back without rebuilding from a
# different git ref:
#   docker tag illama-runner:<previous-version> illama-runner:latest
#   # e.g. after upgrading to 2.3.0:
#   #   docker tag illama-runner:2.2.0 illama-runner:latest
ILLAMA_VERSION="$(tr -d '[:space:]' < /opt/illama/VERSION 2>/dev/null || echo dev)"

# Rebuild builder (tagged :latest and :$ILLAMA_VERSION)
docker build \
  --build-arg "ILLAMA_VERSION=${ILLAMA_VERSION}" \
  -t illama-builder:latest \
  -t "illama-builder:${ILLAMA_VERSION}" \
  builder/

# Rebuild runner (tagged :latest and :$ILLAMA_VERSION)
docker build \
  --build-arg "ILLAMA_VERSION=${ILLAMA_VERSION}" \
  -t illama-runner:latest \
  -t "illama-runner:${ILLAMA_VERSION}" \
  runner/

# Verify both exist
docker images | grep illama
```

### Run the Smoke-Test Suite

The `tests/` directory contains a fast pure-Python smoke suite for the
helpers in `runner/server.py` (SSRF guard, base64 decoding, multimodal
content flattening, config parsing, version resolution, OpenAI error
shape, etc.). It runs on a vanilla Python install with `pytest` and
`fastapi` + `pydantic` only — `openvino`, `PIL`, `numpy`, and
`requests` are stubbed in `tests/conftest.py` so CI does not need the
Intel runtime.

```bash
pip install pytest fastapi pydantic
pytest tests/ -v
```

### Benchmark Installed Models

Use the included benchmark script against the running OpenAI-compatible API:

```bash
python3 scripts/benchmark_api.py \
  --base-url http://localhost:${ILLAMA_PORT:-11434}/v1 \
  --models phi-4-int4-ov Qwen3-14B-int4-ov gpt-oss-20b \
  --cold-start \
  --csv benchmarks/latest.csv \
  --markdown benchmarks/latest.md
```

If auth is enabled:

```bash
python3 scripts/benchmark_api.py \
  --base-url http://localhost:${ILLAMA_PORT:-11434}/v1 \
  --api-key "${ILLAMA_API_KEY}" \
  --cold-start
```

The script measures:

- short non-stream latency
- long non-stream latency and throughput
- stream time-to-first-token
- stream total completion time

---

## Validated Models

Models tested and confirmed working with Illama on Intel Arc GPU (15GB VRAM):

| Model | Size | Device | Notes |
|:------|-----:|:------:|:------|
| `TinyLlama-1.1B-Chat-v1.0` | 632M | GPU | ⚡ Fast, good for testing |
| `gemma-3-4b-it-int4-ov` | 3.3G | GPU | OpenVINO pre-converted |
| `Llama-3.1-8B-Instruct` | 5.2G | GPU | Reliable |
| `gemma-3-12b-it-qat-q4_0-unquantized` | 7.6G | GPU | |
| `phi-4-int4-ov` | 7.6G | GPU | OpenVINO pre-converted |
| `DeepSeek-R1-Distill-Qwen-14B-int4-ov` | 7.9G | GPU | ✨ Recommended |
| `Qwen3-14B-int4-ov` | 9.1G | GPU | OpenVINO pre-converted |
| `gpt-oss-20b` | 12G | GPU | ✨ Working in OpenVINO 2026 |
| `gemma-3-27b-it` | 16G | GPU | ⚠️ Works on Battlemage/B50-class GPU, but long replies are slow enough that non-streaming clients may appear stalled |
| `Qwen3-30B-A3B-Instruct-2507` | 16G | CPU | MoE - runs on CPU |
| `Qwen3-Coder-30B-A3B-Instruct` | 16G | CPU | MoE - runs on CPU |

> **GPU Memory Guide**: Models up to 14B are the comfortable tier on 15-16GB Intel Arc/Battlemage. Some larger models such as `gemma-3-27b-it` can still run on GPU, but they leave far less headroom and feel much slower without streaming.

> **Large-Model UX Note**: Illama now supports streaming, which helps a lot for large models. If a client disables streaming or buffers the response, models like `gemma-3-27b-it` can still take tens of seconds before the full reply is visible.

### B50 Benchmark Summary

The following practical tiers are based on live benchmarking on an Intel Battlemage B50 using Illama's OpenAI-compatible API.

**Recommended defaults**

- `Qwen3-8B-int4-ov`: best balanced default for interactive chat
- `gpt-oss-20b-int4-ov`: highest upside from the tested set; surprisingly strong throughput on B50
- `DeepSeek-R1-Distill-Qwen-14B-int4-ov`: strong reasoning-focused default
- `Qwen3-14B-int4-ov`: strong all-around 14B option

**Good but more situational**

- `phi-4-int4-ov`: solid, but not clearly ahead of the best 14B options in these tests
- `gemma-3-12b-it-qat-q4_0-unquantized`: usable, but not as compelling as DeepSeek/Qwen3-14B
- `Qwen3-30B-A3B-Instruct-2507`: good throughput after startup, but slower first token
- `Qwen3-Coder-30B-A3B-Instruct`: strong coding candidate if slower startup is acceptable

**Experimental / outlier tier**

- `gemma-3-27b-it`: works, but should be treated as an outlier model with smaller per-model token budgets
- `TinyLlama-1.1B-Chat-v1.0`: useful smoke-test model, not a primary production choice

### Recommended Per-Model Overrides

Start with these overrides for the slow outliers:

For `gemma-3-27b-it`:

```json
{
  "overrides": {
    "max_new_tokens_chat": 512,
    "max_new_tokens_completion": 512,
    "recommended_stream": true
  }
}
```

For `Qwen3-30B-A3B-Instruct-2507`:

```json
{
  "overrides": {
    "max_new_tokens_chat": 1024,
    "max_new_tokens_completion": 1024,
    "recommended_stream": true
  }
}
```

For `Qwen3-Coder-30B-A3B-Instruct`:

```json
{
  "overrides": {
    "max_new_tokens_chat": 1024,
    "max_new_tokens_completion": 1024,
    "recommended_stream": true
  }
}
```

The global defaults can remain at `2048`; only a small number of large models need tighter per-model limits.

---

## Configuration

### Environment Variables

Runtime defaults live in `config/server.json`. Environment variables override the most common deployment settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `ILLAMA_HOME` | `/opt/illama` | Base directory for all illama files |
| `ILLAMA_PORT` | `11434` | API server port (OpenAI-compatible) |
| `ILLAMA_IDLE_TIMEOUT` | `60` | Inactivity before auto-unload. Accepts seconds (`60`), seconds with unit (`60s`), or minutes (`1m`). `0` = disabled. |
| `ILLAMA_API_KEY` | unset | Optional bearer token required by `/v1/*`, `/illama/*`, and `/metrics` when set. |
| `ILLAMA_LOAD_FAILURE_POLICY` | unset | Optional override for `load_failure_policy` (`restart` or `return_error`). |

### `config/server.json`

Server-side request defaults and limits live in `config/server.json`.

Example:

```json
{
  "idle_timeout": "60s",
  "allow_stream": true,
  "max_new_tokens_chat": 2048,
  "max_new_tokens_completion": 2048,
  "max_messages_per_request": 128,
  "max_prompt_chars": 131072,
  "max_input_tokens": 0,
  "max_stop_sequences": 8,
  "max_stop_sequence_length": 256,
  "load_failure_policy": "restart"
}
```

Knobs:

| Key | Default | Description |
|-----|---------|-------------|
| `idle_timeout` | `60s` | Unload the currently loaded model after this idle period. |
| `allow_stream` | `true` | Enables `stream=true` SSE responses. Set to `false` only if you want to force non-streaming behavior. |
| `max_new_tokens_chat` | `2048` | Default `max_tokens` for `/v1/chat/completions` when the client does not supply one. |
| `max_new_tokens_completion` | `2048` | Default `max_tokens` for `/v1/completions` when the client does not supply one. |
| `max_messages_per_request` | `128` | Upper bound for the number of chat messages accepted in one request. |
| `max_prompt_chars` | `131072` | Maximum allowed prompt size in characters before tokenization. |
| `max_input_tokens` | `0` | Maximum allowed input tokens after tokenization. `0` disables the token limit. |
| `max_stop_sequences` | `8` | Maximum number of stop sequences accepted in a request. |
| `max_stop_sequence_length` | `256` | Maximum character length of any single stop sequence. |
| `load_failure_policy` | `restart` | `restart` exits the process so Docker can restart cleanly. `return_error` keeps the server alive and returns an API error when a model load fails. |

### Optional API Key Auth

Set `ILLAMA_API_KEY` only if you want bearer-token auth enabled.

```bash
export ILLAMA_API_KEY="sk-local-secret"
illama serve
```

When auth is enabled:

- OpenWebUI should use the same value in its `API Key` field
- raw clients must send `Authorization: Bearer <key>`
- `/`, `/health`, and docs remain open for basic liveness checks

Example:

```bash
curl http://localhost:${ILLAMA_PORT:-11434}/v1/models \
  -H "Authorization: Bearer ${ILLAMA_API_KEY}"
```

### Per-Model Overrides

Global defaults stay in `config/server.json`, but a model can override a small approved subset of request-budget knobs through `models/<model>/model.json`.

Resolution order is:

1. explicit request value
2. per-model override in `model.json`
3. global value from `config/server.json`
4. hardcoded fallback

Supported per-model override keys:

- `max_new_tokens_chat`
- `max_new_tokens_completion`
- `max_input_tokens`
- `max_prompt_chars`
- `recommended_stream`

Example for a large outlier model:

```json
{
  "repo_id": "google/gemma-3-27b-it",
  "source": "converted-openvino",
  "overrides": {
    "max_new_tokens_chat": 512,
    "max_new_tokens_completion": 512
  }
}
```

This is intended for a small number of exceptional models such as `gemma-3-27b-it` or `gpt-oss-20b`, while the global defaults remain at `2048`.

The effective request limits for the currently loaded model are exposed through:

- `/illama/health`
- `/illama/stats`

### Configuring API Port

```bash
# Temporary (current session)
export ILLAMA_PORT=8080
illama serve

# Permanent (add to ~/.bashrc or ~/.profile)
echo 'export ILLAMA_PORT=8080' >> ~/.bashrc
source ~/.bashrc
illama serve
```

After changing port, update OpenWebUI connection:
```
API Base URL: http://<your-ip>:8080/v1
```

### Configuring Idle Timeout (Model Unload)

The idle timeout controls how long a model can sit unused in memory before it is automatically unloaded (the server process keeps running).

```bash
# Set to 10 minutes (600 seconds)
export ILLAMA_IDLE_TIMEOUT=600
illama serve

# Disable idle unload (server keeps model loaded)
export ILLAMA_IDLE_TIMEOUT=0
illama serve

# Set to 1 minute
export ILLAMA_IDLE_TIMEOUT=60
illama serve
```

You can also use a more human-friendly format:

```bash
export ILLAMA_IDLE_TIMEOUT=5m    # 5 minutes
export ILLAMA_IDLE_TIMEOUT=30s   # 30 seconds
```

**How it works**:
1. When `illama serve` runs, the FastAPI server starts an internal idle watchdog thread.
2. The watchdog samples `last_request_time` roughly four times per `ILLAMA_IDLE_TIMEOUT` window (capped between 1s and 30s), so a 60s timeout fires within ~75s wallclock and a 15s timeout fires within ~18s wallclock.
3. If there are no new requests for `ILLAMA_IDLE_TIMEOUT` seconds, it unloads the currently loaded model from memory (VRAM/CPU RAM).
4. The container stays running; the next `illama run` or OpenAI API call automatically reloads the model on demand.

**Inspecting the unload schedule**: `GET /api/ps` returns each currently-resident model and its `expires_at` timestamp (= `last_request_time + idle_timeout`). When `idle_timeout` is `0` the field is set to `9999-12-31T23:59:59.000Z`, matching what stock Ollama emits for `keep_alive=-1` so existing dashboard clients render it as "Forever".

### Error Format

Illama now returns OpenAI-style error objects for request validation and runtime failures:

```json
{
  "error": {
    "message": "Input exceeds maximum length of 131072 characters",
    "type": "invalid_request_error",
    "param": "messages",
    "code": "input_too_large"
  }
}
```

Typical cases include:

- invalid request bodies
- input size or token-limit violations
- model lookup failures
- generation errors

### Configuring Custom Install Path

```bash
# Install to custom location
export ILLAMA_HOME=/data/illama
./setup.sh

# All subsequent commands use this path
export ILLAMA_HOME=/data/illama
illama serve
```

### Example: Production Configuration

Create `/etc/profile.d/illama.sh`:

```bash
#!/bin/bash
export ILLAMA_HOME=/opt/illama
export ILLAMA_PORT=11434
export ILLAMA_IDLE_TIMEOUT=1800  # 30 minutes
```

Then:
```bash
sudo chmod +x /etc/profile.d/illama.sh
source /etc/profile.d/illama.sh
```

---

## Upgrade & Backup

This section shows how to safely upgrade Illama to a new version **without losing your downloaded models**.

### 1. Backup existing installation (recommended)

On your server:

```bash
# Stop the running service (optional but recommended)
illama stop || true

# Create a backup directory in your home
mkdir -p ~/illama-backups

# Archive the entire /opt/illama tree (includes models)
sudo tar czf \
  ~/illama-backups/illama-$(date +%Y%m%d-%H%M).tgz \
  /opt/illama
```

This tarball contains:

- All converted models in `/opt/illama/models`
- Dockerfiles, `server.py`, CLI script, config files, logs, etc.

Optional: also back up current Docker images:

```bash
docker save illama-runner:latest \
  -o ~/illama-backups/illama-runner-$(date +%Y%m%d-%H%M).tar

docker save illama-builder:latest \
  -o ~/illama-backups/illama-builder-$(date +%Y%m%d-%H%M).tar
```

### 2. Deploy the new version (models are preserved)

On the same server, update your Illama source repo (e.g. `~/illama2`):

```bash
cd ~/illama2
git pull          # or copy the new files into this directory
```

Then re-run the setup script:

```bash
cd ~/illama2
chmod +x setup.sh
./setup.sh
```

What `setup.sh` does:

- Copies updated `builder/Dockerfile`, `runner/Dockerfile`, `runner/server.py`, and `illama` into `/opt/illama`.
- Rebuilds `illama-builder:latest` and `illama-runner:latest`.
- Refreshes the `/usr/local/bin/illama` symlink.
- **Does not delete** `/opt/illama/models`, so all previously pulled models remain on disk.

### 3. Start the updated service and verify

```bash
illama serve

# Check status
illama ps

# Verify API is up
curl http://localhost:${ILLAMA_PORT:-11434}/v1/models | jq

# Optionally check stats
curl http://localhost:${ILLAMA_PORT:-11434}/illama/health | jq
curl http://localhost:${ILLAMA_PORT:-11434}/illama/stats | jq
```

Your existing models should appear in both `illama list` and `/v1/models` without needing to re-download.

### 4. Restore from backup (if needed)

If something goes wrong and you want to roll back:

```bash
illama stop || true

# Restore /opt/illama from a previous backup
sudo rm -rf /opt/illama
sudo tar xzf ~/illama-backups/illama-YYYYMMDD-HHMM.tgz -C /

# Optionally restore old Docker images
docker load -i ~/illama-backups/illama-runner-YYYYMMDD-HHMM.tar
docker load -i ~/illama-backups/illama-builder-YYYYMMDD-HHMM.tar

# Start the restored version
illama serve
```

This guarantees that both your code and all downloaded models are restored to a known-good state.

---

## Remote Server Update

This section is the practical step-by-step update path for a remote Illama server.

### 1. Discover the current installation

Run these on the remote server before changing anything:

```bash
echo "illama binary: $(command -v illama || echo not-found)"
readlink -f "$(command -v illama 2>/dev/null)" 2>/dev/null || true
echo "ILLAMA_HOME=${ILLAMA_HOME:-<unset>}"
echo "ILLAMA_PORT=${ILLAMA_PORT:-<unset>}"
illama version || true
docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
docker images | grep -E 'illama|openvino' || true
ls -ld /opt/illama /data/illama 2>/dev/null || true
```

If `ILLAMA_HOME` is unset, the default expected install location is `/opt/illama`.

### 2. Back up the current install

```bash
illama stop || true
mkdir -p ~/illama-backups
sudo tar czf ~/illama-backups/illama-$(date +%Y%m%d-%H%M).tgz /opt/illama
docker save illama-runner:latest -o ~/illama-backups/illama-runner-$(date +%Y%m%d-%H%M).tar || true
docker save illama-builder:latest -o ~/illama-backups/illama-builder-$(date +%Y%m%d-%H%M).tar || true
```

If your live install is not in `/opt/illama`, replace that path with the discovered `ILLAMA_HOME`.

### 3. Upload or update the source tree

On the remote server, place the new source in a working directory such as `~/illama2`, then:

```bash
cd ~/illama2
git pull || true
```

If you are not using `git` on the server, copy the new project files into that working directory instead.

### 4. Review the live config before overwriting it

```bash
sudo cat /opt/illama/config/server.json 2>/dev/null || true
```

Check whether the live server has custom values for:

- `idle_timeout`
- `allow_stream`
- `max_new_tokens_chat`
- `max_new_tokens_completion`
- request limit knobs such as `max_prompt_chars` or `max_input_tokens`

If needed, save a copy first:

```bash
sudo cp /opt/illama/config/server.json ~/illama-backups/server.json.$(date +%Y%m%d-%H%M) 2>/dev/null || true
```

### 5. Run the installer/update script

```bash
cd ~/illama2
chmod +x setup.sh
./setup.sh
```

What this does:

- copies the current `builder/`, `runner/`, `config/server.json`, and `illama` into the live install directory, overwriting `/opt/illama/runner/Dockerfile` and `/opt/illama/builder/Dockerfile`
- rebuilds `illama-runner:latest` (the builder image is not rebuilt here; `illama pull` builds it on demand, or build it explicitly as in [How to update runtime libraries](#how-to-update-runtime-libraries))
- refreshes `/usr/local/bin/illama`
- keeps the existing model files under `ILLAMA_HOME/models`

Run this only when the git Dockerfiles already contain the library pins you want. A `sed` edit made directly on `/opt/illama` is replaced by whatever this checkout pins.

### 6. Verify versions and service state

```bash
illama version
docker run --rm illama-runner:latest python3 -c "import openvino, openvino_genai; print(openvino.__version__, openvino_genai.__version__)"
docker run --rm illama-builder:latest python3 -c "import transformers, optimum; print(transformers.__version__)"
illama serve
illama ps
curl http://localhost:${ILLAMA_PORT:-11434}/illama/health
curl http://localhost:${ILLAMA_PORT:-11434}/v1/models
```

### 7. Smoke test one request

```bash
illama run TinyLlama-1.1B-Chat-v1.0 -p "hello"
```

If that model is not installed, use any small model already present on the server:

```bash
illama list
```

### 8. Re-check OpenWebUI

In OpenWebUI, confirm the connection still points to:

```text
http://<server-ip>:11434/v1
```

Then refresh models and send one test prompt.

### 9. Roll back if needed

```bash
illama stop || true
sudo rm -rf /opt/illama
sudo tar xzf ~/illama-backups/illama-YYYYMMDD-HHMM.tgz -C /
docker load -i ~/illama-backups/illama-runner-YYYYMMDD-HHMM.tar || true
docker load -i ~/illama-backups/illama-builder-YYYYMMDD-HHMM.tar || true
illama serve
```

---

## API Surfaces

Illama exposes **two parallel API surfaces** against the same inference engine:

| Path prefix | Schema | Streaming | Best for |
|---|---|---|---|
| `/v1/*` | OpenAI Chat Completions | SSE (`text/event-stream`) | OpenAI clients, OpenWebUI in OpenAI mode, LangChain `ChatOpenAI`, etc. |
| `/api/*` | Ollama | NDJSON (`application/x-ndjson`) | Continue.dev, LangChain `ChatOllama`, OpenWebUI in Ollama mode, anything that hard-codes Ollama's URL shape |

### Ollama-mode quick test

```bash
curl -s http://localhost:11434/api/version
# {"version":"2.3.0"}

curl -s http://localhost:11434/api/tags | python3 -m json.tool

# Streaming chat
curl -sN http://localhost:11434/api/chat -d '{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [{"role":"user","content":"Hello"}],
  "stream": true
}'
```

The Ollama subset implemented: `GET /api/version`, `GET /api/tags`, `GET /api/ps`,
`POST /api/show`, `POST /api/chat`, `POST /api/generate`. Operations that don't
make sense in illama's "models live on disk, conversion is explicit" model
(`POST /api/pull`, `POST /api/create`, `DELETE /api/delete`) are **not**
implemented — use `illama pull` / `illama rm` for those.

### OpenAI extras

- `seed` is honored (mapped to `GenerationConfig.rng_seed`).
- `response_format={"type":"json_object"}` prepends a system instruction asking
  for JSON output (best-effort — there is no constrained sampling in OpenVINO
  GenAI yet).
- `frequency_penalty > 0` is mapped to `repetition_penalty = 1 + frequency_penalty`.
- `presence_penalty`, `n`, `logprobs` are accepted but ignored.

### Tool calling and reasoning extraction (v2.4+)

Illama parses **Hermes-3 style tool calls** and **`<think>...</think>` reasoning
blocks** out of the raw model output and surfaces them as structured fields:

| Surface | Tools | Reasoning |
|---|---|---|
| `/v1/chat/completions` | `choices[0].message.tool_calls[]` (OpenAI shape) + `finish_reason: "tool_calls"` | `choices[0].message.reasoning_content` |
| `/api/chat` | `message.tool_calls[]` (Ollama shape: no `id`/`type`/`index`, `arguments` is an object) | `message.thinking` |

You pass tools the same way you would to OpenAI / Ollama:

```bash
curl -s http://localhost:11434/v1/chat/completions -d '{
  "model": "Qwen2.5-7B-Instruct",
  "messages": [{"role":"user","content":"What is the weather in Tokyo?"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city.",
      "parameters": {"type":"object","properties":{"city":{"type":"string"}}}
    }
  }]
}'
```

The tool list is plumbed through the tokenizer's chat template
(`apply_chat_template(messages, tools=...)`), so Hermes-trained checkpoints
(Qwen 2.5/3, Hermes-2-Pro, DeepSeek-R1) see the schema in their prompt and
emit the call in the format the parser recognises. This is the same job
OpenVINO Model Server does with `--tool_parser hermes3 --reasoning_parser qwen3`
— illama just does it in Python so OVMS isn't required.

Models that don't natively emit `<tool_call>` tags will simply produce
text answers; you get the same `finish_reason: "stop"` as today and no
`tool_calls` field. Streaming on both `/v1/*` and `/api/*` emits one chunk
per tool call as soon as `</tool_call>` closes.

---

## OpenWebUI Integration

Illama speaks both **OpenAI** (`/v1/...`) and **Ollama** (`/api/...`) APIs. Point Open WebUI at whichever surface you prefer. The default port is **11434**. If that port is already taken (for example by Ollama), set `ILLAMA_PORT` in the environment or in `deploy/.env`.

On a machine with both an Intel iGPU and an Arc dGPU, set `ILLAMA_DEVICE=GPU.1` so Illama uses the discrete GPU. `GPU` alone selects `GPU.0`.

### Connection Settings

| Mode | Setting | Example |
|------|---------|---------|
| Ollama | Ollama Base URL | `http://127.0.0.1:11434` |
| OpenAI-compatible | API Base URL | `http://127.0.0.1:11434/v1` |
| API Key | — | empty, or any string if auth is disabled; match `ILLAMA_API_KEY` if set |

### Steps

1. Start illama:
   ```bash
   illama serve
   # or: cd deploy && docker compose up -d
   ```

2. In Open WebUI **Admin → Connections**, add either:
   - **Ollama:** `http://<host>:11434`, or
   - **OpenAI:** `http://<host>:11434/v1` with any API key if auth is off

3. Save. Models from `/opt/illama/models` appear in the picker.

### Verify Connection

```bash
curl -s http://127.0.0.1:11434/api/tags | python3 -m json.tool | head
curl -s http://127.0.0.1:11434/v1/models | python3 -m json.tool | head
curl -s http://127.0.0.1:11434/illama/health | python3 -c \
  'import sys,json; d=json.load(sys.stdin); print(d["device"], d["status"])'
```

> **Important**: `GET /v1` returning `404` is normal. OpenAI clients still use `http://<host>:11434/v1` because they call `/v1/models` and `/v1/chat/completions`.

If auth is enabled, put the same bearer token value into the OpenWebUI `API Key` field.

---

## Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 22.04 | Ubuntu 24.04+ |
| Docker | 20.10+ | 24.0+ |
| Python | 3.9+ | 3.11+ |
| RAM | 16 GB | 32 GB |
| VRAM (GPU) | 8 GB | 16 GB |
| Disk | 50 GB free | 100 GB+ SSD |

### Supported Hardware

- **Intel Arc GPUs**: A770, A750, A580, B580, etc.
- **Intel Integrated GPUs**: Iris Xe, UHD
- **CPU-only**: Works but slower

### GPU Driver Setup (Ubuntu)

```bash
# Add Intel graphics PPA
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt update

# Install drivers
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd

# Add user to video group
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Reboot
sudo reboot

# Verify
clinfo | head -20
```

---

## Troubleshooting

### Server won't start

```bash
# Check if port is in use
sudo lsof -i :11434

# Check Docker logs
docker logs illama-service

# Run with verbose mode
illama -v serve
```

### "Model cannot be loaded by this OpenVINO runtime" (HTTP 400, code `model_incompatible_with_runtime`)

The model files are valid but the installed OpenVINO GenAI version doesn't yet know the model's architecture. Common message: `Unsupported '<arch>' VLM model type` (or similar for any new model_type that landed in `master` after the pinned release).

Two fixes:

1. **Switch to a supported model** for now. The OpenVINO GenAI release notes track newly-added VLMs. Stable `openvino-genai==2026.4.0.0` includes the types that landed through 2026.4, including `gemma4` (`OpenVINO/gemma-4-E4B-it-int4-ov`).

2. **Bump OpenVINO GenAI** using [How to update runtime libraries](#how-to-update-runtime-libraries). `openvino`, `openvino-genai`, and `openvino-tokenizers` move together. Stable `2026.4.1` / `2026.4.1.0` is the current target and includes Gemma-4.

Unlike a GPU-state error, this error does **not** restart the container — the runner keeps serving other models. Inspect the actual reason returned in the API response or `illama logs`.

### Metrics

Illama exposes a lightweight Prometheus-style metrics endpoint:

```bash
curl http://localhost:${ILLAMA_PORT:-11434}/metrics
```

It includes:

- total requests and errors
- chat vs completion request counters
- streaming vs non-streaming request counters
- auth and validation failure counters
- active generations
- model load/unload counters
- model load time totals
- prompt/completion token totals

### Model pull fails

```bash
# Check HuggingFace authentication
huggingface-cli whoami

# Login if needed
huggingface-cli login

# Check disk space
df -h /opt/illama
```

### Large model looks hung but GPU is busy

Symptoms:

- VRAM is near full
- GPU utilization is high
- the request eventually returns, but there is no visible partial output

This is usually expected behavior for slow generation, not a crash. Streaming is now supported, but if a client disables streaming or buffers the response you may still see long waits before output appears.

For large models such as `gemma-3-27b-it` on Battlemage/B50-class GPUs:

- small replies may return in under a second
- medium/long replies can take 30-60s+ before the full response finishes
- around `8-9 tokens/sec` is a realistic observed range for long `gemma-3-27b-it` replies on this setup

Quick sanity check:

```bash
curl -s http://localhost:${ILLAMA_PORT:-11434}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-27b-it",
    "messages": [{"role":"user","content":"Reply with exactly: hello"}],
    "max_tokens": 16,
    "stream": false
  }'
```

If short prompts succeed but long prompts feel stalled, prefer:

- smaller explicit `max_tokens`
- smaller/faster models such as `gemma-3-12b-it-qat-q4_0-unquantized`, `Qwen3-14B-int4-ov`, or `DeepSeek-R1-Distill-Qwen-14B-int4-ov`
- using streaming-capable clients for very large models so users can see progress immediately

### Model load failed

If a model fails during load, behavior is controlled by `load_failure_policy`:

- `restart`: default, exits the process so Docker can restart cleanly
- `return_error`: keeps the server alive and returns an API error

Example:

```json
{
  "load_failure_policy": "return_error"
}
```

For quick testing on a live server, you can override the config with an environment variable:

```bash
export ILLAMA_LOAD_FAILURE_POLICY=return_error
illama stop
illama serve
```

### GPU not detected

```bash
# Check device exists
ls -la /dev/dri

# Check OpenCL
clinfo

# Verify user is in video group
groups | grep video
```

---

## OpenVINO Dependencies

Libraries are baked into the images at `docker build`. Restarting `illama-service` does not change them. The next upgrade starts by reading the `pip install` lines in the Dockerfiles that Docker actually builds, then replacing those exact version strings.

### Latest stable (checked 2026-10-04)

This is the target until a newer release is confirmed on PyPI. Update this table when that happens, then follow the procedure below.

| Package | Latest stable | Where it is pinned |
|---|---|---|
| `openvino` | `2026.4.1` | runner and builder `pip install` (patch on the 2026-09-16 2026.4 line) |
| `openvino-genai` | `2026.4.1.0` | same line as `openvino`; move them together |
| `openvino-tokenizers` | `2026.4.1.0` | pulled in by `openvino-genai` when the Dockerfile has no explicit pin |
| `optimum-intel` | `2.2.0` | builder only; `transformers` must stay `>=4.51,<5.6` |
| `transformers` | `4.57.6` | builder only; inside optimum-intel's `>=4.51,<5.6` window |
| `fastapi` / `uvicorn` / `pydantic` | `0.136.1` / `0.46.0` / `2.13.3` | runner HTTP stack; independent of the OpenVINO bump |

`openvino`, `openvino-genai`, and `openvino-tokenizers` are one build. Mixing release families breaks the binary ABI.

Release notes: [OpenVINO 2026.4](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html), [optimum-intel 2.2.0](https://huggingface.co/blog/echarlaix/optimum-intel-v22).

`runner/Dockerfile` and `builder/Dockerfile` pin those stable releases directly. `./setup.sh` copies them onto `/opt/illama` and rebuilds. Checked against PyPI on 2026-10-13: OpenVINO 2026.4.1 is still the latest stable. Do not install a newer `transformers` than optimum-intel allows (`<5.6`).

### How to update runtime libraries

Run this on the host that runs Illama. Read the files before editing. Replace the version strings you actually see; the `sed` below is an example (`2026.4.0` → `2026.4.1`).

**1. See what is running, and the lines you will replace**

```bash
docker ps --filter name=illama-service --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'

docker exec illama-service python3 -c "import importlib.metadata as m
for p in ['openvino','openvino-genai','openvino-tokenizers','optimum-intel','fastapi','uvicorn','pydantic']:
    try:
        print(f'{p}=={m.version(p)}')
    except m.PackageNotFoundError:
        print(f'{p} not installed')"

grep -nE 'pip install|openvino|optimum-intel|extra-index-url|--pre' \
  /opt/illama/runner/Dockerfile /opt/illama/builder/Dockerfile
```

Write down the host port from `docker ps`. `illama serve` publishes `11434` unless `ILLAMA_PORT` is set.

**2. Replace those exact pins**

`openvino-genai` is replaced before `openvino` so the shorter name cannot match the longer one. Adjust the old version strings to match the `grep` output.

```bash
sed -i \
  -e 's/"openvino-genai==2026.4.0.0"/"openvino-genai==2026.4.1.0"/' \
  -e 's/"openvino==2026.4.0"/"openvino==2026.4.1"/' \
  /opt/illama/runner/Dockerfile /opt/illama/builder/Dockerfile
```

If a runner line still has the nightly index, drop that too:

```bash
sed -i \
  -e '/^RUN pip install/s/ --pre//' \
  -e '/wheels\/nightly/d' \
  /opt/illama/runner/Dockerfile
```

Confirm before building:

```bash
grep -nE 'openvino|optimum-intel' /opt/illama/runner/Dockerfile /opt/illama/builder/Dockerfile
```

**3. Rebuild and start on the same host port**

```bash
VER="$(tr -d '[:space:]' < /opt/illama/VERSION 2>/dev/null || echo dev)"

docker build --build-arg "ILLAMA_VERSION=${VER}" \
  -t illama-runner:latest -t "illama-runner:${VER}" \
  /opt/illama/runner

docker build --build-arg "ILLAMA_VERSION=${VER}" \
  -t illama-builder:latest -t "illama-builder:${VER}" \
  /opt/illama/builder

export ILLAMA_PORT="${ILLAMA_PORT:-11434}"
illama stop || true
illama serve

docker exec illama-service python3 -c "import importlib.metadata as m
for p in ['openvino','openvino-genai','openvino-tokenizers']:
    print(f'{p}=={m.version(p)}')"
```

Expected after the 2026-10-04 bump: `openvino==2026.4.1`, `openvino-genai==2026.4.1.0`, `openvino-tokenizers==2026.4.1.0`. Models in `/opt/illama/models` stay. The first load of each model recompiles into `/opt/illama/cache`.

The runner build changes the live server. The builder build is the large one (`optimum-intel` pulls PyTorch) and matters on the next `illama pull` that still needs conversion.

When a newer git tree is supposed to carry these pins, update the Dockerfiles in the checkout first, then `./setup.sh` so it copies them onto `/opt/illama` and rebuilds the runner. Build the builder image with the `docker build` above if conversions need the new `optimum-intel` as well. Keep `ILLAMA_PORT` at the port `docker ps` showed.

## Vision models (VLM) usage

VLM models such as `OpenVINO/gemma-4-E4B-it-int4-ov` (Image-Text-to-Text) are loaded automatically when the model directory contains `openvino_language_model.xml` instead of `openvino_model.xml`. Both the OpenAI and the Ollama API surfaces accept image inputs.

### OpenAI-compatible (`/v1/chat/completions`)

Pass content as the multimodal content-array form. Both `data:` URLs (base64) and `https://` URLs are accepted. Decoding budget per image is 20 MB; HTTPS fetch timeout is 10 seconds.

```bash
# Base64 data URL
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-int4-ov",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
      ]
    }],
    "max_tokens": 256,
    "temperature": 0.2
  }'

# Direct HTTPS URL
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-int4-ov",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this picture."},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}
      ]
    }]
  }'
```

### Ollama-compatible (`/api/chat`, `/api/generate`)

Use the `images` field with raw base64 strings (no `data:` prefix).

```bash
curl -s http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-int4-ov",
    "messages": [{
      "role": "user",
      "content": "What do you see?",
      "images": ["iVBORw0KGgo..."]
    }],
    "stream": false
  }'
```

`/api/generate` accepts `images` on the request body itself (single-prompt form).

`temperature`, `top_p`, `max_tokens` (and Ollama's `num_predict`) are honored on VLM models the same way as on plain LLM models.

## Known Limitations

These are tracked items where the current implementation is intentionally narrow or has known sharp edges. None of them block typical OpenWebUI / Ollama-client / OpenAI-client usage but they are worth knowing about when planning deployments.

- **`response_format` is best-effort.** OpenVINO GenAI does not yet expose constrained-grammar sampling. `response_format={"type":"json_object"}` is implemented as a prepended system instruction, so model adherence depends on the model. `response_format={"type":"json_schema"}` is not implemented.
- **`presence_penalty` and `n` are accepted but ignored.** `frequency_penalty > 0` is mapped onto OpenVINO's `repetition_penalty` as `1 + frequency_penalty`, which is a best-effort translation, not a one-to-one.
- **Single in-flight model.** This is by design - one model loaded at a time, sequential generation. There is no batching across concurrent requests.
- **Image fetch is unauthenticated.** When an OpenAI `image_url` points at an HTTPS URL, the runner does a plain `requests.get` with a 10s timeout and 20 MB cap. There is no allowlist; if you need to restrict outbound access, run the runner container in a network namespace that scopes egress.

## Backend Alternatives (September 2026)

Illama is built on OpenVINO GenAI. Alternative inference backends for Intel Arc / Battlemage exist but each has trade-offs:

| Backend | Status (September 2026) | When to consider |
|---------|-------------------|-------------------|
| **OpenVINO GenAI 2026.4** | Current stable. First-class Battlemage support, Gemma-4 VLMs, wider model coverage than 2026.1. | Default. Bump through [How to update runtime libraries](#how-to-update-runtime-libraries). |
| **`llama.cpp` (SYCL or Vulkan)** | Active. Major Battlemage optimizations through 2026 (e.g. `ggml-org/llama.cpp` PRs #21527/#21638/#22066). Q4_K_M ~20 t/s on B70 for 27B models is realistic. | Future migration target if specific quantizations or ggml-format models matter. Not a drop-in - the runner would need a rewrite. |
| **OpenVINO + `llama.cpp` backend** | New in 2026.1: OpenVINO ships a `llama.cpp` backend that targets Intel GPUs. Lets you run ggml/gguf models through OpenVINO's runtime. | Worth evaluating once it stabilises; could give us ggml models without leaving the OpenVINO ecosystem. |
| **IPEX-LLM** | **Archived by Intel on 2026-01-28.** No further updates. | **Do not migrate to IPEX-LLM.** Anything you build on it will be unsupported within the year. |
| **vLLM** | NVIDIA-first. Intel paths are CPU-only or via Habana/Gaudi. | Not relevant for Intel Arc / Battlemage GPU inference. |

The pragmatic recommendation for this hardware is: stay on OpenVINO GenAI, currently stable **2026.4.1**, and revisit another backend only when OpenVINO shows a specific perf or model-coverage gap.

## Next Priority

Stability and CLI ergonomics ahead of feature breadth:

- harden SSE / NDJSON streaming against client disconnect (drain-on-cancel for the worker thread)
- enforce `allow_stream` on the Ollama `/api/*` surface so the global toggle actually applies to all clients
- bound `/metrics` per-model cardinality
- harden `illama pull` against partial-download poisoning and `illama rm` against path-traversal of the model name

### Related Resources

- **[openvino_build_deploy](https://github.com/openvinotoolkit/openvino_build_deploy)**: Pre-built samples and reference kits for OpenVINO applications. Contains:
  - AI Reference Kits (production-ready examples)
  - Interactive demos
  - Training notebooks for GenAI and LLMs
  - Useful for learning but not directly applicable to Illama's architecture

---

## Third-party stacks

Illama is Apache-2.0. It runs on top of these projects. Their licenses stay with them; this table is attribution, not a relicense.

| Stack | Role here | License |
|---|---|---|
| [OpenVINO](https://github.com/openvinotoolkit/openvino), [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai), OpenVINO Tokenizers | Inference runtime in the runner image | Apache-2.0 |
| [Intel Graphics Compute Runtime](https://github.com/intel/compute-runtime) / Level Zero | GPU driver userspace inside the runner image | MIT |
| [optimum-intel](https://github.com/huggingface/optimum-intel) | Model export (`optimum-cli export openvino`) in the builder image | Apache-2.0 |
| [Transformers](https://github.com/huggingface/transformers), [Accelerate](https://github.com/huggingface/accelerate) | Export-time model code in the builder image | Apache-2.0 |
| [NNCF](https://github.com/openvinotoolkit/nncf) | Pulled in by `optimum-intel[openvino]` for INT4 compression | Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | HTTP API | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | ASGI server | BSD-3-Clause |
| [Pydantic](https://github.com/pydantic/pydantic) | Request and response models | MIT |
| [Ubuntu](https://ubuntu.com/) 24.04 | Base image | Various; see Canonical's terms |

Model weights you `illama pull` keep the license of that checkpoint (some require accepting terms on Hugging Face). Illama does not grant rights to those weights.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
