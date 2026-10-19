#!/bin/bash
# ============================================================================
# Illama V2 - Setup Script
# Run this on your Ubuntu 25.10 host to install illama
# ============================================================================

set -e

ILLAMA_HOME="/opt/illama"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Tracks whether we found a customised live config and skipped overwriting
# it. Surfaced at the end of the script so operators see the next-step hint.
PRESERVED_LIVE_CONFIG=0

# Read the canonical project version from VERSION at repo root. Falls back
# to "0.0.0-dev" if VERSION is missing so the build still proceeds, but
# operators see an obviously-wrong tag and can investigate.
if [[ -f "${SCRIPT_DIR}/VERSION" ]]; then
    ILLAMA_VERSION="$(tr -d '[:space:]' < "${SCRIPT_DIR}/VERSION")"
fi
ILLAMA_VERSION="${ILLAMA_VERSION:-0.0.0-dev}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}>>>${NC} $1"; }
success() { echo -e "${GREEN}✅${NC} $1"; }
warn() { echo -e "${YELLOW}⚠️${NC} $1"; }
error() { echo -e "${RED}❌${NC} $1"; exit 1; }

echo ""
echo "============================================"
echo "  Illama V2 Setup - OpenVINO LLM Manager"
echo "============================================"
echo ""

# Check prerequisites
log "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker first:\n  curl -fsSL https://get.docker.com | sh"
fi

if ! docker info &> /dev/null; then
    error "Docker daemon is not running or you don't have permission. Try:\n  sudo usermod -aG docker \$USER && newgrp docker"
fi

if ! command -v python3 &> /dev/null; then
    error "Python3 is not installed. Please install it:\n  sudo apt install python3"
fi

success "Prerequisites OK"

# Create directory structure
log "Creating directory structure at $ILLAMA_HOME..."

sudo mkdir -p "$ILLAMA_HOME"/{builder,runner,scripts,config,models,cache,logs,deploy}
sudo chown -R "$USER:$USER" "$ILLAMA_HOME"

# Copy files
log "Installing files..."

cp "$SCRIPT_DIR/builder/Dockerfile" "$ILLAMA_HOME/builder/"
cp "$SCRIPT_DIR/builder/.dockerignore" "$ILLAMA_HOME/builder/" 2>/dev/null || true
cp "$SCRIPT_DIR/runner/Dockerfile" "$ILLAMA_HOME/runner/"
cp "$SCRIPT_DIR/runner/.dockerignore" "$ILLAMA_HOME/runner/" 2>/dev/null || true
# server.py imports its sibling modules (_helpers.py, _schemas.py) at module
# load, so we copy every Python file in the runner/ directory rather than
# just server.py. The Dockerfile uses `COPY *.py /app/` to pick them up.
cp "$SCRIPT_DIR/runner/"*.py "$ILLAMA_HOME/runner/"
cp "$SCRIPT_DIR/illama" "$ILLAMA_HOME/"
cp "$SCRIPT_DIR/VERSION" "$ILLAMA_HOME/" 2>/dev/null || true
cp "$SCRIPT_DIR/scripts/"*.py "$ILLAMA_HOME/scripts/" 2>/dev/null || true

# Server config: preserve operator customisations on upgrade. If a live
# config already exists, drop the new one alongside as `.new` so the
# operator can diff and merge intentionally. Fresh installs get the
# shipped defaults outright.
if [[ -f "$SCRIPT_DIR/config/server.json" ]]; then
    if [[ -f "$ILLAMA_HOME/config/server.json" ]]; then
        if ! cmp -s "$SCRIPT_DIR/config/server.json" "$ILLAMA_HOME/config/server.json"; then
            cp "$SCRIPT_DIR/config/server.json" "$ILLAMA_HOME/config/server.json.new"
            PRESERVED_LIVE_CONFIG=1
            warn "Live config differs from shipped defaults - kept yours."
            echo "    Live:    $ILLAMA_HOME/config/server.json"
            echo "    Shipped: $ILLAMA_HOME/config/server.json.new"
            echo "    Compare: diff -u $ILLAMA_HOME/config/server.json $ILLAMA_HOME/config/server.json.new"
        fi
    else
        cp "$SCRIPT_DIR/config/server.json" "$ILLAMA_HOME/config/"
    fi
fi

# Deploy artefacts (docker-compose.yml, systemd unit, .env.example) are
# bundled in the install dir but never overwrite an operator's .env.
if [[ -d "$SCRIPT_DIR/deploy" ]]; then
    cp "$SCRIPT_DIR/deploy/docker-compose.yml" "$ILLAMA_HOME/deploy/" 2>/dev/null || true
    cp "$SCRIPT_DIR/deploy/illama-runner.service" "$ILLAMA_HOME/deploy/" 2>/dev/null || true
    cp "$SCRIPT_DIR/deploy/.env.example" "$ILLAMA_HOME/deploy/" 2>/dev/null || true
    if [[ ! -f "$ILLAMA_HOME/deploy/.env" && -f "$ILLAMA_HOME/deploy/.env.example" ]]; then
        cp "$ILLAMA_HOME/deploy/.env.example" "$ILLAMA_HOME/deploy/.env"
    fi
fi

chmod +x "$ILLAMA_HOME/illama"

# Fix Windows line endings (CRLF -> LF) if transferred from Windows
log "Converting line endings..."
sed -i 's/\r$//' "$ILLAMA_HOME/illama"
sed -i 's/\r$//' "$ILLAMA_HOME/builder/Dockerfile"
sed -i 's/\r$//' "$ILLAMA_HOME/runner/Dockerfile"
for f in "$ILLAMA_HOME/runner/"*.py; do
    [[ -f "$f" ]] && sed -i 's/\r$//' "$f"
done
for f in "$ILLAMA_HOME/scripts/"*.py; do
    [[ -f "$f" ]] && sed -i 's/\r$//' "$f"
done

# Create symlink
log "Creating symlink in /usr/local/bin..."
sudo ln -sf "$ILLAMA_HOME/illama" /usr/local/bin/illama

success "CLI installed to /usr/local/bin/illama"

# Build runner image (the only persistent service).
# The conversion-only `illama-builder` image is built lazily by `illama pull`
# the first time you fetch a non-pre-converted model (~5 GB), so it's never
# built on hosts that only ever consume OpenVINO/* HuggingFace repos.
#
# Tag with both `:latest` (rolling pointer used by `illama serve`) and
# `:$ILLAMA_VERSION` (immutable rollback target). After an upgrade, an operator
# can pin back with: docker tag illama-runner:<previous-version> illama-runner:latest
log "Building illama-runner Docker image (version: ${ILLAMA_VERSION})..."
docker build \
    --build-arg "ILLAMA_VERSION=${ILLAMA_VERSION}" \
    -t illama-runner:latest \
    -t "illama-runner:${ILLAMA_VERSION}" \
    "$ILLAMA_HOME/runner"

success "Runner image created (illama-runner:latest, illama-runner:${ILLAMA_VERSION})"

# Verify GPU access
log "Checking GPU access..."
if [[ -e /dev/dri ]]; then
    success "GPU device found at /dev/dri"
    
    # Check if user is in video and render groups
    if ! groups | grep -q video; then
        warn "You may need to add yourself to the video group:"
        echo "  sudo usermod -aG video \$USER"
    fi
    if ! groups | grep -q render; then
        warn "You may need to add yourself to the render group:"
        echo "  sudo usermod -aG render \$USER"
    fi
    if ! groups | grep -qE "(video|render)"; then
        echo "  (Log out and back in for changes to take effect)"
    fi
else
    warn "No GPU found at /dev/dri. Will use CPU inference."
fi

# Report the port the operator will actually reach the API on. deploy/.env is
# the source of truth once compose is in use; fall back to the environment and
# finally to the CLI's own default. Printing a hardcoded 11434 here sent people
# to the wrong port whenever ILLAMA_PORT had been customised (and 11434 is also
# Ollama's default, so the two commonly coexist on different ports).
EFFECTIVE_PORT="${ILLAMA_PORT:-}"
if [[ -z "$EFFECTIVE_PORT" && -f "$ILLAMA_HOME/deploy/.env" ]]; then
    EFFECTIVE_PORT="$(grep -E '^[[:space:]]*ILLAMA_PORT=' "$ILLAMA_HOME/deploy/.env" \
        | tail -1 | cut -d= -f2- | tr -d '[:space:]')"
fi
EFFECTIVE_PORT="${EFFECTIVE_PORT:-11434}"

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-localhost}"

# NOTE: every line below that interpolates a colour variable needs `echo -e`.
# The colour codes are stored as literal backslash escapes ('\033[1;33m'), so a
# bare `echo` prints them verbatim instead of colouring the text.
echo ""
echo "============================================"
echo -e "  ${GREEN}Installation Complete!${NC}"
echo "============================================"
echo ""
echo "Quick Start:"
echo ""
echo "  1. Pull a model:"
echo -e "     ${YELLOW}illama pull OpenVINO/Phi-4-int4-ov${NC}"
echo ""
echo "     Or pull a model together with a speculative-decoding draft model,"
echo "     which is roughly 2.5x faster on text-only models:"
echo -e "     ${YELLOW}illama pull Qwen/Qwen3-8B --draft AngelSlim/Qwen3-8B_eagle3${NC}"
echo ""
echo "  2. Start the server (pick one):"
echo -e "     ${YELLOW}illama serve${NC}                                # bash CLI"
echo -e "     ${YELLOW}cd $ILLAMA_HOME/deploy && docker compose up -d${NC}  # compose"
echo ""
echo "  3. Chat with the model:"
echo -e "     ${YELLOW}illama run Phi-4-int4-ov${NC}"
echo ""
echo "For OpenWebUI, use:"
echo "  API Base URL: http://${HOST_IP}:${EFFECTIVE_PORT}/v1"
echo ""
if [[ $PRESERVED_LIVE_CONFIG -eq 1 ]]; then
    echo -e "${YELLOW}⚠️  Review the new shipped config and merge any intended changes:${NC}"
    echo "   diff -u $ILLAMA_HOME/config/server.json $ILLAMA_HOME/config/server.json.new"
    echo ""
fi
echo "Run 'illama --help' for all commands."
echo ""
