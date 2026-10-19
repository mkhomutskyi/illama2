"""
Illama Server - OpenVINO GenAI FastAPI Server
OpenAI-compatible API for LLM inference
"""

import os
import json
import secrets
import time
import asyncio
import io
import ipaddress
import queue
import socket
import threading
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from PIL import Image as PILImage
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import openvino as ov
import openvino_genai as ov_genai

# Pure helpers and Pydantic schemas live in sibling modules. We re-import
# every public name here so existing callers (and the test suite) can keep
# using ``server.Message``, ``server._b64decode_lenient`` etc. unchanged.
from _helpers import (  # noqa: F401  (re-exported for backwards compat)
    _AUTH_EXEMPT_PATHS,
    _DATA_URL_RE,
    _b64decode_lenient,
    _get_config_bool,
    _get_config_int,
    _parse_idle_timeout,
    build_chat_stream_chunk,
    build_completion_stream_chunk,
    build_usage_payload,
    is_auth_exempt_path,
    make_openai_error,
    make_stream_error_event,
    normalize_load_failure_policy,
    sse_encode,
)
from _schemas import (  # noqa: F401
    ChatCompletionRequest,
    CompletionRequest,
    ContentPart,
    ContentPartImage,
    ContentPartText,
    ImageURL,
    Message,
    ModelInfo,
    ModelsResponse,
    StreamOptions,
)
from _parsers import (  # noqa: F401
    StreamingParser,
    parse_full_output,
    to_ollama_tool_call,
)


# How long to wait after a fatal GPU load failure before forcing the process
# to exit. Gives the in-flight HTTP response a chance to flush to the client
# (with a meaningful 503 + error body) before Docker reaps the container and
# restarts it with a fresh GPU state. Tunable via env for ops who want a
# faster restart loop in dev.
_FATAL_EXIT_GRACE_SECONDS = float(
    (os.environ.get("ILLAMA_FATAL_EXIT_GRACE_SECONDS") or "1.5").strip() or 1.5
)


def _schedule_fatal_exit(reason: str) -> None:
    """Exit the process after a short grace period in a daemon thread.

    Used when load_model hits an unrecoverable GPU error and policy =
    `restart`. The caller still raises an HTTPException so the current
    request gets a clean 503; this function ensures Docker's
    `--restart unless-stopped` reaps and restarts the container instead of
    leaving it in a wedged state.
    """
    def _exit_after_grace() -> None:
        time.sleep(_FATAL_EXIT_GRACE_SECONDS)
        # Drop straight to os._exit to bypass any uvicorn shutdown handlers
        # that might also try to touch the broken GPU pipeline.
        os._exit(1)

    print(
        f"[FATAL] {reason} - scheduling process exit in "
        f"{_FATAL_EXIT_GRACE_SECONDS:.1f}s so the current response can drain"
    )
    threading.Thread(target=_exit_after_grace, daemon=True, name="illama-fatal-exit").start()


# Version is sourced (in priority order) from:
#   1. ILLAMA_VERSION env var (set by runner/Dockerfile from --build-arg)
#   2. /app/VERSION file (copied alongside server.py if the build context provides one)
#   3. The hardcoded fallback below (kept current per release as a last-resort safety net)
_VERSION_FALLBACK = "2.6.0"


def _read_illama_version() -> str:
    env_value = (os.environ.get("ILLAMA_VERSION") or "").strip()
    if env_value:
        return env_value
    version_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")
    try:
        if os.path.exists(version_file):
            with open(version_file, "r", encoding="utf-8") as f:
                value = f.read().strip()
                if value:
                    return value
    except Exception as exc:
        print(f"[WARN] Failed to read VERSION file at {version_file}: {exc}")
    return _VERSION_FALLBACK


ILLAMA_VERSION: str = _read_illama_version()


def _load_server_config() -> Dict[str, Any]:
    """
    Load optional server configuration from JSON.

    The config path can be overridden via ILLAMA_CONFIG_PATH, otherwise
    defaults to /config/server.json inside the container.
    """
    config_path = os.environ.get("ILLAMA_CONFIG_PATH", "/config/server.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    print(f"[INFO] Loaded config from {config_path}")
                    return data
    except Exception as exc:
        print(f"[WARN] Failed to load config {config_path}: {exc}")
    return {}


# Config helpers (_parse_idle_timeout, _get_config_int, _get_config_bool,
# normalize_load_failure_policy) live in _helpers.py and are re-imported at
# the top of this file. They're pure functions of their args, so unit tests
# import them directly without spinning up the full server module.


# Configuration
CONFIG: Dict[str, Any] = _load_server_config()
MODELS_DIR = os.environ.get("ILLAMA_MODELS", "/opt/illama/models")
IDLE_TIMEOUT = _parse_idle_timeout(CONFIG, default_seconds=60)
DEFAULT_DEVICE = os.environ.get("ILLAMA_DEVICE", "GPU")
CACHE_DIR = os.environ.get("ILLAMA_CACHE_DIR", "").strip()
STATS_PATH = os.environ.get(
    "ILLAMA_STATS_PATH",
    os.path.join(CACHE_DIR, "illama_stats.json") if CACHE_DIR else "",
).strip()
MAX_NEW_TOKENS_CHAT = int(CONFIG.get("max_new_tokens_chat", 2048))
MAX_NEW_TOKENS_COMPLETION = int(CONFIG.get("max_new_tokens_completion", 2048))
FALLBACK_MAX_NEW_TOKENS = 2048
ALLOW_STREAM = _get_config_bool(CONFIG, "allow_stream", True)
MAX_MESSAGES_PER_REQUEST = _get_config_int(CONFIG, "max_messages_per_request", 128)
MAX_PROMPT_CHARS = _get_config_int(CONFIG, "max_prompt_chars", 131072)
# Default of 8192 input tokens (was 0/disabled). Picks a conservative value that
# fits inside every chat model we ship (Qwen, Llama, Phi, GPT-OSS, Gemma all
# support >=8K context) while still preventing an unauthenticated client from
# sending a 200K-character prompt and pinning the GPU. Operators with larger
# contexts should override via config/server.json (`"max_input_tokens": 32768`)
# or per-model in models/<name>/model.json. Setting 0 in config disables the
# check, which preserves the pre-2026-05 behaviour for callers who rely on it.
MAX_INPUT_TOKENS = _get_config_int(CONFIG, "max_input_tokens", 8192)
MAX_STOP_SEQUENCES = _get_config_int(CONFIG, "max_stop_sequences", 8)
MAX_STOP_SEQUENCE_LENGTH = _get_config_int(CONFIG, "max_stop_sequence_length", 256)
LOAD_FAILURE_POLICY = normalize_load_failure_policy(
    str(os.environ.get("ILLAMA_LOAD_FAILURE_POLICY") or CONFIG.get("load_failure_policy", "restart"))
)
API_KEY = str(os.environ.get("ILLAMA_API_KEY") or CONFIG.get("api_key") or "").strip()
AUTH_ENABLED = bool(API_KEY)

# ---------------------------------------------------------------------------
# Pipeline backend selection (2.5.0)
# ---------------------------------------------------------------------------
#
# openvino-genai's LLMPipeline has two very different backends:
#
#   stateful   - the default when no scheduler_config is supplied. Uses the
#                SDPA attention path. Simplest, lowest memory, but does NOT
#                support speculative decoding or prompt-lookup, and Intel has
#                been moving models off it for performance (openvino.genai
#                PR #4244 switched Qwen3.5/Qwen3.6 from SDPA to PagedAttention
#                "for performance reasons", 2026-08-13).
#
#   continuous - selected by passing {"scheduler_config": SchedulerConfig()}
#                in the config mapping. Uses PagedAttention, and is the ONLY
#                backend that supports draft-model speculative decoding
#                (incl. EAGLE-3 and DFlash - openvino.genai PR #3938) and
#                prefix caching.
#
# "auto" tries continuous and transparently falls back to stateful if the
# runtime rejects it, so a bad SchedulerConfig can never take the server down.
_PIPELINE_BACKEND_VALUES = {"auto", "stateful", "continuous"}


def normalize_pipeline_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend in _PIPELINE_BACKEND_VALUES:
        return backend
    print(f"[WARN] Invalid pipeline_backend '{value}', defaulting to auto")
    return "auto"


PIPELINE_BACKEND = normalize_pipeline_backend(
    os.environ.get("ILLAMA_PIPELINE_BACKEND") or CONFIG.get("pipeline_backend", "auto")
)
# KV-cache budget for the continuous-batching backend, in GB. Too small and
# long contexts thrash; too large and it competes with the model weights for
# VRAM. 2 GB is the upstream sample default and leaves room for a 9B INT4
# model plus headroom on a 16 GB card.
SCHEDULER_CACHE_SIZE_GB = _get_config_int(CONFIG, "scheduler_cache_size_gb", 2)
# Upper bound for the cache-size escalation ladder. Models with linear
# attention (Qwen3.5's Gated DeltaNet, for example) need a much larger paged
# cache than a plain KV model - Qwen3.5-9B rejects 2/4/6 GB and only
# constructs at 8 GB. Rather than make every operator discover that by hand,
# we retry with progressively larger caches up to this ceiling. Keep it below
# (VRAM - model size) or the escalation just OOMs at the top of the ladder.
SCHEDULER_CACHE_MAX_GB = _get_config_int(CONFIG, "scheduler_cache_max_gb", 8)
# Reuse KV blocks across requests that share a prompt prefix. Big win for
# multi-turn chat (each turn resends the whole conversation) and for agentic
# loops with a fixed system prompt.
ENABLE_PREFIX_CACHING = _get_config_bool(CONFIG, "enable_prefix_caching", True)
# Prompt-lookup decoding: drafts candidate tokens by n-gram matching against
# the prompt itself - no draft model, no extra VRAM. Strong on code editing,
# summarisation, and agentic turns where output echoes input; neutral-to-
# slightly-negative on open-ended prose. Off by default, opt in per model via
# model.json overrides.
PROMPT_LOOKUP_DEFAULT = _get_config_bool(CONFIG, "prompt_lookup", False)

MODEL_OVERRIDE_KEYS = {
    "max_new_tokens_chat",
    "max_new_tokens_completion",
    "max_input_tokens",
    "max_prompt_chars",
    "recommended_stream",
    # 2.5.0 - per-model pipeline tuning
    "prompt_lookup",
    "draft_model",
    "num_assistant_tokens",
    "scheduler_cache_size_gb",
}

# Global state
current_model: Optional[ov_genai.LLMPipeline] = None
current_model_name: Optional[str] = None
last_request_time: float = time.time()
model_lock = threading.Lock()
active_generations: int = 0

# Inference statistics
#
# stats_lock guards all per_model_stats mutations and the totals counters
# below. It is intentionally NOT held while doing disk I/O (e.g. reading
# model.json via get_effective_request_limits) - call _snapshot_inference_stats
# to copy out the values cheaply, then do any I/O outside the lock.
#
# /metrics caps the per-model rows it emits so an operator who churns through
# many model names doesn't grow Prometheus cardinality unboundedly. Records
# are kept in per_model_stats for /illama/stats inspection but only the
# top-N most-recently-active are emitted to /metrics.
MAX_METRICS_PER_MODEL_LABELS = 50
stats_lock = threading.Lock()
server_start_time: float = time.time()
total_requests: int = 0
total_errors: int = 0
total_prompt_tokens: int = 0
total_completion_tokens: int = 0
total_generation_time: float = 0.0
# Streaming-only metric: cumulative time-to-first-token and the count of
# streamed requests that contributed to it. Splitting TTFT from total
# generation time lets ops dashboards show prompt-processing latency
# separately from per-token decode speed (tokens-per-second after the first
# token), which is the more useful metric when comparing model sizes.
total_first_token_time: float = 0.0
total_first_token_samples: int = 0
per_model_stats: Dict[str, Dict[str, Any]] = {}
total_model_loads: int = 0
total_model_unloads: int = 0
total_model_load_errors: int = 0
total_model_load_time: float = 0.0
total_chat_requests: int = 0
total_completion_requests: int = 0
total_stream_requests: int = 0
total_non_stream_requests: int = 0
total_auth_failures: int = 0
total_validation_failures: int = 0


# ============================================================================
# Pydantic schemas (Message, ChatCompletionRequest, ...) live in _schemas.py
# and are re-imported at the top of this file.


# ============================================================================
# Model Management
# ============================================================================

def get_available_models() -> List[str]:
    """List models in the registry that have OpenVINO model files"""
    models = []
    if os.path.isdir(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, name)
            if os.path.isdir(model_path):
                # Check for OpenVINO model files (standard or multi-modal)
                if (os.path.exists(os.path.join(model_path, "openvino_model.xml")) or
                    os.path.exists(os.path.join(model_path, "openvino_language_model.xml"))):
                    models.append(name)
    return models


def is_vision_model(model_path: str) -> bool:
    """Check if model is a vision/multi-modal model (uses VLMPipeline)"""
    # Vision models have openvino_language_model.xml instead of openvino_model.xml
    return os.path.exists(os.path.join(model_path, "openvino_language_model.xml"))


def load_model_metadata(model_name: str) -> Dict[str, Any]:
    model_json = os.path.join(MODELS_DIR, model_name, "model.json")
    try:
        if os.path.exists(model_json):
            with open(model_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as exc:
        print(f"[WARN] Failed to read {model_json}: {exc}")
    return {}


def get_model_overrides(model_name: str) -> Dict[str, Any]:
    metadata = load_model_metadata(model_name)
    overrides = metadata.get("overrides")
    if isinstance(overrides, dict):
        return overrides
    return {}


def get_model_override_int(model_name: str, key: str) -> Optional[int]:
    if key not in MODEL_OVERRIDE_KEYS:
        return None
    overrides = get_model_overrides(model_name)
    if key not in overrides:
        return None
    raw = overrides.get(key)
    try:
        return int(raw)
    except Exception:
        print(f"[WARN] Invalid model override for {model_name}.{key}: {raw!r}")
        return None


def get_effective_int_limit(model_name: str, key: str, global_default: int) -> int:
    override = get_model_override_int(model_name, key)
    return override if override is not None else global_default


def _coerce_bool(raw: Any) -> Optional[bool]:
    """Parse a JSON-ish boolean, returning None when it isn't one."""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return None


def _get_model_override_bool(model_name: str, key: str, global_default: bool) -> bool:
    """Read a boolean per-model override, falling back to the server default."""
    if key not in MODEL_OVERRIDE_KEYS:
        return global_default
    parsed = _coerce_bool(get_model_overrides(model_name).get(key))
    return global_default if parsed is None else parsed


def get_recommended_stream(model_name: str) -> Optional[bool]:
    return _coerce_bool(get_model_overrides(model_name).get("recommended_stream"))


# Track if current model is VLM
current_model_is_vlm: bool = False
# Which pipeline backend the loaded model actually got ("continuous" or
# "stateful"). Surfaced on /illama/health and /illama/stats so an operator can
# tell at a glance whether PagedAttention / speculative decoding is active,
# rather than inferring it from throughput.
current_model_backend: str = "none"


def _set_current_backend(backend: str) -> None:
    global current_model_backend
    current_model_backend = backend


def _build_pipeline_properties() -> Dict[str, Any]:
    """Device properties passed as **kwargs to the pipeline constructor."""
    props: Dict[str, Any] = {}
    if CACHE_DIR:
        props["CACHE_DIR"] = CACHE_DIR
    return props


def _build_scheduler_config(cache_gb: int, allow_prefix_caching: bool = True) -> Any:
    """SchedulerConfig for the continuous-batching (PagedAttention) backend.

    Returns None when the installed openvino-genai has no SchedulerConfig,
    which keeps this a no-op on older runtimes instead of raising at import.

    ``allow_prefix_caching`` exists because prefix caching and speculative
    decoding cannot be combined: with both enabled, generation dies at
    ``Check 'content_length <= prompt_ids.size() + m_generated_ids.size()'
    failed at src/cpp/src/sequence_group.cpp:32``. Reproduced on
    openvino-genai 2026.4.0.0 with Qwen3-1.7B + its AngelSlim eagle3 drafter;
    the same request succeeds with prefix caching off. Speculative decoding
    wins, since it is the larger speedup.
    """
    ctor = getattr(ov_genai, "SchedulerConfig", None)
    if ctor is None:
        return None
    sched = ctor()
    # Default cache_size is 0, which the allocator rejects for most models,
    # so always set it explicitly.
    if cache_gb > 0 and hasattr(sched, "cache_size"):
        sched.cache_size = cache_gb
    if hasattr(sched, "enable_prefix_caching"):
        sched.enable_prefix_caching = ENABLE_PREFIX_CACHING and allow_prefix_caching
    return sched


# Allocator complaints that mean "same config, just needs a bigger cache".
# Anything else is a real failure and should not be retried.
_CACHE_TOO_SMALL_PATTERNS = (
    "exceeds the configured cache size",
    "cache allocation",
    "not enough cache",
)


def _is_cache_too_small_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(p in text for p in _CACHE_TOO_SMALL_PATTERNS)


def _cache_size_ladder(start_gb: int, max_gb: int) -> List[int]:
    """Cache sizes to try, smallest first, doubling up to `max_gb`.

    Starting small matters: an oversized paged cache steals VRAM from the
    weights and reduces how large a model fits at all. We only grow when the
    allocator actually refuses.
    """
    if max_gb <= 0:
        return []
    sizes: List[int] = []
    gb = max(1, start_gb)
    while gb < max_gb:
        sizes.append(gb)
        gb *= 2
    sizes.append(max_gb)
    # De-duplicate while preserving ascending order.
    return sorted(set(s for s in sizes if 0 < s <= max_gb))


def _resolve_draft_model_dir_by_name(draft_name: str) -> Optional[str]:
    """Validate a draft model by registry name and return its directory.

    ov_genai.draft_model() loads a plain LLM IR and requires
    openvino_model.xml specifically; a VLM-shaped export fails deep in the
    OpenVINO frontend with an opaque file-not-found error.
    """
    name = os.path.basename(str(draft_name).strip())
    if not name:
        return None
    draft_path = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(draft_path):
        print(f"[WARN] draft_model '{name}' not found in registry")
        return None
    if not os.path.exists(os.path.join(draft_path, "openvino_model.xml")):
        if os.path.exists(os.path.join(draft_path, "openvino_language_model.xml")):
            print(
                f"[WARN] draft_model '{name}' is a VLM-shaped export "
                "(openvino_language_model.xml). Speculative decoding needs a "
                "text-only draft export containing openvino_model.xml; ignoring"
            )
        else:
            print(f"[WARN] draft_model '{name}' has no openvino_model.xml; ignoring")
        return None
    return draft_path


def _resolve_draft_model_dir(model_name: str) -> Optional[str]:
    """Resolve a per-model `draft_model` override to an on-disk directory.

    The override names another entry in the registry (not a HuggingFace repo
    id), so a drafter is pulled with `illama pull` like any other model and
    then linked. Returns None when unset or when the target doesn't look like
    a usable OpenVINO model directory.
    """
    overrides = get_model_overrides(model_name)
    raw = overrides.get("draft_model")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return _resolve_draft_model_dir_by_name(raw)


def resolve_model_alias(model_name: str) -> Tuple[str, Optional[str]]:
    """Map a requested model name to (effective_target, forced_draft).

    A draft model pulled via `illama pull <drafter>` records
    ``{"role": "draft", "base_model": "<base>"}`` in its model.json. Requesting
    the drafter by name then serves the base pipeline with that drafter
    attached, so "Qwen3-8B DFlash" behaves like a single selectable
    configuration rather than two downloads the caller has to know about.

    Returns (model_name, None) for ordinary models.
    """
    metadata = load_model_metadata(model_name)
    if str(metadata.get("role", "")).strip().lower() != "draft":
        return model_name, None
    base = metadata.get("base_model")
    if not isinstance(base, str) or not base.strip():
        return model_name, None
    base_name = os.path.basename(base.strip())
    if not os.path.isdir(os.path.join(MODELS_DIR, base_name)):
        print(f"[WARN] '{model_name}' points at missing base model '{base_name}'")
        return model_name, None
    return base_name, model_name


def _build_acceleration_kwargs(
    model_name: str, forced_draft: Optional[str] = None
) -> Dict[str, Any]:
    """Speculative-decoding / prompt-lookup kwargs, if configured for a model.

    Both features exist only on the continuous-batching backend.
    """
    extra: Dict[str, Any] = {}
    if forced_draft:
        draft_dir = _resolve_draft_model_dir_by_name(forced_draft)
    else:
        draft_dir = _resolve_draft_model_dir(model_name)
    if draft_dir and hasattr(ov_genai, "draft_model"):
        extra["draft_model"] = ov_genai.draft_model(draft_dir, DEFAULT_DEVICE)
        print(f"[INFO] Speculative decoding enabled, draft={os.path.basename(draft_dir)}")
    if _get_model_override_bool(model_name, "prompt_lookup", PROMPT_LOOKUP_DEFAULT):
        extra["prompt_lookup"] = True
        print("[INFO] Prompt-lookup decoding enabled")
    return extra


def _build_pipeline(
    model_path: str,
    model_name: str,
    is_vlm: bool,
    forced_draft: Optional[str] = None,
) -> Tuple[Any, str]:
    """Construct a pipeline, preferring the continuous-batching backend.

    Returns (pipeline, backend_name) where backend_name is "continuous" or
    "stateful", so callers can log/report which path actually took effect.

    Both LLMPipeline and VLMPipeline accept a SchedulerConfig - LLMPipeline
    via its `config` mapping (3rd positional), VLMPipeline via **kwargs, which
    land in the same ov::AnyMap on the C++ side. Speculative decoding
    (classic draft, EAGLE-3, DFlash) and prompt-lookup only exist there, so we
    try it first unless configured otherwise.

    Cache sizing is auto-escalated: linear-attention models (Qwen3.5's Gated
    DeltaNet) need far more paged cache than a plain KV model and refuse to
    construct below a threshold. We start small and double until the allocator
    is satisfied, so operators don't have to discover the floor by hand.
    """
    ctor = ov_genai.VLMPipeline if is_vlm else ov_genai.LLMPipeline
    properties = _build_pipeline_properties()

    if PIPELINE_BACKEND != "stateful" and hasattr(ov_genai, "SchedulerConfig"):
        start_gb = get_effective_int_limit(
            model_name, "scheduler_cache_size_gb", SCHEDULER_CACHE_SIZE_GB
        )
        ladder = _cache_size_ladder(start_gb, SCHEDULER_CACHE_MAX_GB)
        last_exc: Optional[Exception] = None

        for cache_gb in ladder:
            try:
                extra = _build_acceleration_kwargs(model_name, forced_draft)
                has_draft = "draft_model" in extra
                sched = _build_scheduler_config(
                    cache_gb, allow_prefix_caching=not has_draft
                )
                if sched is None:
                    break
                # scheduler_config goes through **kwargs for both pipeline
                # types. The positional `config` mapping that LLMPipeline also
                # accepts is deprecated in 2026.4 ("'config' parameters is
                # deprecated, please use kwargs").
                pipeline = ctor(
                    model_path,
                    DEFAULT_DEVICE,
                    scheduler_config=sched,
                    **properties,
                    **extra,
                )
                prefix_note = "" if not has_draft else ", prefix_caching=off (draft model)"
                print(
                    f"[INFO] Continuous-batching backend active "
                    f"(cache_size={cache_gb}GB{prefix_note})"
                )
                return pipeline, "continuous"
            except Exception as exc:
                last_exc = exc
                if _is_cache_too_small_error(exc) and cache_gb < ladder[-1]:
                    print(
                        f"[INFO] scheduler cache_size={cache_gb}GB too small for "
                        f"'{model_name}', retrying larger"
                    )
                    continue
                break

        if last_exc is not None:
            if PIPELINE_BACKEND == "continuous":
                # Explicitly requested - surface the failure rather than
                # silently degrading to a slower backend.
                raise last_exc
            print(
                f"[WARN] Continuous-batching pipeline unavailable for '{model_name}' "
                f"({_clean_openvino_error(last_exc)}); falling back to stateful backend"
            )

    return ctor(model_path, DEFAULT_DEVICE, **properties), "stateful"


# Patterns in OpenVINO GenAI exception text that indicate a *config* error
# (the runtime literally cannot represent this model) rather than a transient
# GPU state error. Restarting the container won't change which model_types
# the C++ vlm_config / llm_config code knows about, so we surface a clean
# 400 to the client and KEEP the runner alive instead of triggering the
# Phase 2 graceful-exit loop. New patterns can be appended over time.
_LOAD_ERROR_USER_PATTERNS = (
    "unsupported '",                # "Unsupported 'gemma4' VLM model type"
    "vlm model type",
    "llm model type",
    "model_type",
    "unsupported architecture",
    "unsupported model_type",
    "failed to open",               # missing tokenizer.xml etc.
    "failed to read",
    "no such file",
    "invalid model format",
)


def _clean_openvino_error(exc: Exception) -> str:
    """Return a single-line, build-path-stripped summary of an OV exception.

    OpenVINO's ``what()`` typically looks like::

        Exception from /home/jenkins/.../vlm_config.cpp:34:
        Unsupported 'gemma4' VLM model type

    We strip the Jenkins build path (noisy + leaks no useful info) and
    collapse to the first non-empty line after it.
    """
    text = str(exc).strip()
    if not text:
        return exc.__class__.__name__
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    # Drop the leading "Exception from <path>:<line>:" framing if present.
    if lines[0].lower().startswith("exception from ") and len(lines) > 1:
        return lines[1]
    return lines[0]


def _is_user_load_error(exc: Exception) -> bool:
    """Heuristic: True if this load error means 'this model can't load on
    this runtime', not 'the GPU is in a bad state'."""
    text = str(exc).lower()
    return any(p in text for p in _LOAD_ERROR_USER_PATTERNS)


def load_model(model_name: str) -> Tuple[Any, bool]:
    """Load a model, unloading any previously loaded model.

    Returns (pipeline, is_vlm) so callers don't need to read global state.
    """
    global current_model, current_model_name, current_model_is_vlm, last_request_time

    # A drafter pulled with `illama pull <drafter>` records its base model, so
    # requesting the drafter by name serves base+draft as one configuration.
    effective_name, forced_draft = resolve_model_alias(model_name)
    if forced_draft:
        print(f"[INFO] '{model_name}' is an alias for base '{effective_name}' + draft")

    model_path = os.path.join(MODELS_DIR, effective_name)

    if not os.path.isdir(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

    if not (os.path.exists(os.path.join(model_path, "openvino_model.xml")) or
            os.path.exists(os.path.join(model_path, "openvino_language_model.xml"))):
        raise HTTPException(status_code=400, detail=f"Invalid model format: {model_name}")

    with model_lock:
        # Unload current model if different
        if current_model_name != model_name:
            if current_model is not None:
                # Skip the explicit driver-reclaim sleep: a model switch is
                # immediately followed by a multi-second pipeline construction
                # which gives the GPU driver plenty of reclaim headroom, and
                # the sleep would otherwise hold model_lock for 2s while any
                # other pending request waits.
                unload_current_model_locked("model-switch", sleep_for_driver=False)

            is_vlm = is_vision_model(model_path)
            pipeline_type = "VLMPipeline" if is_vlm else "LLMPipeline"
            properties = _build_pipeline_properties()

            print(f"[INFO] Loading model: {model_name} on {DEFAULT_DEVICE} ({pipeline_type})")
            if properties:
                print(f"[INFO] Pipeline properties: {properties}")
            load_started_at = time.time()
            try:
                current_model, backend = _build_pipeline(
                    model_path, effective_name, is_vlm, forced_draft=forced_draft
                )
                current_model_name = model_name
                current_model_is_vlm = is_vlm
                _set_current_backend(backend)
                load_time = time.time() - load_started_at
                _record_model_load_success(model_name, load_time)
                print(
                    f"[INFO] Model loaded successfully: {model_name} "
                    f"({load_time:.2f}s, backend={backend})"
                )
            except Exception as e:
                current_model = None
                current_model_name = None
                current_model_is_vlm = False
                _record_model_load_error(model_name)
                cleaned = _clean_openvino_error(e)

                # User/config errors (e.g. "Unsupported '<arch>' VLM model
                # type" because the pinned OpenVINO GenAI build doesn't know
                # that architecture yet) are not GPU-state issues. Restarting won't
                # change which model_types the C++ runtime knows about, so
                # return 400 IMMEDIATELY and keep the runner alive. The
                # client gets the actual reason in the error envelope and
                # can switch to a supported model without waiting for the
                # container to recycle.
                if _is_user_load_error(e):
                    print(
                        f"[ERROR] model '{model_name}' is incompatible with "
                        f"this OpenVINO runtime: {cleaned}"
                    )
                    raise_openai_http_error(
                        400,
                        (
                            f"Model '{model_name}' cannot be loaded by this "
                            f"OpenVINO runtime: {cleaned}. The model files "
                            "are valid; the installed runtime version doesn't "
                            "yet support this architecture. Try a different "
                            "model or upgrade openvino-genai."
                        ),
                        error_type="invalid_request_error",
                        code="model_incompatible_with_runtime",
                        param="model",
                    )

                if LOAD_FAILURE_POLICY == "return_error":
                    raise_openai_http_error(
                        500,
                        f"Failed to load model '{model_name}': {cleaned}",
                        error_type="server_error",
                        code="model_load_failed",
                    )
                # restart policy: schedule a delayed os._exit so the current
                # request still gets a clean 503 response (and any other
                # in-flight responses can drain) before Docker reaps the
                # container and starts a fresh one with a clean GPU state.
                print(f"[FATAL] GPU load failed: {e}")
                _schedule_fatal_exit(
                    f"load_failure_policy=restart, model_load_failed for '{model_name}'"
                )
                raise_openai_http_error(
                    503,
                    (
                        f"Failed to load model '{model_name}' ({cleaned}); "
                        "the runner is restarting to recover GPU state. "
                        "Retry in a few seconds."
                    ),
                    error_type="server_error",
                    code="model_load_failed_restarting",
                )

        last_request_time = time.time()
        return current_model, current_model_is_vlm


def _record_success(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    generation_time: float,
    first_token_time: Optional[float] = None,
) -> None:
    """Record a successful generation.

    ``first_token_time`` is the wall-clock seconds from request-start to the
    first streamed token, and is recorded only for streaming requests that
    actually emitted at least one token. Non-streaming and zero-output
    requests pass None and do not contribute to the TTFT averages.
    """
    global total_requests, total_prompt_tokens, total_completion_tokens, total_generation_time
    global total_first_token_time, total_first_token_samples
    with stats_lock:
        total_requests += 1
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_generation_time += generation_time
        if first_token_time is not None and first_token_time > 0:
            total_first_token_time += first_token_time
            total_first_token_samples += 1

        model_stats = per_model_stats.setdefault(
            model_name,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "generation_time": 0.0,
            },
        )
        model_stats["requests"] += 1
        model_stats["prompt_tokens"] += prompt_tokens
        model_stats["completion_tokens"] += completion_tokens
        model_stats["generation_time"] += generation_time
        if first_token_time is not None and first_token_time > 0:
            model_stats["first_token_time"] = (
                model_stats.get("first_token_time", 0.0) + first_token_time
            )
            model_stats["first_token_samples"] = (
                model_stats.get("first_token_samples", 0) + 1
            )
        model_stats["last_activity_ts"] = time.time()


def _record_error(model_name: Optional[str]) -> None:
    global total_errors
    if not model_name:
        return
    with stats_lock:
        total_errors += 1
        model_stats = per_model_stats.setdefault(
            model_name,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "generation_time": 0.0,
                "errors": 0,
            },
        )
        model_stats.setdefault("errors", 0)
        model_stats["errors"] += 1
        model_stats["last_activity_ts"] = time.time()


def record_request_shape(is_chat: bool, stream: bool) -> None:
    global total_chat_requests, total_completion_requests, total_stream_requests, total_non_stream_requests
    with stats_lock:
        if is_chat:
            total_chat_requests += 1
        else:
            total_completion_requests += 1
        if stream:
            total_stream_requests += 1
        else:
            total_non_stream_requests += 1


def touch_last_request_time() -> None:
    global last_request_time
    last_request_time = time.time()


def begin_generation() -> None:
    global active_generations
    with model_lock:
        active_generations += 1
    touch_last_request_time()


def end_generation() -> None:
    global active_generations
    with model_lock:
        if active_generations > 0:
            active_generations -= 1
    touch_last_request_time()


def _record_model_load_success(model_name: str, load_time: float) -> None:
    global total_model_loads, total_model_load_time
    with stats_lock:
        total_model_loads += 1
        total_model_load_time += load_time
        model_stats = per_model_stats.setdefault(
            model_name,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "generation_time": 0.0,
            },
        )
        model_stats["loads"] = model_stats.get("loads", 0) + 1
        model_stats["load_time"] = model_stats.get("load_time", 0.0) + load_time
        model_stats["last_activity_ts"] = time.time()


def _record_model_load_error(model_name: str) -> None:
    global total_model_load_errors
    with stats_lock:
        total_model_load_errors += 1
        model_stats = per_model_stats.setdefault(
            model_name,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "generation_time": 0.0,
            },
        )
        model_stats["load_errors"] = model_stats.get("load_errors", 0) + 1
        model_stats["last_activity_ts"] = time.time()


def _record_model_unload(model_name: Optional[str]) -> None:
    global total_model_unloads
    if not model_name:
        return
    with stats_lock:
        total_model_unloads += 1
        model_stats = per_model_stats.setdefault(
            model_name,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "generation_time": 0.0,
            },
        )
        model_stats["unloads"] = model_stats.get("unloads", 0) + 1
        model_stats["last_activity_ts"] = time.time()


def _snapshot_inference_stats() -> Dict[str, Any]:
    """Copy out all stats counters under stats_lock and return them.

    Callers MUST do any disk I/O (e.g. get_effective_request_limits) AFTER
    this returns - the goal of this helper is to keep stats_lock hold time
    bounded to a couple of dict copies so that /metrics and /illama/stats
    do not block hot-path _record_success calls during scrape.
    """
    with stats_lock:
        return {
            "uptime": time.time() - server_start_time,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "total_chat_requests": total_chat_requests,
            "total_completion_requests": total_completion_requests,
            "total_stream_requests": total_stream_requests,
            "total_non_stream_requests": total_non_stream_requests,
            "total_auth_failures": total_auth_failures,
            "total_validation_failures": total_validation_failures,
            "total_model_loads": total_model_loads,
            "total_model_unloads": total_model_unloads,
            "total_model_load_errors": total_model_load_errors,
            "total_model_load_time": total_model_load_time,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_generation_time": total_generation_time,
            "total_first_token_time": total_first_token_time,
            "total_first_token_samples": total_first_token_samples,
            "per_model": {name: dict(m) for name, m in per_model_stats.items()},
        }


# Empirical wait for the Intel GPU driver to fully reclaim VRAM after a model
# pipeline is dropped. Two seconds is what the original implementation used;
# kept as a named constant so callers can reuse it after dropping model_lock.
_GPU_DRIVER_RECLAIM_SECONDS = 2


def unload_current_model_locked(reason: str, sleep_for_driver: bool = True) -> bool:
    """Drop the loaded model + record the unload. MUST be called with model_lock held.

    Returns True if the caller should perform a post-unload GPU-driver reclaim
    sleep (currently ``_GPU_DRIVER_RECLAIM_SECONDS`` seconds). The sleep is
    deliberately NOT performed here so the caller can release ``model_lock``
    first - holding the lock for two seconds while idle would otherwise block
    every concurrent request.

    The legacy keyword is kept for backwards compatibility with callers that
    explicitly pass ``sleep_for_driver=False``: when False, this function
    returns False unconditionally so the caller skips the sleep. Callers that
    want the sleep should release the lock and then check the return value.
    """
    global current_model, current_model_name, current_model_is_vlm
    if current_model is None:
        return False

    unloaded_name = current_model_name
    print(f"[INFO] Unloading model: {unloaded_name} ({reason})")

    current_model = None
    current_model_name = None
    current_model_is_vlm = False
    _set_current_backend("none")
    _record_model_unload(unloaded_name)

    import gc

    gc.collect()

    return bool(sleep_for_driver)


# is_auth_exempt_path lives in _helpers.py (along with the underlying
# AUTH_EXEMPT_PATHS constant). It is re-imported at the top of this file.


def is_authorized(request: Request) -> bool:
    if not AUTH_ENABLED or is_auth_exempt_path(request.url.path):
        return True
    header = request.headers.get("authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        return False
    token = header[len(prefix):].strip()
    return secrets.compare_digest(token, API_KEY)


def log_request_start(endpoint: str, model_name: str, stream: bool) -> float:
    started_at = time.time()
    mode = "stream" if stream else "non-stream"
    print(f"[INFO] {endpoint} started model={model_name} mode={mode}")
    return started_at


def log_request_finish(
    endpoint: str,
    model_name: str,
    started_at: float,
    completion_tokens: int,
    success: bool,
) -> None:
    duration = time.time() - started_at
    status = "ok" if success else "error"
    print(
        f"[INFO] {endpoint} finished model={model_name} "
        f"status={status} duration={duration:.2f}s completion_tokens={completion_tokens}"
    )


def build_prompt(messages: List[Message]) -> str:
    """Fallback prompt format when no model chat template is available.

    Multimodal content lists are flattened to text via _message_text so this
    helper keeps producing a string regardless of whether the caller passed
    plain-string content or a list of OpenAI content parts.
    """
    prompt_parts = []
    for msg in messages:
        text = _message_text(msg)
        if msg.role == "system":
            prompt_parts.append(f"<|system|>\n{text}\n")
        elif msg.role == "user":
            prompt_parts.append(f"<|user|>\n{text}\n")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{text}\n")
        else:
            prompt_parts.append(f"<|{msg.role}|>\n{text}\n")
    prompt_parts.append("<|assistant|>\n")
    return "".join(prompt_parts)


def resolve_max_new_tokens(
    request_value: Optional[int],
    request_completion_value: Optional[int],
    configured_default: int,
    model_name: Optional[str] = None,
    override_key: Optional[str] = None,
) -> int:
    """Resolve request or config defaults for max_new_tokens."""
    if request_value is not None:
        return request_value
    if request_completion_value is not None:
        return request_completion_value
    if model_name and override_key:
        override = get_model_override_int(model_name, override_key)
        if override is not None and override > 0:
            return override
    if configured_default > 0:
        return configured_default
    return FALLBACK_MAX_NEW_TOKENS


def build_generation_config(
    model_name: str,
    override_key: str,
    request_max_tokens: Optional[int],
    request_max_completion_tokens: Optional[int],
    configured_default: int,
    temperature: Optional[float],
    top_p: Optional[float],
    seed: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> ov_genai.GenerationConfig:
    """Build a generation config from request + server defaults."""
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = resolve_max_new_tokens(
        request_max_tokens,
        request_max_completion_tokens,
        configured_default,
        model_name=model_name,
        override_key=override_key,
    )
    config.temperature = temperature if temperature is not None else 0.7
    if hasattr(config, "do_sample"):
        config.do_sample = bool(config.temperature and config.temperature > 0)
    if hasattr(config, "top_p") and top_p is not None:
        config.top_p = top_p
    if seed is not None and hasattr(config, "rng_seed"):
        try:
            config.rng_seed = int(seed)
        except Exception as exc:
            print(f"[WARN] Failed to apply seed={seed}: {exc}")
    if repetition_penalty is not None and hasattr(config, "repetition_penalty"):
        try:
            config.repetition_penalty = float(repetition_penalty)
        except Exception as exc:
            print(f"[WARN] Failed to apply repetition_penalty={repetition_penalty}: {exc}")
    # Speculative decoding: how many candidate tokens the drafter proposes per
    # iteration. Only meaningful when the model has a linked draft_model;
    # harmless otherwise. Defaults to 5 inside openvino-genai when unset.
    assistant_tokens = get_model_override_int(model_name, "num_assistant_tokens")
    if assistant_tokens and assistant_tokens > 0 and hasattr(config, "num_assistant_tokens"):
        try:
            config.num_assistant_tokens = int(assistant_tokens)
        except Exception as exc:
            print(f"[WARN] Failed to apply num_assistant_tokens={assistant_tokens}: {exc}")
    return config


def apply_response_format(
    messages: List[Message],
    response_format: Optional[Dict[str, Any]],
) -> List[Message]:
    """If the client asks for JSON output, prepend a system instruction.

    OpenVINO GenAI doesn't have constrained JSON sampling, so we approximate
    by prompting. This still helps small/medium models stay in shape.
    """
    if not response_format:
        return messages
    fmt_type = str(response_format.get("type", "")).lower()
    if fmt_type != "json_object":
        return messages
    instruction = (
        "You must respond with a single valid JSON object and no other text, "
        "no code fences, no commentary."
    )
    return [Message(role="system", content=instruction)] + list(messages)


def apply_stop_sequences(
    config: ov_genai.GenerationConfig,
    stop: Optional[List[str]],
) -> None:
    if not stop:
        return
    if hasattr(config, "stop_strings"):
        try:
            config.stop_strings = set(stop)
            return
        except Exception as exc:
            print(
                f"[DEBUG] GenerationConfig.stop_strings assignment failed, "
                f"falling through to stop_token_ids: {exc!r}"
            )
    if hasattr(config, "stop_token_ids"):
        print("[WARN] stop sequences requested but stop token id mapping is unavailable")


# ============================================================================
# Multimodal (VLM) input handling
# ============================================================================
#
# OpenVINO GenAI's VLMPipeline.generate accepts a prompt string + a list of
# ov.Tensor images (HWC, uint8, RGB) and applies the model's chat template +
# image-token insertion internally. We therefore decode each image source the
# client sends (OpenAI image_url base64 / https URL, Ollama raw base64) into
# ov.Tensor here, and let the pipeline take it from there.

# 20 MB hard cap per image so a malicious client cannot cause an OOM by
# pointing us at a 4 GB JPEG.
MAX_IMAGE_BYTES = 20 * 1024 * 1024
_HTTP_FETCH_TIMEOUT_SECONDS = 10

# _DATA_URL_RE and _b64decode_lenient live in _helpers.py and are re-imported
# at the top of this file.

# SSRF guard for VLM image_url fetches.
#
# `_assert_image_url_is_safe` resolves the host's IP and rejects link-local
# (cloud metadata services live at 169.254.169.254), loopback, private
# RFC1918, multicast, and reserved ranges. This is best-effort: a malicious
# DNS server can still rebind between the resolution we do here and the
# connection requests makes, so this is a defence-in-depth measure rather
# than a complete fix. Operators that need stronger isolation should run the
# runner in a network-restricted Docker network or behind an egress proxy.
#
# Set ILLAMA_ALLOW_PRIVATE_IMAGE_URLS=1 to disable the check (e.g. when the
# runner sits inside a private network and intentionally fetches from a
# colocated image server).
ALLOW_PRIVATE_IMAGE_URLS = os.environ.get(
    "ILLAMA_ALLOW_PRIVATE_IMAGE_URLS", ""
).strip().lower() in {"1", "true", "yes", "on"}


def _assert_image_url_is_safe(url: str) -> None:
    """Reject URLs that resolve to a private / link-local / loopback IP.

    Raises HTTPException(400) if the URL is unsafe. Returns silently if the
    URL is OK to fetch. ``ILLAMA_ALLOW_PRIVATE_IMAGE_URLS=1`` bypasses the
    check entirely.
    """
    if ALLOW_PRIVATE_IMAGE_URLS:
        return
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image URL: {exc}") from exc
    host = parsed.hostname
    if not host:
        raise HTTPException(status_code=400, detail=f"Image URL has no host: {url!r}")
    try:
        # Resolve all addresses (v4 + v6) and reject if ANY is unsafe; that
        # way we cannot be tricked by a hostname that returns 1.2.3.4 plus
        # 127.0.0.1 and have requests pick the loopback.
        addrinfo = socket.getaddrinfo(host, None)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resolve image host {host!r}: {exc}",
        ) from exc
    for entry in addrinfo:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Refusing to fetch image from {host!r} (resolves to {ip_str}). "
                    "Private/loopback/link-local addresses are blocked. Set "
                    "ILLAMA_ALLOW_PRIVATE_IMAGE_URLS=1 to override."
                ),
            )


def _read_url_bytes(url: str) -> bytes:
    """Fetch the bytes at an http(s) URL with a hard timeout and size cap.

    The size cap protects against a malicious server feeding us an unbounded
    response. The timeout prevents a request from hanging the worker. The
    SSRF guard rejects URLs that resolve to private / loopback / link-local
    addresses (e.g. cloud metadata services) before any HTTP round trip.
    """
    _assert_image_url_is_safe(url)
    try:
        resp = requests.get(url, timeout=_HTTP_FETCH_TIMEOUT_SECONDS, stream=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        if cl and cl.isdigit() and int(cl) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Image at {url!r} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB cap",
            )
        chunks: List[bytes] = []
        size = 0
        for chunk in resp.iter_content(chunk_size=131_072):
            if not chunk:
                continue
            size += len(chunk)
            if size > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Image at {url!r} exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB cap",
                )
            chunks.append(chunk)
        return b"".join(chunks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch image from {url!r}: {exc}",
        ) from exc


def _decode_image_to_tensor(src: str) -> ov.Tensor:
    """Decode an image source into an OpenVINO Tensor (HWC, uint8, RGB).

    Accepts:
      - 'data:image/<type>;base64,<payload>' (OpenAI multimodal default)
      - 'http(s)://...'                     (OpenAI image_url with URL)
      - raw base64                          (Ollama 'images' field)
    """
    src = src.strip()
    if not src:
        raise HTTPException(status_code=400, detail="Empty image source")

    m = _DATA_URL_RE.match(src)
    if m:
        try:
            data = _b64decode_lenient(m.group("payload"))
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 in data URL: {exc}",
            ) from exc
    elif src.startswith(("http://", "https://")):
        data = _read_url_bytes(src)
    else:
        try:
            data = _b64decode_lenient(src)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Image source must be a 'data:...;base64,...' URL, an "
                    "http(s):// URL, or a raw base64 string"
                ),
            ) from exc

    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB cap",
        )

    try:
        with PILImage.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to decode image: {exc}",
        ) from exc

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Decoded image has unexpected shape {arr.shape}",
        )
    # Make sure the buffer is contiguous before handing it to ov.Tensor;
    # numpy slicing or color conversion can leave a non-contiguous view that
    # OpenVINO refuses.
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return ov.Tensor(arr)


def _message_text(msg: Message) -> str:
    """Plain-text content of a message, dropping any image parts.

    Images are surfaced separately via _extract_images_from_messages, so the
    existing chat-template / fallback prompt builders can keep treating
    content as a flat string.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for part in content:
        if getattr(part, "type", None) == "text":
            parts.append(getattr(part, "text", "") or "")
    return "\n".join(p for p in parts if p)


def _extract_images_from_messages(messages: List[Message]) -> List[ov.Tensor]:
    """Decode every OpenAI image_url part in the conversation."""
    images: List[ov.Tensor] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            continue
        for part in content:
            if getattr(part, "type", None) == "image_url":
                url = getattr(getattr(part, "image_url", None), "url", "") or ""
                if url:
                    images.append(_decode_image_to_tensor(url))
    return images


def _extract_ollama_chat_images(
    messages: List["OllamaChatMessage"],
) -> List[ov.Tensor]:
    """Decode every Ollama-style 'images' entry in the conversation."""
    images: List[ov.Tensor] = []
    for msg in messages:
        srcs = getattr(msg, "images", None)
        if not srcs:
            continue
        for src in srcs:
            if src:
                images.append(_decode_image_to_tensor(src))
    return images


def _vlm_generate(
    pipeline: Any,
    prompt: str,
    generation_config: Optional[ov_genai.GenerationConfig],
    images: Optional[List[ov.Tensor]],
    streamer: Optional[Any] = None,
) -> Any:
    """Single helper used by every VLM call site to call VLMPipeline.generate
    with whatever subset of (generation_config, images, streamer) is set.

    Keeps all four call sites (OpenAI stream/non-stream, Ollama stream/non-
    stream) in sync about exactly which kwargs are forwarded.
    """
    kwargs: Dict[str, Any] = {}
    if generation_config is not None:
        kwargs["generation_config"] = generation_config
    if images:
        kwargs["images"] = images
    if streamer is not None:
        kwargs["streamer"] = streamer
    return pipeline.generate(prompt, **kwargs)


def generate_with_config(
    pipeline: Any,
    prompt: str,
    generation_config: ov_genai.GenerationConfig,
    is_vlm: bool,
    images: Optional[List[ov.Tensor]] = None,
) -> str:
    if is_vlm:
        result = _vlm_generate(pipeline, prompt, generation_config, images)
        return str(result) if not isinstance(result, str) else result
    return pipeline.generate(prompt, generation_config)


def _count_encoded_tokens(encoded: Any) -> int:
    input_ids = getattr(encoded, "input_ids", None)
    shape = getattr(input_ids, "shape", None)
    if shape is None:
        return 0
    if len(shape) == 0:
        return 0
    return int(shape[-1])


def count_tokens(text: str, pipeline: Optional[Any] = None) -> int:
    """Count tokens with the model tokenizer and fall back to a word count."""
    if pipeline is not None:
        try:
            tokenizer = pipeline.get_tokenizer()
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                encoded = tokenizer.encode(text)
            count = _count_encoded_tokens(encoded)
            if count > 0:
                return count
        except Exception as exc:
            print(f"[WARN] Failed to count tokens with tokenizer: {exc}")
    return len(text.split())


def build_chat_prompt(
    pipeline: Any,
    messages: List[Message],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Use tokenizer chat templates when available, otherwise use fallback formatting.

    Multimodal content lists are flattened to text before the chat template is
    applied. Image tokens are inserted by VLMPipeline.generate itself when
    `images=` is passed alongside this prompt - we deliberately do NOT bake
    `<start_of_image>` markers (or model-specific equivalents) into the prompt,
    because OpenVINO GenAI's VLMPipeline owns that semantics and gets it right
    per-model. If we ever need fine-grained image positioning across multiple
    user turns, switch to start_chat()/finish_chat() and replay history; for
    typical single-turn vision queries the current path works.

    ``tools`` is the OpenAI / Ollama tool-list schema. When supplied, we pass
    it to the tokenizer's chat template - Qwen 2.5/3, DeepSeek, Hermes-Pro,
    and most other recent chat templates accept ``tools=`` and emit a Hermes-3
    style tool-list block in the prompt so the model knows what's callable.
    Older templates that don't accept ``tools`` ignore the kwarg silently
    (we fall back to the no-tools call path on TypeError).
    """
    try:
        tokenizer = pipeline.get_tokenizer()
        if hasattr(tokenizer, "apply_chat_template"):
            template_messages = [
                {"role": msg.role, "content": _message_text(msg)}
                for msg in messages
            ]
            template_kwargs: Dict[str, Any] = {
                "add_generation_prompt": True,
                "tokenize": False,
            }
            if tools:
                template_kwargs["tools"] = tools
            try:
                prompt = tokenizer.apply_chat_template(template_messages, **template_kwargs)
            except TypeError:
                # Older template signature: drop the kwargs it doesn't know.
                fallback_kwargs = dict(template_kwargs)
                fallback_kwargs.pop("tokenize", None)
                fallback_kwargs.pop("tools", None)
                try:
                    prompt = tokenizer.apply_chat_template(template_messages, **fallback_kwargs)
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        template_messages,
                        add_generation_prompt=True,
                    )
            if isinstance(prompt, str) and prompt:
                return prompt
    except Exception as exc:
        print(f"[WARN] Chat template unavailable, using fallback prompt: {exc}")
    return build_prompt(messages)


# OpenAI error/stream chunk/usage helpers (make_openai_error, sse_encode,
# build_chat_stream_chunk, build_completion_stream_chunk,
# make_stream_error_event, build_usage_payload) live in _helpers.py and are
# re-imported at the top of this file.


_STREAM_ENQUEUE_HARD_DEADLINE_SECONDS = 30.0
_STREAM_ENQUEUE_POLL_SECONDS = 0.1


def _stream_stop_status() -> Any:
    return getattr(ov_genai.StreamingStatus, "STOP", ov_genai.StreamingStatus.RUNNING)


def create_streaming_response(
    *,
    http_request: Request,
    endpoint: str,
    model_name: str,
    pipeline: Any,
    prompt: str,
    generation_config: Optional[ov_genai.GenerationConfig],
    is_chat: bool,
    is_vlm: bool,
    include_usage: bool,
    images: Optional[List[ov.Tensor]] = None,
    parse_output: bool = False,
) -> StreamingResponse:
    completion_id = ("chatcmpl" if is_chat else "cmpl") + f"-{int(time.time())}"
    created = int(time.time())
    started_at = log_request_start(endpoint, model_name, stream=True)
    # Bounded so a stalled SSE consumer cannot grow memory unboundedly.
    # The worker uses cancel_event-aware non-blocking put via _enqueue() so a
    # disconnected client never deadlocks the producer thread.
    events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)
    streamed_parts: List[str] = []
    cancel_event = threading.Event()
    # Per-request parser. Only used on the chat path with parse_output=True;
    # for /v1/completions the raw text is the answer, so we don't tag-parse.
    parser: Optional[StreamingParser] = StreamingParser() if (parse_output and is_chat) else None
    saw_tool_call: List[bool] = [False]
    # Captured the first time the streamer fires with non-empty text. Used
    # to record TTFT separately from total generation time so dashboards can
    # report prompt-processing latency vs decode speed independently.
    first_token_at: List[float] = []

    def _enqueue(event: Dict[str, Any]) -> bool:
        """Cancel-aware enqueue.

        Returns True on success, False if the consumer cancelled or capacity
        was unavailable for longer than the hard deadline. Callers must NOT
        retry on False - the worker is expected to bail and let `finally` clean
        up so end_generation() runs and active_generations is released.
        """
        if cancel_event.is_set():
            return False
        deadline = time.monotonic() + _STREAM_ENQUEUE_HARD_DEADLINE_SECONDS
        while True:
            try:
                events.put(event, timeout=_STREAM_ENQUEUE_POLL_SECONDS)
                return True
            except queue.Full:
                if cancel_event.is_set() or time.monotonic() > deadline:
                    return False

    def _emit_chat_delta(delta: Dict[str, Any]) -> bool:
        return _enqueue(
            {
                "type": "chunk",
                "payload": build_chat_stream_chunk(
                    completion_id,
                    created,
                    model_name,
                    delta,
                ),
            }
        )

    def emit_chunk(text: str) -> bool:
        """Emit a delta for raw streamer output.

        For chat with parser enabled, route ``text`` through the parser and
        emit ``content`` / ``reasoning_content`` / ``tool_calls`` deltas
        accordingly. Otherwise emit ``text`` as-is.
        """
        if is_chat and parser is not None:
            return _flush_parser_events(parser.feed(text))
        if is_chat:
            return _emit_chat_delta({"content": text})
        return _enqueue(
            {
                "type": "chunk",
                "payload": build_completion_stream_chunk(
                    completion_id,
                    created,
                    model_name,
                    text,
                ),
            }
        )

    def _flush_parser_events(events: List[Tuple[str, Any]]) -> bool:
        for kind, payload in events:
            if kind == "content":
                if not _emit_chat_delta({"content": payload}):
                    return False
            elif kind == "reasoning":
                # OpenAI / DeepSeek convention: reasoning_content alongside
                # content. Clients that don't know the field ignore it,
                # OpenWebUI and Continue.dev render it in a separate panel.
                if not _emit_chat_delta({"reasoning_content": payload}):
                    return False
            elif kind == "tool":
                saw_tool_call[0] = True
                # Emit the entire tool_call atomically. Most clients
                # (OpenWebUI, Cline, Continue.dev) accept this just fine
                # because they rebuild the call from streamed deltas anyway.
                if not _emit_chat_delta({"tool_calls": [payload]}):
                    return False
        return True

    def streamer(token: Any) -> Any:
        if cancel_event.is_set():
            return _stream_stop_status()
        text = str(token)
        if text:
            if not first_token_at:
                first_token_at.append(time.time())
            touch_last_request_time()
            streamed_parts.append(text)
            if not emit_chunk(text):
                return _stream_stop_status()
        return ov_genai.StreamingStatus.RUNNING

    def worker() -> None:
        completion_tokens = 0
        try:
            if is_vlm:
                result = _vlm_generate(
                    pipeline,
                    prompt,
                    generation_config,
                    images,
                    streamer=streamer,
                )
            elif generation_config is not None:
                result = pipeline.generate(prompt, generation_config, streamer=streamer)
            else:
                result = pipeline.generate(prompt, streamer=streamer)

            output = "".join(streamed_parts).strip("\x00")
            if not output and result is not None:
                output = str(result)

            generation_time = time.time() - started_at
            ttft = (first_token_at[0] - started_at) if first_token_at else None
            prompt_tokens = count_tokens(prompt, pipeline)
            completion_tokens = count_tokens(output, pipeline)
            _record_success(
                model_name,
                prompt_tokens,
                completion_tokens,
                generation_time,
                first_token_time=ttft,
            )
            usage_payload = build_usage_payload(prompt_tokens, completion_tokens, generation_time)

            if is_chat:
                # Flush any held-back parser bytes so the last few chars
                # don't disappear when generation ends mid-buffer.
                if parser is not None:
                    if not _flush_parser_events(parser.finalize()):
                        log_request_finish(endpoint, model_name, started_at, completion_tokens, success=True)
                        return

                # OpenAI-correct finish_reason: 'tool_calls' when we surfaced
                # any tool, else 'stop'. Clients use this to decide whether
                # to dispatch the tool.
                finish_reason = "tool_calls" if saw_tool_call[0] else "stop"
                if not _enqueue(
                    {
                        "type": "chunk",
                        "payload": build_chat_stream_chunk(
                            completion_id,
                            created,
                            model_name,
                            {},
                            finish_reason=finish_reason,
                        ),
                    }
                ):
                    log_request_finish(endpoint, model_name, started_at, completion_tokens, success=True)
                    return
                if include_usage and not _enqueue(
                    {
                        "type": "chunk",
                        "payload": {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": usage_payload,
                        },
                    }
                ):
                    log_request_finish(endpoint, model_name, started_at, completion_tokens, success=True)
                    return
            else:
                if not _enqueue(
                    {
                        "type": "chunk",
                        "payload": build_completion_stream_chunk(
                            completion_id,
                            created,
                            model_name,
                            "",
                            finish_reason="stop",
                        ),
                    }
                ):
                    log_request_finish(endpoint, model_name, started_at, completion_tokens, success=True)
                    return
                if include_usage and not _enqueue(
                    {
                        "type": "chunk",
                        "payload": {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [],
                            "usage": usage_payload,
                        },
                    }
                ):
                    log_request_finish(endpoint, model_name, started_at, completion_tokens, success=True)
                    return
            _enqueue({"type": "done"})
            log_request_finish(endpoint, model_name, started_at, completion_tokens, success=True)
        except Exception as exc:
            if cancel_event.is_set():
                log_request_finish(endpoint, model_name, started_at, completion_tokens, success=True)
                return
            _record_error(model_name)
            _enqueue(
                {
                    "type": "error",
                    "payload": make_stream_error_event(f"Generation failed: {exc}"),
                }
            )
            log_request_finish(endpoint, model_name, started_at, completion_tokens, success=False)
        finally:
            end_generation()

    async def event_generator():
        if is_chat:
            yield sse_encode(
                build_chat_stream_chunk(
                    completion_id,
                    created,
                    model_name,
                    {"role": "assistant"},
                )
            )
        while True:
            if await http_request.is_disconnected():
                cancel_event.set()
                return
            try:
                event = await asyncio.to_thread(events.get, True, 0.1)
            except queue.Empty:
                continue
            event_type = event.get("type")
            if event_type == "chunk":
                yield sse_encode(event["payload"])
            elif event_type == "error":
                yield sse_encode(event["payload"])
                return
            elif event_type == "done":
                yield sse_encode("[DONE]")
                return

    begin_generation()
    threading.Thread(target=worker, daemon=True).start()
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def raise_openai_http_error(
    status_code: int,
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> None:
    raise HTTPException(
        status_code=status_code,
        detail=make_openai_error(message, error_type, param, code),
    )


def validate_stop_sequences(stop: Optional[List[str]]) -> None:
    if stop is None:
        return
    if len(stop) > MAX_STOP_SEQUENCES:
        raise_openai_http_error(
            400,
            f"Too many stop sequences: maximum is {MAX_STOP_SEQUENCES}",
            param="stop",
            code="too_many_stop_sequences",
        )
    for item in stop:
        if len(item) > MAX_STOP_SEQUENCE_LENGTH:
            raise_openai_http_error(
                400,
                f"Stop sequence exceeds maximum length of {MAX_STOP_SEQUENCE_LENGTH} characters",
                param="stop",
                code="stop_sequence_too_long",
            )


def validate_stream_requested(stream: Optional[bool]) -> None:
    if stream and not ALLOW_STREAM:
        raise_openai_http_error(
            400,
            "stream=true is not supported by this server",
            param="stream",
            code="stream_not_supported",
        )


def validate_generation_parameters(
    max_tokens: Optional[int],
    max_completion_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
) -> None:
    if max_tokens is not None and max_tokens <= 0:
        raise_openai_http_error(
            400,
            "max_tokens must be greater than 0",
            param="max_tokens",
            code="invalid_max_tokens",
        )
    if max_completion_tokens is not None and max_completion_tokens <= 0:
        raise_openai_http_error(
            400,
            "max_completion_tokens must be greater than 0",
            param="max_completion_tokens",
            code="invalid_max_completion_tokens",
        )
    if temperature is not None and temperature < 0:
        raise_openai_http_error(
            400,
            "temperature must be greater than or equal to 0",
            param="temperature",
            code="invalid_temperature",
        )
    if top_p is not None and (top_p <= 0 or top_p > 1):
        raise_openai_http_error(
            400,
            "top_p must be greater than 0 and less than or equal to 1",
            param="top_p",
            code="invalid_top_p",
        )


def validate_prompt_chars(prompt: str, param: str, model_name: str) -> None:
    max_prompt_chars = get_effective_int_limit(
        model_name,
        "max_prompt_chars",
        MAX_PROMPT_CHARS,
    )
    if max_prompt_chars > 0 and len(prompt) > max_prompt_chars:
        raise_openai_http_error(
            400,
            f"Input exceeds maximum length of {max_prompt_chars} characters",
            param=param,
            code="input_too_large",
        )


def validate_messages(messages: List[Message], model_name: str) -> None:
    if len(messages) == 0:
        raise_openai_http_error(
            400,
            "messages must contain at least one item",
            param="messages",
            code="empty_messages",
        )
    if MAX_MESSAGES_PER_REQUEST > 0 and len(messages) > MAX_MESSAGES_PER_REQUEST:
        raise_openai_http_error(
            400,
            f"Too many messages: maximum is {MAX_MESSAGES_PER_REQUEST}",
            param="messages",
            code="too_many_messages",
        )
    for msg in messages:
        # Multimodal content lists are flattened to text for the char-budget
        # check; image bytes are bounded separately by MAX_IMAGE_BYTES.
        validate_prompt_chars(_message_text(msg), "messages", model_name)


def validate_input_tokens(prompt: str, pipeline: Any, param: str, model_name: str) -> int:
    prompt_tokens = count_tokens(prompt, pipeline)
    max_input_tokens = get_effective_int_limit(
        model_name,
        "max_input_tokens",
        MAX_INPUT_TOKENS,
    )
    if max_input_tokens > 0 and prompt_tokens > max_input_tokens:
        raise_openai_http_error(
            400,
            f"Input exceeds maximum token limit of {max_input_tokens}",
            param=param,
            code="input_too_many_tokens",
        )
    return prompt_tokens


def build_validation_error_message(exc: RequestValidationError) -> str:
    errors = []
    for item in exc.errors():
        location = ".".join(str(part) for part in item.get("loc", []) if part != "body")
        message = item.get("msg", "Invalid request")
        if location:
            errors.append(f"{location}: {message}")
        else:
            errors.append(message)
    return "; ".join(errors) if errors else "Invalid request body"


def get_effective_request_limits(model_name: Optional[str]) -> Dict[str, Any]:
    if not model_name:
        return {
            "allow_stream": ALLOW_STREAM,
            "max_messages_per_request": MAX_MESSAGES_PER_REQUEST,
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_stop_sequences": MAX_STOP_SEQUENCES,
            "max_stop_sequence_length": MAX_STOP_SEQUENCE_LENGTH,
            "max_new_tokens_chat": MAX_NEW_TOKENS_CHAT,
            "max_new_tokens_completion": MAX_NEW_TOKENS_COMPLETION,
            "recommended_stream": None,
        }
    return {
        "allow_stream": ALLOW_STREAM,
        "max_messages_per_request": MAX_MESSAGES_PER_REQUEST,
        "max_prompt_chars": get_effective_int_limit(model_name, "max_prompt_chars", MAX_PROMPT_CHARS),
        "max_input_tokens": get_effective_int_limit(model_name, "max_input_tokens", MAX_INPUT_TOKENS),
        "max_stop_sequences": MAX_STOP_SEQUENCES,
        "max_stop_sequence_length": MAX_STOP_SEQUENCE_LENGTH,
        "max_new_tokens_chat": get_effective_int_limit(model_name, "max_new_tokens_chat", MAX_NEW_TOKENS_CHAT),
        "max_new_tokens_completion": get_effective_int_limit(model_name, "max_new_tokens_completion", MAX_NEW_TOKENS_COMPLETION),
        "recommended_stream": get_recommended_stream(model_name),
    }


# ============================================================================
# Idle Watchdog
# ============================================================================

_PERSISTED_COUNTERS = (
    "total_requests",
    "total_errors",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_chat_requests",
    "total_completion_requests",
    "total_stream_requests",
    "total_non_stream_requests",
    "total_auth_failures",
    "total_validation_failures",
    "total_model_loads",
    "total_model_unloads",
    "total_model_load_errors",
)
_PERSISTED_FLOATS = ("total_generation_time", "total_model_load_time")


def load_persisted_stats() -> None:
    """Seed in-memory counters from a previous run, if available."""
    global total_requests, total_errors, total_prompt_tokens, total_completion_tokens
    global total_generation_time, total_chat_requests, total_completion_requests
    global total_stream_requests, total_non_stream_requests, total_auth_failures
    global total_validation_failures, total_model_loads, total_model_unloads
    global total_model_load_errors, total_model_load_time, per_model_stats

    if not STATS_PATH or not os.path.exists(STATS_PATH):
        return
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        with stats_lock:
            for key in _PERSISTED_COUNTERS:
                value = data.get(key)
                if isinstance(value, (int, float)):
                    globals()[key] = int(value)
            for key in _PERSISTED_FLOATS:
                value = data.get(key)
                if isinstance(value, (int, float)):
                    globals()[key] = float(value)
            models = data.get("per_model")
            if isinstance(models, dict):
                for name, m in models.items():
                    if isinstance(m, dict):
                        per_model_stats[str(name)] = dict(m)
        print(f"[INFO] Loaded persisted stats from {STATS_PATH}")
    except Exception as exc:
        print(f"[WARN] Failed to load persisted stats from {STATS_PATH}: {exc}")


def save_persisted_stats() -> None:
    if not STATS_PATH:
        return
    try:
        with stats_lock:
            payload: Dict[str, Any] = {key: globals().get(key, 0) for key in _PERSISTED_COUNTERS}
            for key in _PERSISTED_FLOATS:
                payload[key] = globals().get(key, 0.0)
            payload["per_model"] = {name: dict(m) for name, m in per_model_stats.items()}
            payload["saved_at"] = int(time.time())
        os.makedirs(os.path.dirname(STATS_PATH) or ".", exist_ok=True)
        tmp_path = STATS_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, STATS_PATH)
    except Exception as exc:
        print(f"[WARN] Failed to save stats to {STATS_PATH}: {exc}")


def stats_persistence_thread() -> None:
    """Periodically snapshot stats so a hard crash still preserves recent counters."""
    while True:
        time.sleep(300)  # 5 minutes
        save_persisted_stats()


def _watchdog_sleep_interval(idle_timeout_seconds: int) -> int:
    """Pick a poll interval that re-checks ~4x per idle window.

    With a hard-coded 60s sleep an IDLE_TIMEOUT of 15s could drift to
    75s+ before a model actually unloads. Sample more often than the
    timeout itself, but cap at 30s so a 1-hour timeout doesn't burn a
    thread waking up every 15 minutes either. Floor at 1s as a safety
    rail; never sleep 0 (busy loop). For IDLE_TIMEOUT <= 0 the
    watchdog is never started, so the interval is moot.
    """
    if idle_timeout_seconds <= 0:
        return 60
    return max(1, min(30, idle_timeout_seconds // 4 or 1))


def idle_watchdog():
    """Background thread to unload model after idle timeout"""
    global current_model, current_model_name, last_request_time, active_generations

    interval = _watchdog_sleep_interval(IDLE_TIMEOUT)

    while True:
        time.sleep(interval)

        if IDLE_TIMEOUT <= 0:
            continue

        with model_lock:
            if current_model is not None:
                if active_generations > 0:
                    continue
                idle_seconds = time.time() - last_request_time
                if idle_seconds > IDLE_TIMEOUT:
                    print(f"[INFO] Idle timeout ({IDLE_TIMEOUT}s). Unloading model: {current_model_name}")
                    unload_current_model_locked("idle-timeout", sleep_for_driver=False)


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[INFO] Illama Server starting...")
    print(f"[INFO] Models directory: {MODELS_DIR}")
    print(f"[INFO] Default device: {DEFAULT_DEVICE}")
    print(f"[INFO] Idle timeout: {IDLE_TIMEOUT}s")
    print(f"[INFO] Cache dir: {CACHE_DIR or '(disabled)'}")
    print(f"[INFO] Stats path: {STATS_PATH or '(in-memory only)'}")

    load_persisted_stats()

    if IDLE_TIMEOUT > 0:
        watchdog_thread = threading.Thread(target=idle_watchdog, daemon=True)
        watchdog_thread.start()

    if STATS_PATH:
        threading.Thread(target=stats_persistence_thread, daemon=True).start()

    yield

    print("[INFO] Illama Server shutting down...")
    save_persisted_stats()


app = FastAPI(
    title="Illama Server",
    description="OpenVINO GenAI LLM Server with OpenAI-compatible API",
    version=ILLAMA_VERSION,
    lifespan=lifespan
)


@app.middleware("http")
async def optional_bearer_auth(request: Request, call_next):
    if is_authorized(request):
        return await call_next(request)
    global total_auth_failures
    with stats_lock:
        total_auth_failures += 1
    return JSONResponse(
        status_code=401,
        content={
            "error": make_openai_error(
                "Unauthorized",
                error_type="authentication_error",
                code="invalid_api_key",
            )
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.exception_handler(HTTPException)
async def openai_http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "message" in detail and "type" in detail:
        error = detail
    else:
        error_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
        error = make_openai_error(str(detail), error_type=error_type)
    return JSONResponse(status_code=exc.status_code, content={"error": error})


@app.exception_handler(RequestValidationError)
async def openai_validation_exception_handler(request: Request, exc: RequestValidationError):
    global total_validation_failures
    with stats_lock:
        total_validation_failures += 1
    return JSONResponse(
        status_code=400,
        content={
            "error": make_openai_error(
                build_validation_error_message(exc),
                error_type="invalid_request_error",
                code="invalid_request",
            )
        },
    )


@app.exception_handler(Exception)
async def openai_unhandled_exception_handler(request: Request, exc: Exception):
    print(f"[ERROR] Unhandled server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": make_openai_error(
                "The server encountered an internal error while processing the request",
                error_type="server_error",
                code="internal_error",
            )
        },
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Illama Server", "version": ILLAMA_VERSION}


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models (OpenAI-compatible)"""
    models = get_available_models()
    return ModelsResponse(
        data=[ModelInfo(id=name) for name in models]
    )


@app.post("/v1/unload")
async def unload_model():
    """Unload the current model from memory"""
    global current_model, current_model_name, current_model_is_vlm, active_generations

    with model_lock:
        if current_model is None:
            return {"status": "ok", "message": "No model loaded"}
        if active_generations > 0:
            raise_openai_http_error(
                409,
                "Cannot unload while generation is in progress",
                error_type="invalid_request_error",
                code="generation_in_progress",
            )

        model_name = current_model_name
        # Do the in-lock unload work, but ask for the driver-reclaim sleep to
        # be deferred. Holding model_lock for two seconds while idle would
        # block every concurrent request that just wanted to look up state.
        needs_sleep = unload_current_model_locked("manual-unload", sleep_for_driver=True)

    if needs_sleep:
        time.sleep(_GPU_DRIVER_RECLAIM_SECONDS)
        print(f"[INFO] Model unloaded, GPU driver reclaim wait complete (model={model_name})")

    return {"status": "ok", "message": f"Unloaded: {model_name}"}


@app.post("/v1/chat/completions")
async def chat_completions(http_request: Request, request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI-compatible)"""
    global last_request_time

    validate_stream_requested(request.stream)
    validate_messages(request.messages, request.model)
    validate_stop_sequences(request.stop)
    validate_generation_parameters(
        request.max_tokens,
        request.max_completion_tokens,
        request.temperature,
        request.top_p,
    )
    record_request_shape(is_chat=True, stream=bool(request.stream))

    # Load model
    pipeline, is_vlm = await run_in_threadpool(load_model, request.model)
    last_request_time = time.time()

    # Apply response_format=json_object by prepending a system instruction.
    effective_messages = apply_response_format(request.messages, request.response_format)

    # Decode any image_url content parts into ov.Tensor BEFORE we kick off
    # generation so a bad image fails fast with a 400 rather than during
    # the streaming response. Empty list for plain-text requests.
    if is_vlm:
        images = await run_in_threadpool(_extract_images_from_messages, effective_messages)
    else:
        images = []

    # Build prompt from messages
    prompt = await run_in_threadpool(
        build_chat_prompt, pipeline, effective_messages, request.tools
    )
    validate_prompt_chars(prompt, "messages", request.model)
    prompt_tokens = await run_in_threadpool(
        validate_input_tokens,
        prompt,
        pipeline,
        "messages",
        request.model,
    )

    # presence_penalty and frequency_penalty are accepted but not mapped —
    # OpenVINO GenAI exposes repetition_penalty, which is closer to
    # frequency_penalty in spirit. Map frequency_penalty (>0) onto it as a
    # best-effort translation: repetition_penalty = 1 + frequency_penalty.
    repetition_penalty: Optional[float] = None
    if request.frequency_penalty and request.frequency_penalty > 0:
        repetition_penalty = 1.0 + float(request.frequency_penalty)

    config = build_generation_config(
        request.model,
        "max_new_tokens_chat",
        request.max_tokens,
        request.max_completion_tokens,
        MAX_NEW_TOKENS_CHAT,
        request.temperature,
        request.top_p,
        seed=request.seed,
        repetition_penalty=repetition_penalty,
    )
    apply_stop_sequences(config, request.stop)

    # Whether to extract <think>/<tool_call> blocks from the model output.
    # Plain chat (no tools, no reasoning model) still benefits from <think>
    # extraction, so we default ON for the chat surface; clients that want
    # the raw text can disable via response_format or a future config knob.
    parse_output = True

    if request.stream:
        return create_streaming_response(
            http_request=http_request,
            endpoint="/v1/chat/completions",
            model_name=request.model,
            pipeline=pipeline,
            prompt=prompt,
            generation_config=config,
            is_chat=True,
            is_vlm=is_vlm,
            include_usage=bool(request.stream_options and request.stream_options.include_usage),
            images=images,
            parse_output=parse_output,
        )
    endpoint_started_at = log_request_start("/v1/chat/completions", request.model, stream=False)

    # Generate response
    try:
        begin_generation()
        output = await run_in_threadpool(
            generate_with_config,
            pipeline,
            prompt,
            config,
            is_vlm,
            images,
        )
        
        generation_time = time.time() - endpoint_started_at
        
        completion_tokens = await run_in_threadpool(count_tokens, str(output), pipeline)
        usage_payload = build_usage_payload(prompt_tokens, completion_tokens, generation_time)
        _record_success(request.model, prompt_tokens, completion_tokens, generation_time)
        log_request_finish("/v1/chat/completions", request.model, endpoint_started_at, completion_tokens, success=True)

        # Pull <think>/<tool_call> blocks back out of the raw model text so
        # OpenAI clients see structured reasoning_content / tool_calls fields
        # instead of having to regex over content themselves.
        parsed = parse_full_output(str(output)) if parse_output else {"content": str(output)}
        message: Dict[str, Any] = {"role": "assistant", "content": parsed["content"]}
        if "reasoning_content" in parsed:
            message["reasoning_content"] = parsed["reasoning_content"]
        finish_reason = "stop"
        if "tool_calls" in parsed:
            message["tool_calls"] = parsed["tool_calls"]
            # OpenAI semantics: when the model called a tool, content is null
            # and finish_reason becomes "tool_calls".
            if not message["content"]:
                message["content"] = None
            finish_reason = "tool_calls"

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage_payload
        }
    except Exception as e:
        _record_error(request.model)
        log_request_finish("/v1/chat/completions", request.model, endpoint_started_at, 0, success=False)
        raise_openai_http_error(
            500,
            f"Generation failed: {e}",
            error_type="server_error",
            code="generation_failed",
        )
    finally:
        end_generation()


@app.post("/v1/completions")
async def completions(http_request: Request, request: CompletionRequest):
    """Text completions endpoint (OpenAI-compatible)"""
    global last_request_time
    
    validate_stream_requested(request.stream)
    validate_stop_sequences(request.stop)
    validate_prompt_chars(request.prompt, "prompt", request.model)
    validate_generation_parameters(
        request.max_tokens,
        request.max_completion_tokens,
        request.temperature,
        request.top_p,
    )
    record_request_shape(is_chat=False, stream=bool(request.stream))

    pipeline, _is_vlm = await run_in_threadpool(load_model, request.model)
    last_request_time = time.time()

    prompt_tokens = await run_in_threadpool(
        validate_input_tokens,
        request.prompt,
        pipeline,
        "prompt",
        request.model,
    )

    repetition_penalty: Optional[float] = None
    if request.frequency_penalty and request.frequency_penalty > 0:
        repetition_penalty = 1.0 + float(request.frequency_penalty)

    config = build_generation_config(
        request.model,
        "max_new_tokens_completion",
        request.max_tokens,
        request.max_completion_tokens,
        MAX_NEW_TOKENS_COMPLETION,
        request.temperature,
        request.top_p,
        seed=request.seed,
        repetition_penalty=repetition_penalty,
    )
    apply_stop_sequences(config, request.stop)
    if request.stream:
        return create_streaming_response(
            http_request=http_request,
            endpoint="/v1/completions",
            model_name=request.model,
            pipeline=pipeline,
            prompt=request.prompt,
            generation_config=config,
            is_chat=False,
            is_vlm=False,
            include_usage=bool(request.stream_options and request.stream_options.include_usage),
        )
    endpoint_started_at = log_request_start("/v1/completions", request.model, stream=False)
    
    try:
        begin_generation()
        output = await run_in_threadpool(
            generate_with_config,
            pipeline,
            request.prompt,
            config,
            False,
        )
        generation_time = time.time() - endpoint_started_at

        completion_tokens = await run_in_threadpool(count_tokens, str(output), pipeline)
        usage_payload = build_usage_payload(prompt_tokens, completion_tokens, generation_time)
        _record_success(request.model, prompt_tokens, completion_tokens, generation_time)
        log_request_finish("/v1/completions", request.model, endpoint_started_at, completion_tokens, success=True)

        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "text": output,
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": usage_payload
        }
    except Exception as e:
        _record_error(request.model)
        log_request_finish("/v1/completions", request.model, endpoint_started_at, 0, success=False)
        raise_openai_http_error(
            500,
            f"Generation failed: {e}",
            error_type="server_error",
            code="generation_failed",
        )
    finally:
        end_generation()


@app.get("/health")
async def health():
    """Health check endpoint"""
    with model_lock:
        loaded_model = current_model_name
    return {
        "status": "healthy",
        "loaded_model": loaded_model,
        "available_models": get_available_models(),
    }


@app.get("/illama/health")
async def illama_health():
    """Extended health endpoint with runtime configuration"""
    uptime = int(time.time() - server_start_time)
    with model_lock:
        loaded_model = current_model_name
        in_flight = active_generations
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "idle_timeout_seconds": IDLE_TIMEOUT,
        "device": DEFAULT_DEVICE,
        "models_dir": MODELS_DIR,
        "loaded_model": loaded_model,
        "active_generations": in_flight,
        "auth_enabled": AUTH_ENABLED,
        "load_failure_policy": LOAD_FAILURE_POLICY,
        # "continuous" = PagedAttention backend (speculative decoding and
        # prompt-lookup available); "stateful" = SDPA backend; "none" = idle.
        "pipeline_backend": current_model_backend,
        "pipeline_backend_requested": PIPELINE_BACKEND,
        "available_models": get_available_models(),
        "request_limits": get_effective_request_limits(loaded_model),
    }


@app.get("/illama/stats")
async def illama_stats():
    """Aggregated inference statistics"""
    snap = _snapshot_inference_stats()
    with model_lock:
        current = current_model_name
        in_flight = active_generations
    request_limits = get_effective_request_limits(current)

    total_reqs = snap["total_requests"]
    total_gen_time = snap["total_generation_time"]
    avg_latency = (total_gen_time / total_reqs) if total_reqs > 0 else 0.0
    overall_tps = (
        (snap["total_completion_tokens"] / total_gen_time)
        if total_gen_time > 0
        else 0.0
    )
    # TTFT averages and decode-only TPS. TTFT is captured only on streaming
    # requests (the streamer fires per-token); non-streaming requests do not
    # contribute. tps_after_first divides completion tokens by the time spent
    # decoding AFTER the first token, so prompt-processing latency does not
    # pollute the "what's the model's actual decode speed" number.
    ttft_samples = snap.get("total_first_token_samples", 0)
    avg_ttft = (snap.get("total_first_token_time", 0.0) / ttft_samples) if ttft_samples > 0 else 0.0
    decode_time = max(total_gen_time - snap.get("total_first_token_time", 0.0), 0.0)
    overall_tps_after_first = (
        (snap["total_completion_tokens"] / decode_time) if decode_time > 0 else 0.0
    )

    models: Dict[str, Any] = {}
    for name, m in snap["per_model"].items():
        reqs = m.get("requests", 0)
        gen_time = m.get("generation_time", 0.0)
        completion_toks = m.get("completion_tokens", 0)
        per_ttft_samples = m.get("first_token_samples", 0)
        per_ttft_total = m.get("first_token_time", 0.0)
        per_decode_time = max(gen_time - per_ttft_total, 0.0)
        models[name] = {
            "requests": reqs,
            "errors": m.get("errors", 0),
            "loads": m.get("loads", 0),
            "load_errors": m.get("load_errors", 0),
            "unloads": m.get("unloads", 0),
            "prompt_tokens": m.get("prompt_tokens", 0),
            "completion_tokens": completion_toks,
            "load_time_seconds": round(m.get("load_time", 0.0), 2),
            "generation_time_seconds": round(gen_time, 2),
            "avg_latency_seconds": round(gen_time / reqs, 3) if reqs > 0 else 0.0,
            "tokens_per_second": round(completion_toks / gen_time, 1)
            if gen_time > 0
            else 0.0,
            "avg_ttft_seconds": round(per_ttft_total / per_ttft_samples, 3)
            if per_ttft_samples > 0
            else 0.0,
            "ttft_samples": per_ttft_samples,
            "tokens_per_second_after_first": round(completion_toks / per_decode_time, 1)
            if per_decode_time > 0
            else 0.0,
        }

    return {
        "uptime_seconds": int(snap["uptime"]),
        "idle_timeout_seconds": IDLE_TIMEOUT,
        "device": DEFAULT_DEVICE,
        "models_dir": MODELS_DIR,
        "current_model": current,
        "active_generations": in_flight,
        "auth_enabled": AUTH_ENABLED,
        "load_failure_policy": LOAD_FAILURE_POLICY,
        "pipeline_backend": current_model_backend,
        "request_limits": request_limits,
        "total_requests": total_reqs,
        "total_errors": snap["total_errors"],
        "total_chat_requests": snap["total_chat_requests"],
        "total_completion_requests": snap["total_completion_requests"],
        "total_stream_requests": snap["total_stream_requests"],
        "total_non_stream_requests": snap["total_non_stream_requests"],
        "total_auth_failures": snap["total_auth_failures"],
        "total_validation_failures": snap["total_validation_failures"],
        "total_model_loads": snap["total_model_loads"],
        "total_model_unloads": snap["total_model_unloads"],
        "total_model_load_errors": snap["total_model_load_errors"],
        "total_model_load_time_seconds": round(snap["total_model_load_time"], 2),
        "total_prompt_tokens": snap["total_prompt_tokens"],
        "total_completion_tokens": snap["total_completion_tokens"],
        "total_generation_time_seconds": round(total_gen_time, 2),
        "avg_latency_seconds": round(avg_latency, 3),
        "overall_tokens_per_second": round(overall_tps, 1),
        "avg_ttft_seconds": round(avg_ttft, 3),
        "ttft_samples": ttft_samples,
        "overall_tokens_per_second_after_first": round(overall_tps_after_first, 1),
        "per_model": models,
    }


@app.get("/metrics")
async def metrics():
    snap = _snapshot_inference_stats()
    with model_lock:
        current = current_model_name
        in_flight = active_generations
    loaded_model = current or ""

    lines = [
        "# HELP illama_uptime_seconds Process uptime in seconds",
        "# TYPE illama_uptime_seconds gauge",
        f"illama_uptime_seconds {int(snap['uptime'])}",
        "# HELP illama_active_generations Currently active generation requests",
        "# TYPE illama_active_generations gauge",
        f"illama_active_generations {in_flight}",
        "# HELP illama_total_requests Total successful generation requests",
        "# TYPE illama_total_requests counter",
        f"illama_total_requests {snap['total_requests']}",
        "# HELP illama_total_errors Total generation errors",
        "# TYPE illama_total_errors counter",
        f"illama_total_errors {snap['total_errors']}",
        "# HELP illama_total_chat_requests Total chat-completion requests",
        "# TYPE illama_total_chat_requests counter",
        f"illama_total_chat_requests {snap['total_chat_requests']}",
        "# HELP illama_total_completion_requests Total text-completion requests",
        "# TYPE illama_total_completion_requests counter",
        f"illama_total_completion_requests {snap['total_completion_requests']}",
        "# HELP illama_total_stream_requests Total streaming requests",
        "# TYPE illama_total_stream_requests counter",
        f"illama_total_stream_requests {snap['total_stream_requests']}",
        "# HELP illama_total_non_stream_requests Total non-streaming requests",
        "# TYPE illama_total_non_stream_requests counter",
        f"illama_total_non_stream_requests {snap['total_non_stream_requests']}",
        "# HELP illama_total_auth_failures Total authentication failures",
        "# TYPE illama_total_auth_failures counter",
        f"illama_total_auth_failures {snap['total_auth_failures']}",
        "# HELP illama_total_validation_failures Total validation failures",
        "# TYPE illama_total_validation_failures counter",
        f"illama_total_validation_failures {snap['total_validation_failures']}",
        "# HELP illama_total_model_loads Total successful model loads",
        "# TYPE illama_total_model_loads counter",
        f"illama_total_model_loads {snap['total_model_loads']}",
        "# HELP illama_total_model_unloads Total model unloads",
        "# TYPE illama_total_model_unloads counter",
        f"illama_total_model_unloads {snap['total_model_unloads']}",
        "# HELP illama_total_model_load_errors Total model load errors",
        "# TYPE illama_total_model_load_errors counter",
        f"illama_total_model_load_errors {snap['total_model_load_errors']}",
        "# HELP illama_total_model_load_time_seconds Total model load time",
        "# TYPE illama_total_model_load_time_seconds counter",
        f"illama_total_model_load_time_seconds {snap['total_model_load_time']:.6f}",
        "# HELP illama_total_prompt_tokens Total prompt tokens",
        "# TYPE illama_total_prompt_tokens counter",
        f"illama_total_prompt_tokens {snap['total_prompt_tokens']}",
        "# HELP illama_total_completion_tokens Total completion tokens",
        "# TYPE illama_total_completion_tokens counter",
        f"illama_total_completion_tokens {snap['total_completion_tokens']}",
        "# HELP illama_total_generation_time_seconds Total generation time",
        "# TYPE illama_total_generation_time_seconds counter",
        f"illama_total_generation_time_seconds {snap['total_generation_time']:.6f}",
        "# HELP illama_total_first_token_time_seconds Cumulative time-to-first-token across streamed requests",
        "# TYPE illama_total_first_token_time_seconds counter",
        f"illama_total_first_token_time_seconds {snap.get('total_first_token_time', 0.0):.6f}",
        "# HELP illama_total_first_token_samples Streamed requests that contributed a TTFT measurement",
        "# TYPE illama_total_first_token_samples counter",
        f"illama_total_first_token_samples {snap.get('total_first_token_samples', 0)}",
        "# HELP illama_loaded_model_info Loaded model info",
        "# TYPE illama_loaded_model_info gauge",
        f'illama_loaded_model_info{{model="{loaded_model}"}} {1 if loaded_model else 0}',
    ]

    per_model = snap["per_model"]
    total_models = len(per_model)
    if total_models > MAX_METRICS_PER_MODEL_LABELS:
        items = sorted(
            per_model.items(),
            key=lambda kv: kv[1].get("last_activity_ts", 0.0),
            reverse=True,
        )[:MAX_METRICS_PER_MODEL_LABELS]
    else:
        items = list(per_model.items())

    lines.extend(
        [
            "# HELP illama_metrics_models_total Number of models with stats records",
            "# TYPE illama_metrics_models_total gauge",
            f"illama_metrics_models_total {total_models}",
            "# HELP illama_metrics_models_emitted Per-model rows actually emitted (capped by MAX_METRICS_PER_MODEL_LABELS)",
            "# TYPE illama_metrics_models_emitted gauge",
            f"illama_metrics_models_emitted {len(items)}",
        ]
    )

    for name, m in items:
        safe_name = (
            name.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
        )
        lines.append(
            f'illama_model_requests_total{{model="{safe_name}"}} {m.get("requests", 0)}'
        )
        lines.append(
            f'illama_model_errors_total{{model="{safe_name}"}} {m.get("errors", 0)}'
        )
        lines.append(
            f'illama_model_loads_total{{model="{safe_name}"}} {m.get("loads", 0)}'
        )
        lines.append(
            f'illama_model_unloads_total{{model="{safe_name}"}} {m.get("unloads", 0)}'
        )
        lines.append(
            f'illama_model_first_token_time_seconds_total{{model="{safe_name}"}} '
            f'{m.get("first_token_time", 0.0):.6f}'
        )
        lines.append(
            f'illama_model_first_token_samples_total{{model="{safe_name}"}} '
            f'{m.get("first_token_samples", 0)}'
        )
    return PlainTextResponse("\n".join(lines) + "\n")


# ============================================================================
# Ollama-compatible API
# ============================================================================
#
# Implements the subset of https://github.com/ollama/ollama/blob/main/docs/api.md
# that Ollama-targeting clients (Continue.dev, LangChain ChatOllama, OpenWebUI's
# Ollama backend, etc.) actually depend on:
#   GET  /api/version
#   GET  /api/tags
#   GET  /api/ps
#   POST /api/show
#   POST /api/chat
#   POST /api/generate
#
# Streaming uses NDJSON (one JSON object per line, no `data:` prefix), not SSE.
# Timing fields are reported in nanoseconds, matching Ollama's response shape.
#
# ILLAMA_VERSION is defined once at the top of this module (read from env / VERSION
# file / fallback) and is reused in the /api/version response below.


class OllamaOptions(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    stop: Optional[List[str]] = None
    repeat_penalty: Optional[float] = None


class OllamaChatMessage(BaseModel):
    role: str
    content: str
    # Ollama's multimodal convention: list of base64-encoded image strings
    # alongside the text content. Optional; absent for text-only models.
    images: Optional[List[str]] = None


class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    stream: Optional[bool] = True  # Ollama default
    options: Optional[OllamaOptions] = None
    format: Optional[str] = None
    keep_alive: Optional[str] = None
    # Ollama tool calling. Same schema as OpenAI's ``tools`` field; the
    # tokenizer's chat template renders it so the model sees a tool list,
    # and we parse <tool_call>...</tool_call> back out of the response.
    tools: Optional[List[Dict[str, Any]]] = None


class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    # Same convention as OllamaChatMessage.images, but on the generate request
    # itself since /api/generate is single-prompt.
    images: Optional[List[str]] = None
    options: Optional[OllamaOptions] = None
    format: Optional[str] = None
    keep_alive: Optional[str] = None


class OllamaShowRequest(BaseModel):
    name: Optional[str] = None
    model: Optional[str] = None
    verbose: Optional[bool] = False


def _ollama_timestamp(unix_seconds: Optional[float] = None) -> str:
    """Format a unix timestamp (or now()) in Ollama's RFC3339-with-millis form."""
    seconds = time.time() if unix_seconds is None else max(0.0, unix_seconds)
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(seconds))


# Sentinel returned by /api/ps when IDLE_TIMEOUT == 0 (model is pinned to
# VRAM forever). Mirrors what stock Ollama emits when keep_alive=-1, so
# `ollama ps`-style clients render it as "Forever" instead of a stale time.
_OLLAMA_NEVER_EXPIRES = "9999-12-31T23:59:59.000Z"


def _seconds_to_ns(seconds: float) -> int:
    return int(max(seconds, 0.0) * 1_000_000_000)


def _model_dir_size(path: str) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat(follow_symlinks=False).st_size
            elif entry.is_dir(follow_symlinks=False):
                total += _model_dir_size(entry.path)
    except OSError:
        pass
    return total


def _read_arch(model_path: str) -> Optional[str]:
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures") or []
        if isinstance(archs, list) and archs:
            return str(archs[0])
    except Exception:
        return None
    return None


def _ollama_model_descriptor(name: str) -> Dict[str, Any]:
    model_path = os.path.join(MODELS_DIR, name)
    metadata = load_model_metadata(name)
    arch = _read_arch(model_path) or "unknown"

    # Pick a representative file for modified_at / digest seed
    primary = None
    for candidate in ("openvino_model.xml", "openvino_language_model.xml"):
        p = os.path.join(model_path, candidate)
        if os.path.exists(p):
            primary = p
            break

    if primary:
        mtime = os.path.getmtime(primary)
    elif os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
    else:
        mtime = time.time()

    modified_at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(mtime))
    size = _model_dir_size(model_path)

    return {
        "name": name,
        "model": name,
        "modified_at": modified_at,
        "size": size,
        "digest": metadata.get("digest", ""),
        "details": {
            "parent_model": "",
            "format": "openvino",
            "family": arch.lower(),
            "families": [arch.lower()] if arch != "unknown" else None,
            "parameter_size": metadata.get("parameter_size", "unknown"),
            "quantization_level": metadata.get("quantization", "INT4"),
        },
    }


def _ollama_options_to_kwargs(
    options: Optional[OllamaOptions],
) -> Dict[str, Any]:
    if options is None:
        return {}
    return {
        "temperature": options.temperature,
        "top_p": options.top_p,
        "seed": options.seed,
        "num_predict": options.num_predict,
        "stop": options.stop,
        "repeat_penalty": options.repeat_penalty,
    }


def _build_config_for_ollama(
    model_name: str,
    override_key: str,
    configured_default: int,
    options: Optional[OllamaOptions],
) -> ov_genai.GenerationConfig:
    opts = _ollama_options_to_kwargs(options)
    return build_generation_config(
        model_name,
        override_key,
        opts.get("num_predict"),
        None,
        configured_default,
        opts.get("temperature") if opts.get("temperature") is not None else 0.7,
        opts.get("top_p"),
        seed=opts.get("seed"),
        repetition_penalty=opts.get("repeat_penalty"),
    )


def _ndjson(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def _validate_ollama_format(format_value: Optional[str]) -> Optional[Dict[str, Any]]:
    if not format_value:
        return None
    if format_value.lower() == "json":
        return {"type": "json_object"}
    return None


@app.get("/api/version")
async def ollama_version():
    return {"version": ILLAMA_VERSION}


@app.get("/api/tags")
async def ollama_tags():
    return {"models": [_ollama_model_descriptor(name) for name in get_available_models()]}


@app.get("/api/ps")
async def ollama_ps():
    if current_model_name is None:
        return {"models": []}
    descriptor = _ollama_model_descriptor(current_model_name)
    # `expires_at` is documented as "when the model will be unloaded"; for
    # the idle-timeout watchdog that's `last_request_time + IDLE_TIMEOUT`.
    # When IDLE_TIMEOUT is 0 the model never auto-unloads, so emit the
    # far-future sentinel that stock Ollama uses for keep_alive=-1.
    if IDLE_TIMEOUT > 0:
        descriptor["expires_at"] = _ollama_timestamp(last_request_time + IDLE_TIMEOUT)
    else:
        descriptor["expires_at"] = _OLLAMA_NEVER_EXPIRES
    descriptor["size_vram"] = descriptor["size"]
    return {"models": [descriptor]}


@app.post("/api/show")
async def ollama_show(request: OllamaShowRequest):
    name = request.name or request.model
    if not name:
        raise_openai_http_error(
            400,
            "name is required",
            param="name",
            code="missing_name",
        )
    if name not in get_available_models():
        raise_openai_http_error(
            404,
            f"model '{name}' not found",
            error_type="invalid_request_error",
            code="model_not_found",
        )
    desc = _ollama_model_descriptor(name)
    return {
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": desc["details"],
        "model_info": {
            "general.architecture": desc["details"]["family"],
        },
        "modified_at": desc["modified_at"],
    }


def _build_ollama_chat_response(
    *,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_seconds: float,
    load_seconds: float,
    eval_seconds: float,
    done: bool,
    role: str = "assistant",
    thinking: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": role, "content": content}
    if thinking:
        # Ollama's official key for chain-of-thought is "thinking"; we surface
        # it whenever the model produced a <think> block so OpenWebUI in
        # Ollama mode renders the reasoning panel.
        message["thinking"] = thinking
    if tool_calls:
        # Ollama tool_calls drop the OpenAI ``id``/``type``/``index`` envelope
        # and use ``arguments`` as a parsed object rather than a JSON string.
        message["tool_calls"] = [to_ollama_tool_call(tc) for tc in tool_calls]
    payload: Dict[str, Any] = {
        "model": model,
        "created_at": _ollama_timestamp(),
        "message": message,
        "done": done,
    }
    if done:
        payload["done_reason"] = "stop"
        payload["total_duration"] = _seconds_to_ns(total_seconds)
        payload["load_duration"] = _seconds_to_ns(load_seconds)
        payload["prompt_eval_count"] = prompt_tokens
        payload["prompt_eval_duration"] = _seconds_to_ns(max(total_seconds - eval_seconds, 0.0))
        payload["eval_count"] = completion_tokens
        payload["eval_duration"] = _seconds_to_ns(eval_seconds)
    return payload


def _build_ollama_generate_response(
    *,
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_seconds: float,
    load_seconds: float,
    eval_seconds: float,
    done: bool,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "created_at": _ollama_timestamp(),
        "response": text,
        "done": done,
    }
    if done:
        payload["done_reason"] = "stop"
        payload["context"] = []
        payload["total_duration"] = _seconds_to_ns(total_seconds)
        payload["load_duration"] = _seconds_to_ns(load_seconds)
        payload["prompt_eval_count"] = prompt_tokens
        payload["prompt_eval_duration"] = _seconds_to_ns(max(total_seconds - eval_seconds, 0.0))
        payload["eval_count"] = completion_tokens
        payload["eval_duration"] = _seconds_to_ns(eval_seconds)
    return payload


def _create_ollama_streaming_response(
    *,
    http_request: Request,
    model_name: str,
    pipeline: Any,
    prompt: str,
    generation_config: ov_genai.GenerationConfig,
    is_vlm: bool,
    is_chat: bool,
    load_seconds: float,
    images: Optional[List[ov.Tensor]] = None,
    parse_output: bool = True,
) -> StreamingResponse:
    started_at = time.time()
    # See create_streaming_response for the cancel-aware enqueue contract.
    events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)
    streamed_parts: List[str] = []
    cancel_event = threading.Event()
    # Captured the first time the streamer fires with non-empty text. Used
    # to record TTFT separately from total eval time on the Ollama path.
    first_token_at: List[float] = []
    parser: Optional[StreamingParser] = StreamingParser() if (parse_output and is_chat) else None
    accumulated_tool_calls: List[Dict[str, Any]] = []

    def _enqueue(event: Dict[str, Any]) -> bool:
        if cancel_event.is_set():
            return False
        deadline = time.monotonic() + _STREAM_ENQUEUE_HARD_DEADLINE_SECONDS
        while True:
            try:
                events.put(event, timeout=_STREAM_ENQUEUE_POLL_SECONDS)
                return True
            except queue.Full:
                if cancel_event.is_set() or time.monotonic() > deadline:
                    return False

    def _flush_parser_events_ollama(events_list: List[Tuple[str, Any]]) -> bool:
        for kind, payload in events_list:
            if kind == "content":
                if not _enqueue({"type": "chunk", "payload": {"content": payload}}):
                    return False
            elif kind == "reasoning":
                # Ollama uses ``thinking`` rather than ``reasoning_content``.
                # We surface incremental thinking deltas the same way Ollama
                # does so OpenWebUI's reasoning panel updates live.
                if not _enqueue({"type": "chunk", "payload": {"thinking": payload}}):
                    return False
            elif kind == "tool":
                # Ollama's chat stream typically only includes tool_calls in
                # the final ``done: true`` chunk. Buffer them here and emit
                # them at finalize time.
                accumulated_tool_calls.append(payload)
        return True

    def emit_chunk(text: str) -> bool:
        if is_chat and parser is not None:
            return _flush_parser_events_ollama(parser.feed(text))
        if is_chat:
            return _enqueue({"type": "chunk", "payload": {"content": text}})
        return _enqueue({"type": "chunk", "payload": {"text": text}})

    def streamer(token: Any) -> Any:
        if cancel_event.is_set():
            return _stream_stop_status()
        text = str(token)
        if text:
            if not first_token_at:
                first_token_at.append(time.time())
            touch_last_request_time()
            streamed_parts.append(text)
            if not emit_chunk(text):
                return _stream_stop_status()
        return ov_genai.StreamingStatus.RUNNING

    def worker() -> None:
        try:
            if is_vlm:
                result = _vlm_generate(
                    pipeline,
                    prompt,
                    generation_config,
                    images,
                    streamer=streamer,
                )
            else:
                result = pipeline.generate(prompt, generation_config, streamer=streamer)

            output = "".join(streamed_parts).strip("\x00")
            if not output and result is not None:
                output = str(result)

            eval_seconds = time.time() - started_at
            total_seconds = eval_seconds + load_seconds
            ttft = (first_token_at[0] - started_at) if first_token_at else None
            prompt_tokens = count_tokens(prompt, pipeline)
            completion_tokens = count_tokens(output, pipeline)
            _record_success(
                model_name,
                prompt_tokens,
                completion_tokens,
                eval_seconds,
                first_token_time=ttft,
            )

            if is_chat:
                # Flush any held-back parser bytes before sealing the stream.
                final_thinking: Optional[str] = None
                if parser is not None:
                    if not _flush_parser_events_ollama(parser.finalize()):
                        return
                    # Whole-block thinking aggregation for the final chunk:
                    # Ollama's spec puts the COMPLETE thinking string on the
                    # done=True chunk too, in addition to the streamed deltas,
                    # so clients that didn't catch the incremental stream can
                    # still see it. We rebuild it from the raw output.
                    rebuilt = parse_full_output(output)
                    final_thinking = rebuilt.get("reasoning_content")

                final = _build_ollama_chat_response(
                    model=model_name,
                    content="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_seconds=total_seconds,
                    load_seconds=load_seconds,
                    eval_seconds=eval_seconds,
                    done=True,
                    thinking=final_thinking,
                    tool_calls=accumulated_tool_calls or None,
                )
            else:
                final = _build_ollama_generate_response(
                    model=model_name,
                    text="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_seconds=total_seconds,
                    load_seconds=load_seconds,
                    eval_seconds=eval_seconds,
                    done=True,
                )
            if not _enqueue({"type": "final", "payload": final}):
                return
            _enqueue({"type": "done"})
        except Exception as exc:
            if cancel_event.is_set():
                return
            _record_error(model_name)
            _enqueue({"type": "error", "payload": {"error": f"Generation failed: {exc}"}})
        finally:
            end_generation()

    async def event_generator():
        while True:
            if await http_request.is_disconnected():
                cancel_event.set()
                return
            try:
                event = await asyncio.to_thread(events.get, True, 0.1)
            except queue.Empty:
                continue
            event_type = event.get("type")
            if event_type == "chunk":
                if is_chat:
                    yield _ndjson(
                        {
                            "model": model_name,
                            "created_at": _ollama_timestamp(),
                            "message": {"role": "assistant", "content": event["payload"]["content"]},
                            "done": False,
                        }
                    )
                else:
                    yield _ndjson(
                        {
                            "model": model_name,
                            "created_at": _ollama_timestamp(),
                            "response": event["payload"]["text"],
                            "done": False,
                        }
                    )
            elif event_type == "final":
                yield _ndjson(event["payload"])
            elif event_type == "error":
                yield _ndjson(event["payload"])
                return
            elif event_type == "done":
                return

    begin_generation()
    threading.Thread(target=worker, daemon=True).start()
    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def ollama_chat(http_request: Request, request: OllamaChatRequest):
    # Ollama clients default to streaming (request.stream defaults to True).
    # Honor the global allow_stream config so disabling it actually disables
    # streaming for both API surfaces, not just /v1/*.
    stream_requested = request.stream is None or request.stream
    validate_stream_requested(stream_requested)
    record_request_shape(is_chat=True, stream=bool(request.stream))

    load_started_at = time.time()
    pipeline, is_vlm = await run_in_threadpool(load_model, request.model)
    load_seconds = time.time() - load_started_at

    response_format = _validate_ollama_format(request.format)
    base_messages = [Message(role=m.role, content=m.content) for m in request.messages]
    effective_messages = apply_response_format(base_messages, response_format)
    validate_messages(effective_messages, request.model)

    # Decode any Ollama-style 'images' fields (base64 strings on the original
    # messages) into ov.Tensor BEFORE generation so a bad image fails fast
    # with a 400. Empty list when the model isn't a VLM.
    if is_vlm:
        images = await run_in_threadpool(_extract_ollama_chat_images, request.messages)
    else:
        images = []

    prompt = await run_in_threadpool(
        build_chat_prompt, pipeline, effective_messages, request.tools
    )
    validate_prompt_chars(prompt, "messages", request.model)
    prompt_tokens = await run_in_threadpool(
        validate_input_tokens,
        prompt,
        pipeline,
        "messages",
        request.model,
    )

    config = _build_config_for_ollama(
        request.model,
        "max_new_tokens_chat",
        MAX_NEW_TOKENS_CHAT,
        request.options,
    )
    apply_stop_sequences(config, request.options.stop if request.options else None)

    if request.stream is None or request.stream:
        return _create_ollama_streaming_response(
            http_request=http_request,
            model_name=request.model,
            pipeline=pipeline,
            prompt=prompt,
            generation_config=config,
            is_vlm=is_vlm,
            is_chat=True,
            load_seconds=load_seconds,
            images=images,
        )

    started_at = time.time()
    try:
        begin_generation()
        output = await run_in_threadpool(
            generate_with_config,
            pipeline,
            prompt,
            config,
            is_vlm,
            images,
        )
        eval_seconds = time.time() - started_at
        completion_tokens = await run_in_threadpool(count_tokens, str(output), pipeline)
        _record_success(request.model, prompt_tokens, completion_tokens, eval_seconds)
        # Pull <think>/<tool_call> blocks out for Ollama-shaped fields.
        parsed = parse_full_output(str(output))
        return _build_ollama_chat_response(
            model=request.model,
            content=parsed["content"],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_seconds=eval_seconds + load_seconds,
            load_seconds=load_seconds,
            eval_seconds=eval_seconds,
            done=True,
            thinking=parsed.get("reasoning_content"),
            tool_calls=parsed.get("tool_calls"),
        )
    except Exception as exc:
        _record_error(request.model)
        raise_openai_http_error(
            500,
            f"Generation failed: {exc}",
            error_type="server_error",
            code="generation_failed",
        )
    finally:
        end_generation()


@app.post("/api/generate")
async def ollama_generate(http_request: Request, request: OllamaGenerateRequest):
    stream_requested = request.stream is None or request.stream
    validate_stream_requested(stream_requested)
    record_request_shape(is_chat=False, stream=bool(request.stream))

    load_started_at = time.time()
    pipeline, is_vlm = await run_in_threadpool(load_model, request.model)
    load_seconds = time.time() - load_started_at

    # Compose prompt: optional system prefix, then prompt body. `template`,
    # `context`, `suffix` and `raw` are accepted but not deeply supported.
    prompt = request.prompt
    if request.system:
        if request.raw:
            prompt = request.system + "\n" + prompt
        else:
            try:
                tokenizer = pipeline.get_tokenizer()
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": request.system},
                            {"role": "user", "content": request.prompt},
                        ],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
            except Exception:
                prompt = f"{request.system}\n{request.prompt}"

    # Ollama's /api/generate carries images on the request itself rather than
    # per-message. Decode them up front so a bad image fails fast.
    if is_vlm and request.images:
        images = await run_in_threadpool(
            lambda srcs: [_decode_image_to_tensor(s) for s in srcs if s],
            request.images,
        )
    else:
        images = []

    validate_prompt_chars(prompt, "prompt", request.model)
    prompt_tokens = await run_in_threadpool(
        validate_input_tokens,
        prompt,
        pipeline,
        "prompt",
        request.model,
    )

    config = _build_config_for_ollama(
        request.model,
        "max_new_tokens_completion",
        MAX_NEW_TOKENS_COMPLETION,
        request.options,
    )
    apply_stop_sequences(config, request.options.stop if request.options else None)

    if request.stream is None or request.stream:
        return _create_ollama_streaming_response(
            http_request=http_request,
            model_name=request.model,
            pipeline=pipeline,
            prompt=prompt,
            generation_config=config,
            is_vlm=is_vlm,
            is_chat=False,
            load_seconds=load_seconds,
            images=images,
        )

    started_at = time.time()
    try:
        begin_generation()
        output = await run_in_threadpool(
            generate_with_config,
            pipeline,
            prompt,
            config,
            is_vlm,
            images,
        )
        eval_seconds = time.time() - started_at
        completion_tokens = await run_in_threadpool(count_tokens, str(output), pipeline)
        _record_success(request.model, prompt_tokens, completion_tokens, eval_seconds)
        return _build_ollama_generate_response(
            model=request.model,
            text=str(output),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_seconds=eval_seconds + load_seconds,
            load_seconds=load_seconds,
            eval_seconds=eval_seconds,
            done=True,
        )
    except Exception as exc:
        _record_error(request.model)
        raise_openai_http_error(
            500,
            f"Generation failed: {exc}",
            error_type="server_error",
            code="generation_failed",
        )
    finally:
        end_generation()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("ILLAMA_PORT", "11434"))
    uvicorn.run(app, host="0.0.0.0", port=port)
