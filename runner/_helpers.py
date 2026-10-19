"""Pure-Python helpers for the runner.

This module collects functions that have no dependency on the runner's
mutable global state (``current_pipeline``, ``model_lock``, ``CONFIG``,
``stats_lock``, ...). Keeping them separate gives us:

  - A clean unit-test surface (no need to spin up the full server module).
  - A smaller server.py that focuses on the orchestration / lifecycle
    concerns (model loading, streaming, FastAPI plumbing).
  - A single import point for things like the data-URL regex and the
    OpenAI error envelope shape, which are used in multiple endpoints.

server.py re-imports every name in __all__ so that ``server._b64decode_lenient``
and friends still work for existing callers and tests.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _parse_idle_timeout(config: Dict[str, Any], default_seconds: int = 60) -> int:
    """Parse the idle-timeout value from config or environment.

    Supports plain seconds ("600"), seconds with unit ("600s"), and minutes
    with unit ("10m"). Falls back to ``default_seconds`` on invalid values.
    """
    raw = None
    if "idle_timeout" in config:
        raw = str(config["idle_timeout"])
    if raw is None:
        raw = (
            os.environ.get("ILLAMA_IDLE_TIMEOUT")
            or os.environ.get("ILLAMA_IDLE_TIMEOUT_SEC")
        )
    if raw is None:
        raw = str(default_seconds)
    try:
        value = raw.strip().lower()
        if value.endswith("m"):
            return int(float(value[:-1]) * 60)
        if value.endswith("s"):
            return int(float(value[:-1]))
        return int(float(value))
    except Exception:
        print(f"[WARN] Invalid ILLAMA_IDLE_TIMEOUT value '{raw}', using default {default_seconds}s")
        return default_seconds


def _get_config_int(config: Dict[str, Any], key: str, default: int) -> int:
    """Read an integer config value with a safe fallback."""
    raw = config.get(key, default)
    try:
        return int(raw)
    except Exception:
        print(f"[WARN] Invalid integer config for '{key}': {raw!r}, using {default}")
        return default


def _get_config_bool(config: Dict[str, Any], key: str, default: bool) -> bool:
    """Read a boolean config value with a safe fallback."""
    raw = config.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    if isinstance(raw, (int, float)):
        return bool(raw)
    print(f"[WARN] Invalid boolean config for '{key}': {raw!r}, using {default}")
    return default


def normalize_load_failure_policy(value: str) -> str:
    policy = value.strip().lower()
    if policy in {"restart", "return_error"}:
        return policy
    print(f"[WARN] Invalid load_failure_policy '{value}', defaulting to restart")
    return "restart"


# ---------------------------------------------------------------------------
# Image / data-URL helpers
# ---------------------------------------------------------------------------


# Data-URL pattern used by both OpenAI multimodal content and Ollama's
# ``images: ["..."]`` field. ``re.DOTALL`` lets the payload span newlines.
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[^;]+)?;base64,(?P<payload>.+)$",
    re.DOTALL,
)


def _b64decode_lenient(payload: str) -> bytes:
    """Decode a base64 payload, tolerating embedded whitespace.

    ``base64 < image.png`` wraps output at 76 chars by default and many
    clients copy-paste with newlines, tabs, or stray spaces. The standard
    library's ``base64.b64decode(..., validate=True)`` rejects those, so we
    strip ASCII whitespace first. Still raises ``binascii.Error`` for
    actually-invalid base64.
    """
    cleaned = "".join(payload.split())
    return base64.b64decode(cleaned, validate=True)


# ---------------------------------------------------------------------------
# Auth exemption
# ---------------------------------------------------------------------------


# Endpoints that never require an API key, even when ``ILLAMA_API_KEY`` is
# set. ``/api/version`` is what Ollama clients hit first to test connectivity.
_AUTH_EXEMPT_PATHS = frozenset({
    "/",
    "/health",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/api/version",
})


def is_auth_exempt_path(path: str) -> bool:
    return path in _AUTH_EXEMPT_PATHS


# ---------------------------------------------------------------------------
# OpenAI-shaped responses
# ---------------------------------------------------------------------------


def make_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "message": message,
        "type": error_type,
        "param": param,
        "code": code,
    }


def make_stream_error_event(message: str, code: str = "generation_failed") -> Dict[str, Any]:
    return {
        "error": make_openai_error(
            message,
            error_type="server_error",
            code=code,
        )
    }


def sse_encode(payload: Any) -> str:
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_chat_stream_chunk(
    completion_id: str,
    created: int,
    model_name: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def build_completion_stream_chunk(
    completion_id: str,
    created: int,
    model_name: str,
    text: str,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": finish_reason,
            }
        ],
    }


def build_usage_payload(
    prompt_tokens: int, completion_tokens: int, generation_time: float
) -> Dict[str, Any]:
    total_tokens = prompt_tokens + completion_tokens
    tps = completion_tokens / generation_time if generation_time > 0 else 0.0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # Non-standard fields - OpenAI ignores unknown keys, but Ollama
        # consumers and our own dashboards rely on them.
        "generation_time_seconds": round(generation_time, 2),
        "tokens_per_second": round(tps, 1),
    }


__all__ = [
    "_parse_idle_timeout",
    "_get_config_int",
    "_get_config_bool",
    "normalize_load_failure_policy",
    "_DATA_URL_RE",
    "_b64decode_lenient",
    "_AUTH_EXEMPT_PATHS",
    "is_auth_exempt_path",
    "make_openai_error",
    "make_stream_error_event",
    "sse_encode",
    "build_chat_stream_chunk",
    "build_completion_stream_chunk",
    "build_usage_payload",
]
