#!/usr/bin/env python3
"""Benchmark Illama's OpenAI-compatible API with stdlib only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SHORT_PROMPT = "Reply with exactly: hello"
LONG_PROMPT = "Count from 1 to 120, one number per line, and stop at 120."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Illama/OpenAI-compatible endpoints")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Base OpenAI API URL")
    parser.add_argument("--api-key", default="", help="Optional bearer token")
    parser.add_argument("--models", nargs="*", default=[], help="Model names to benchmark; defaults to /v1/models")
    parser.add_argument("--cold-start", action="store_true", help="Unload before each test")
    parser.add_argument("--short-max-tokens", type=int, default=32, help="Max completion tokens for short test")
    parser.add_argument("--long-max-tokens", type=int, default=256, help="Max completion tokens for long tests")
    parser.add_argument("--short-prompt", default=SHORT_PROMPT, help="Prompt for short-latency test")
    parser.add_argument("--long-prompt", default=LONG_PROMPT, help="Prompt for long-throughput tests")
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    parser.add_argument("--markdown", default="", help="Optional Markdown output path")
    return parser.parse_args()


def build_request(url: str, payload: Optional[Dict[str, Any]] = None, api_key: str = "") -> urllib.request.Request:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(url, data=data, headers=headers)


def http_json(url: str, payload: Optional[Dict[str, Any]] = None, api_key: str = "") -> Dict[str, Any]:
    request = build_request(url, payload=payload, api_key=api_key)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc


def fetch_models(base_url: str, api_key: str) -> List[str]:
    payload = http_json(f"{base_url}/models", api_key=api_key)
    return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and "id" in item]


def unload(base_url: str, api_key: str) -> None:
    try:
        http_json(f"{base_url}/unload", payload={}, api_key=api_key)
    except Exception as exc:
        print(f"[WARN] unload failed: {exc}", file=sys.stderr)


def nonstream_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    result = http_json(f"{base_url}/chat/completions", payload=payload, api_key=api_key)
    elapsed = time.perf_counter() - started
    usage = result.get("usage", {}) if isinstance(result, dict) else {}
    content = ""
    try:
        content = result["choices"][0]["message"]["content"]
    except Exception:
        pass
    return {
        "elapsed_seconds": round(elapsed, 3),
        "usage": usage,
        "content": content,
    }


def stream_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = build_request(f"{base_url}/chat/completions", payload=payload, api_key=api_key)
    first_token_at: Optional[float] = None
    started = time.perf_counter()
    usage: Dict[str, Any] = {}
    chunks: List[str] = []

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_text = line[6:]
                if payload_text == "[DONE]":
                    break
                data = json.loads(payload_text)
                if "error" in data:
                    raise RuntimeError(f"stream error: {data['error']}")
                if "usage" in data and isinstance(data["usage"], dict):
                    usage = data["usage"]
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    chunks.append(text)
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} stream failure for {model}: {body}") from exc

    ended = time.perf_counter()
    return {
        "ttft_seconds": round((first_token_at - started), 3) if first_token_at is not None else None,
        "elapsed_seconds": round((ended - started), 3),
        "usage": usage,
        "content": "".join(chunks),
    }


def benchmark_model(args: argparse.Namespace, model: str) -> Dict[str, Any]:
    if args.cold_start:
        unload(args.base_url, args.api_key)

    short_result = nonstream_chat(
        args.base_url,
        args.api_key,
        model,
        args.short_prompt,
        args.short_max_tokens,
    )

    if args.cold_start:
        unload(args.base_url, args.api_key)

    long_result = nonstream_chat(
        args.base_url,
        args.api_key,
        model,
        args.long_prompt,
        args.long_max_tokens,
    )

    if args.cold_start:
        unload(args.base_url, args.api_key)

    stream_result = stream_chat(
        args.base_url,
        args.api_key,
        model,
        args.long_prompt,
        args.long_max_tokens,
    )

    long_usage = long_result.get("usage", {})
    stream_usage = stream_result.get("usage", {})
    return {
        "model": model,
        "short_elapsed_s": short_result["elapsed_seconds"],
        "short_completion_tokens": long_or_zero(short_result["usage"].get("completion_tokens")),
        "short_tps": long_or_zero(short_result["usage"].get("tokens_per_second")),
        "long_elapsed_s": long_result["elapsed_seconds"],
        "long_completion_tokens": long_or_zero(long_usage.get("completion_tokens")),
        "long_tps": long_or_zero(long_usage.get("tokens_per_second")),
        "stream_ttft_s": stream_result["ttft_seconds"] or 0.0,
        "stream_elapsed_s": stream_result["elapsed_seconds"],
        "stream_completion_tokens": long_or_zero(stream_usage.get("completion_tokens")),
        "stream_tps": long_or_zero(stream_usage.get("tokens_per_second")),
    }


def long_or_zero(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def render_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "model",
        "short_elapsed_s",
        "short_tps",
        "long_elapsed_s",
        "long_tps",
        "stream_ttft_s",
        "stream_elapsed_s",
        "stream_tps",
    ]
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(format_cell(row.get(header))))

    def render_row(values: Iterable[str]) -> str:
        return " | ".join(value.ljust(widths[header]) for value, header in zip(values, headers))

    lines = [
        render_row(headers),
        "-+-".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append(render_row([format_cell(row.get(header)) for header in headers]))
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    fieldnames = list(rows[0].keys()) if rows else []
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: str, rows: List[Dict[str, Any]]) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "model",
        "short_elapsed_s",
        "short_tps",
        "long_elapsed_s",
        "long_tps",
        "stream_ttft_s",
        "stream_elapsed_s",
        "stream_tps",
    ]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(format_cell(row.get(h)) for h in headers) + " |\n")


def main() -> int:
    args = parse_args()
    models = args.models or fetch_models(args.base_url, args.api_key)
    if not models:
        print("No models found to benchmark.", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    for model in models:
        print(f"[INFO] Benchmarking {model}...", file=sys.stderr)
        try:
            rows.append(benchmark_model(args, model))
        except Exception as exc:
            print(f"[ERROR] {model}: {exc}", file=sys.stderr)
            rows.append(
                {
                    "model": model,
                    "short_elapsed_s": "error",
                    "short_completion_tokens": 0,
                    "short_tps": 0,
                    "long_elapsed_s": "error",
                    "long_completion_tokens": 0,
                    "long_tps": 0,
                    "stream_ttft_s": "error",
                    "stream_elapsed_s": "error",
                    "stream_completion_tokens": 0,
                    "stream_tps": 0,
                }
            )

    print(render_table(rows))
    write_csv(args.csv, rows)
    write_markdown(args.markdown, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
