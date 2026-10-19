"""Reproducible single-user article benchmark; run inside the AI LXC.

Uses only Python stdlib. Raw SSE, timings, request bodies and telemetry are
preserved. API token counts are retokenized output, not native decode counters.
"""

import datetime
import glob
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.request

OUT = pathlib.Path(sys.argv[1])
if (OUT / "results.jsonl").exists():
    raise SystemExit(
        "Use a new evidence directory; refusing to append to an existing run."
    )
OUT.mkdir(parents=True, exist_ok=True)
BASE = os.environ.get("ILLAMA_BENCH_URL", "http://127.0.0.1:11435").rstrip("/")
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("ILLAMA_API_KEY"):
    HEADERS["Authorization"] = "Bearer " + os.environ["ILLAMA_API_KEY"]
MODELS = ["gemma-4-E4B-it-int4-ov", "gemma-4-12B-it-qat-int4-ov", "Qwen3.5-9B-ov-awq"]
SHORT = "Reply with exactly these words and nothing else: The GPU is working."
GENERATE = (
    "Write a detailed practical guide to organizing a community library. "
    "Cover cataloging, lending, volunteers, accessibility, events, budgeting, "
    "and preservation. Use numbered sections and continue until all topics "
    "are thoroughly explained. Do not include a preamble."
)
CONTEXT = "\n".join(
    f"Record {i:03d}: Branch {(i%7)+1} received {i%13+2} books and lent {i%9+1} books. Volunteers checked labels and updated the catalog."
    for i in range(1, 81)
)
LONG = (
    "Read the following library activity records.\n"
    + CONTEXT
    + "\nWrite a detailed operational summary discussing inventory, lending, volunteers, "
    "catalog accuracy, and recommended next steps. Use the records and explain limitations."
)
WORK = [
    ("latency", SHORT, 32),
    ("generation", GENERATE, 256),
    ("long_input", LONG, 256),
]


def cmd(*args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=90)
    return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def call(path, data=None, base=BASE):
    req = urllib.request.Request(
        base + path,
        data=None if data is None else json.dumps(data).encode(),
        headers=HEADERS if base == BASE else {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        body = r.read()
        return json.loads(body) if body.strip() else {}


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


protocol = {
    "version": "b50-article-v1",
    "models": MODELS,
    "workloads": [
        {
            "name": n,
            "prompt": p,
            "max_tokens": t,
            "sha256": hashlib.sha256(p.encode()).hexdigest(),
        }
        for n, p, t in WORK
    ],
    "temperature": 0,
    "seed": 42,
    "top_p": 1,
    "concurrency": 1,
    "warm_repetitions": 3,
    "sequence": "For each model: unload; one reload latency request; one unscored warmup per workload followed immediately by three measured repetitions.",
    "cache": "Existing persistent OpenVINO compile cache retained; reload is not first-ever compilation or disk-cache-cold.",
    "ttft": "Client request start to first nonempty content OR reasoning SSE delta; includes load when unloaded.",
    "decode_proxy": "(API completion_tokens - 1)/(last nonempty delta - first nonempty delta). Chunks can contain multiple tokens; approximate, not native decoder speed.",
    "token_counts": "Server retokenizes generated text, including reasoning; tokenizer counts differ across models.",
    "scope": "Text-only, single-user. No image inference, accuracy benchmark, concurrent serving, or wall-outlet power measurement.",
    "comparison": "Repeat identical prompts/settings and report exact weights, quantization, runtime, cache, backend and software on the second GPU. Different conversions/runtimes measure whole stacks, not isolated GPU hardware.",
}
save("protocol.json", protocol)
metadata = {
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "health": call("/illama/health"),
    "tags": call("/api/tags"),
    "kernel": cmd("uname", "-a"),
    "cpu": cmd("lscpu"),
    "image": cmd(
        "docker",
        "inspect",
        "illama-service",
        "--format",
        "{{.Image}} {{.Config.Image}}",
    ),
    "packages": cmd(
        "docker", "exec", "illama-service", "python", "-m", "pip", "freeze"
    ),
    "source_hash": cmd(
        "docker", "exec", "illama-service", "sha256sum", "/app/server.py"
    ),
    "gpu": cmd("lspci", "-nnk", "-s", "04:00.0"),
    "driver_packages": cmd(
        "docker",
        "exec",
        "illama-service",
        "dpkg-query",
        "-W",
        "intel-opencl-icd",
        "level-zero",
    ),
    "model_configs": {},
    "sensor_labels": {},
}
for model in MODELS:
    directory = pathlib.Path("/opt/illama/models") / model
    metadata["model_configs"][model] = {}
    for f in directory.glob("*.json"):
        if f.name in ["config.json", "generation_config.json", "openvino_config.json"]:
            metadata["model_configs"][model][f.name] = json.loads(f.read_text())
    metadata["model_configs"][model]["files"] = [
        {"name": str(f.relative_to(directory)), "bytes": f.stat().st_size}
        for f in directory.rglob("*")
        if f.is_file()
    ]

GPU = "/sys/bus/pci/devices/0000:04:00.0/"
sensors = []
for pattern in [
    "hwmon/hwmon*/temp*_input",
    "hwmon/hwmon*/energy*_input",
    "hwmon/hwmon*/power*_average",
    "tile0/gt*/freq0/act_freq",
    "tile0/gt*/freq0/throttle/status",
]:
    sensors.extend(glob.glob(GPU + pattern))
for s in sensors:
    label = pathlib.Path(s.replace("_input", "_label"))
    if label.exists():
        metadata["sensor_labels"][s] = label.read_text().strip()
save("metadata.json", metadata)
stop = threading.Event()
active = {"run": "idle"}


def monitor():
    with (OUT / "telemetry.jsonl").open("a") as f:
        while not stop.is_set():
            row = {"unix": time.time(), "run": active["run"], "sensors": {}}
            for s in sensors:
                try:
                    row["sensors"][s] = int(pathlib.Path(s).read_text().strip())
                except (OSError, ValueError):
                    pass
            # DRM fdinfo reports process GPU allocations without creating a competing GPU context.
            try:
                pid = (
                    cmd("docker", "top", "illama-service", "-eo", "pid")["stdout"]
                    .splitlines()[1]
                    .strip()
                )
                row["process_status"] = pathlib.Path("/proc", pid, "status").read_text()
                row["drm_fdinfo"] = []
                seen = set()
                for p in pathlib.Path("/proc", pid, "fdinfo").glob("*"):
                    try:
                        txt = p.read_text()
                    except OSError:
                        continue
                    if "drm-pdev:\t0000:04:00.0" in txt:
                        client = next(
                            (
                                x
                                for x in txt.splitlines()
                                if x.startswith("drm-client-id:")
                            ),
                            txt,
                        )
                        if client not in seen:
                            row["drm_fdinfo"].append(txt)
                            seen.add(client)
            except Exception as e:
                row["process_error"] = str(e)
            f.write(json.dumps(row) + "\n")
            f.flush()
            stop.wait(1)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()


def run(model, workload, prompt, limit, phase, rep):
    key = f"{MODELS.index(model)+1}-{workload}-{phase}-{rep}"
    active["run"] = key
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": limit,
        "temperature": 0,
        "seed": 42,
        "top_p": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    row = {
        "id": key,
        "model": model,
        "workload": workload,
        "phase": phase,
        "rep": rep,
        "request": payload,
        "unix_start": time.time(),
    }
    first = last = None
    usage = {}
    events = []
    texts = []
    reasoning = []
    done = False
    start = time.perf_counter()
    try:
        req = urllib.request.Request(
            BASE + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=HEADERS,
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                at = time.perf_counter() - start
                events.append({"seconds": at, "data": body})
                if body == "[DONE]":
                    done = True
                    break
                d = json.loads(body)
                if "error" in d:
                    raise RuntimeError(str(d["error"]))
                if d.get("usage"):
                    usage = d["usage"]
                for choice in d.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if content or thought:
                        if first is None:
                            first = at
                        last = at
                    texts.append(content)
                    reasoning.append(thought)
        elapsed = time.perf_counter() - start
        n = usage.get("completion_tokens", 0)
        row.update(
            elapsed_s=elapsed,
            ttft_s=first,
            last_delta_s=last,
            usage=usage,
            content="".join(texts),
            reasoning="".join(reasoning),
            done=done,
            output_tokens_per_s=n / elapsed if elapsed else None,
            decode_proxy_tokens_per_s=(n - 1) / (last - first)
            if n > 1 and last is not None and last > first
            else None,
        )
    except Exception as e:
        row["error"] = str(e)
    row["events"] = events
    save(key + ".json", row)
    with (OUT / "results.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {
                    k: v
                    for k, v in row.items()
                    if k not in ["events", "request", "content", "reasoning"]
                }
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in row.items()
                if k not in ["events", "request", "content", "reasoning"]
            }
        ),
        flush=True,
    )
    active["run"] = "idle"
    return row


completed = False
try:
    queue = call("/queue", base="http://127.0.0.1:8188")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI queue is not idle")
    call(
        "/free",
        {"unload_models": True, "free_memory": True},
        base="http://127.0.0.1:8188",
    )
    for model in MODELS:
        call("/v1/unload", {})
        run(model, "latency", SHORT, 32, "reload", 0)
        save(f"{MODELS.index(model)+1}-loaded-health.json", call("/illama/health"))
        for name, prompt, limit in WORK:
            warm = run(model, name, prompt, limit, "warmup", 0)
            if "error" in warm:
                break
            for rep in range(1, 4):
                run(model, name, prompt, limit, "warm", rep)
        call("/v1/unload", {})
    completed = True
finally:
    stop.set()
    thread.join(timeout=5)
    save("health-after.json", call("/illama/health"))
    save(
        "logs.json", cmd("docker", "logs", "--since", metadata["utc"], "illama-service")
    )
    (OUT / ("DONE" if completed else "FAILED")).write_text(
        datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
