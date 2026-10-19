"""Benchmark only existing local ComfyUI weights; never downloads models."""

import argparse
import copy
import datetime
import glob
import json
import os
import pathlib
import re
import subprocess
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--illama-url", default="http://127.0.0.1:11435")
    parser.add_argument(
        "--skip-illama-unload",
        action="store_true",
        help="For standalone ComfyUI only; ensure no other GPU workload is active",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.output)
    if (out / "results.jsonl").exists():
        raise SystemExit(
            "Use a new evidence directory; refusing to append to an existing run."
        )
    out.mkdir(parents=True, exist_ok=True)

    def call(path, data=None):
        req = urllib.request.Request(
            args.base_url + path,
            data=None if data is None else json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body.strip() else {}

    def save(name, data):
        (out / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    prompt = "A sunlit community library with wooden shelves, green plants, a reading table, and large windows overlooking a garden, realistic architectural photography, detailed natural materials, soft morning light, no people, no lettering."
    variants = [
        ("zimage-fp8", "z-image-turbo-fp8-e4m3fn.safetensors", "UNETLoader"),
        ("zimage-int8", "z_image_int8_convrot.safetensors", "NunchakuZImageDiTLoader"),
        ("zimage-bf16", "z_image_bf16.safetensors", "UNETLoader"),
        (
            "sdxl-lightning",
            "Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors",
            "CheckpointLoaderSimple",
        ),
    ]
    info = call("/object_info")
    save("object_info.json", info)
    save("system-before.json", call("/system_stats"))
    save(
        "protocol.json",
        {
            "date": "2026-10-18",
            "prompt": prompt,
            "variants": variants,
            "resolutions": [512, 1024],
            "batch": 1,
            "warmup_seed": 41,
            "measured_seeds": [42, 43, 44],
            "zimage": {
                "steps": 8,
                "cfg": 1,
                "sampler": "euler",
                "scheduler": "simple",
                "shift": 3,
            },
            "sdxl": {
                "steps": 4,
                "cfg": 1,
                "sampler": "euler",
                "scheduler": "sgm_uniform",
            },
            "caching": "Model and conditioning caches retained for warm repetitions; seeds vary to force sampler execution. First job per model includes initial loading; not a clean disk/compile cold start.",
            "fit_definition": "Success at these settings, with existing dynamic VRAM/offload enabled, is not proof the whole pipeline resides in 16GB.",
            "downloads": "Only existing local loader filenames. No model downloads or cloud API nodes.",
        },
    )

    queue = call("/queue")
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("Refusing to mix benchmark jobs with an occupied queue")
    # LLM and image inference must not compete for the B50.
    headers = {"Content-Type": "application/json"}
    if os.environ.get("ILLAMA_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["ILLAMA_API_KEY"]
    req = urllib.request.Request(
        args.illama_url.rstrip("/") + "/v1/unload",
        data=b"{}",
        headers=headers,
    )
    if not args.skip_illama_unload:
        urllib.request.urlopen(req, timeout=30).close()
    stop = threading.Event()
    active = {"id": "idle"}
    sensor_paths = []
    for pattern in [
        "hwmon/hwmon*/temp*_input",
        "hwmon/hwmon*/energy*_input",
        "tile0/gt*/freq0/act_freq",
    ]:
        sensor_paths += glob.glob("/sys/bus/pci/devices/0000:04:00.0/" + pattern)

    def monitor():
        with (out / "telemetry.jsonl").open("a") as handle:
            while not stop.is_set():
                sample = {"unix": time.time(), "id": active["id"], "sensors": {}}
                for path in sensor_paths:
                    try:
                        sample["sensors"][path] = int(pathlib.Path(path).read_text())
                    except (OSError, ValueError):
                        pass
                try:
                    text = subprocess.check_output(
                        ["docker", "top", "comfyui", "-eo", "pid,comm"], text=True
                    )
                    vram = rss = 0
                    for line in text.splitlines()[1:]:
                        if "python" not in line:
                            continue
                        pid = line.split()[0]
                        status = pathlib.Path("/proc", pid, "status").read_text()
                        match = re.search(r"^VmRSS:\s*(\d+) kB", status, re.M)
                        if match:
                            rss += int(match[1]) * 1024
                        seen = set()
                        for fd in pathlib.Path("/proc", pid, "fdinfo").glob("*"):
                            try:
                                data = fd.read_text()
                            except OSError:
                                continue
                            if "drm-pdev:\t0000:04:00.0" not in data:
                                continue
                            client = re.search(r"drm-client-id:\s*(\d+)", data)
                            if client and client[1] not in seen:
                                seen.add(client[1])
                                vram += sum(
                                    int(x) * 1024
                                    for x in re.findall(
                                        r"drm-resident-vram\d+:\s*(\d+) KiB", data
                                    )
                                )
                    sample.update(process_vram_bytes=vram, process_rss_bytes=rss)
                except Exception as exc:
                    sample["monitor_error"] = str(exc)
                handle.write(json.dumps(sample) + "\n")
                handle.flush()
                stop.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        for name, filename, loader in variants:
            call("/free", {"unload_models": True, "free_memory": True})
            model_failed = False
            for size in [512, 1024]:
                for repetition, seed in enumerate([41, 42, 43, 44]):
                    key = f"{name}-{size}-{repetition}"
                    graph = {
                        "2": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": prompt, "clip": ["1", 1]},
                        },
                        "3": {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": "", "clip": ["1", 1]},
                        },
                        "4": {
                            "class_type": "EmptyLatentImage",
                            "inputs": {"width": size, "height": size, "batch_size": 1},
                        },
                        "5": {
                            "class_type": "KSampler",
                            "inputs": {
                                "model": ["1", 0],
                                "seed": seed,
                                "steps": 4,
                                "cfg": 1.0,
                                "sampler_name": "euler",
                                "scheduler": "sgm_uniform",
                                "positive": ["2", 0],
                                "negative": ["3", 0],
                                "latent_image": ["4", 0],
                                "denoise": 1.0,
                            },
                        },
                        "6": {
                            "class_type": "VAEDecode",
                            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                        },
                        "7": {
                            "class_type": "SaveImage",
                            "inputs": {
                                "images": ["6", 0],
                                "filename_prefix": "b50-benchmark-20261018/" + key,
                            },
                        },
                    }
                    if loader == "CheckpointLoaderSimple":
                        graph["1"] = {
                            "class_type": loader,
                            "inputs": {"ckpt_name": filename},
                        }
                    else:
                        graph["1"] = {
                            "class_type": loader,
                            "inputs": {"model_name": filename}
                            if loader == "NunchakuZImageDiTLoader"
                            else {"unet_name": filename, "weight_dtype": "default"},
                        }
                        graph["8"] = {
                            "class_type": "CLIPLoader",
                            "inputs": {
                                "clip_name": "qwen_3_4b_fp8_mixed.safetensors",
                                "type": "lumina2",
                                "device": "default",
                            },
                        }
                        graph["9"] = {
                            "class_type": "VAELoader",
                            "inputs": {"vae_name": "ae.safetensors"},
                        }
                        graph["10"] = {
                            "class_type": "ModelSamplingAuraFlow",
                            "inputs": {"model": ["1", 0], "shift": 3.0},
                        }
                        graph["2"]["inputs"]["clip"] = ["8", 0]
                        graph["3"]["inputs"]["clip"] = ["8", 0]
                        graph["4"]["class_type"] = "EmptySD3LatentImage"
                        graph["5"]["inputs"].update(
                            model=["10", 0], steps=8, scheduler="simple"
                        )
                        graph["6"]["inputs"]["vae"] = ["9", 0]
                    active["id"] = key
                    row = {
                        "id": key,
                        "model": name,
                        "file": filename,
                        "resolution": size,
                        "rep": repetition,
                        "seed": seed,
                        "unix_start": time.time(),
                        "graph": copy.deepcopy(graph),
                    }
                    start = time.perf_counter()
                    try:
                        response = call(
                            "/prompt",
                            {"prompt": graph, "client_id": "b50-article-benchmark"},
                        )
                        row["submission"] = response
                        prompt_id = response["prompt_id"]
                        while True:
                            history = call("/history/" + prompt_id)
                            if prompt_id in history:
                                entry = history[prompt_id]
                                row["history"] = entry
                                row["success"] = (
                                    entry.get("status", {}).get("status_str")
                                    == "success"
                                )
                                break
                            if time.perf_counter() - start > args.timeout:
                                call("/interrupt", {})
                                raise TimeoutError("Interrupted at benchmark timeout")
                            time.sleep(0.25)
                    except Exception as exc:
                        row["error"] = str(exc)
                        row["success"] = False
                    row["wall_s"] = time.perf_counter() - start
                    save(key + ".json", row)
                    with (out / "results.jsonl").open("a") as handle:
                        handle.write(
                            json.dumps({k: v for k, v in row.items() if k != "graph"})
                            + "\n"
                        )
                    print(
                        json.dumps({k: row[k] for k in ["id", "success", "wall_s"]}),
                        flush=True,
                    )
                    active["id"] = "idle"
                    if not row["success"]:
                        model_failed = True
                        break
                if model_failed:
                    break
    finally:
        stop.set()
        thread.join(timeout=5)
        call("/free", {"unload_models": True, "free_memory": True})
        save("system-after.json", call("/system_stats"))
        logs = subprocess.run(
            ["docker", "logs", "--since", started, "comfyui"],
            capture_output=True,
            text=True,
        )
        save("logs.json", {"stdout": logs.stdout, "stderr": logs.stderr})
    (out / "DONE").write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())


if __name__ == "__main__":
    main()
