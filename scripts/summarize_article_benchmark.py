"""Build CSV and a standalone HTML report from article benchmark evidence."""

import csv
import html
import json
import pathlib
import re
import statistics
import sys

root = pathlib.Path(sys.argv[1])
rows = [json.loads(x) for x in (root / "results.jsonl").read_text().splitlines()]
samples = [json.loads(x) for x in (root / "telemetry.jsonl").read_text().splitlines()]
meta = json.loads((root / "metadata.json").read_text())
summary = []
for model in dict.fromkeys(r["model"] for r in rows):
    for workload in ["latency", "generation", "long_input"]:
        group = [
            r
            for r in rows
            if r["model"] == model
            and r["workload"] == workload
            and r["phase"] == "warm"
            and "error" not in r
        ]
        if not group:
            continue
        for r in group:
            raw = json.loads((root / (r["id"] + ".json")).read_text())
            r["visible_content_chars"] = len(raw.get("content", ""))
            r["reasoning_chars"] = len(raw.get("reasoning", ""))
            r["thinking_text_in_content"] = (
                raw.get("content", "").lstrip().startswith("Thinking Process:")
            )
            r["exact_reply_ok"] = (
                raw.get("content", "").strip() == "The GPU is working."
            )
            r["first_visible_content_s"] = None
            r["first_reasoning_s"] = None
            for event in raw.get("events", []):
                if event["data"] == "[DONE]":
                    continue
                event_data = json.loads(event["data"])
                for choice in event_data.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") and r["first_visible_content_s"] is None:
                        r["first_visible_content_s"] = event["seconds"]
                    if (
                        delta.get("reasoning_content")
                        and r["first_reasoning_s"] is None
                    ):
                        r["first_reasoning_s"] = event["seconds"]
        ids = {r["id"] for r in group}
        tele = [s for s in samples if s["run"] in ids]

        def median(key):
            vals = [r[key] for r in group if r.get(key) is not None]
            return statistics.median(vals) if vals else None

        temps = [
            v / 1000
            for s in tele
            for k, v in s["sensors"].items()
            if "/temp" in k and k.endswith("_input") and v > 0
        ]
        vram = []
        rss = []
        for s in tele:
            total = 0
            for text in s.get("drm_fdinfo", []):
                total += sum(
                    int(v)
                    for v in re.findall(r"drm-resident-vram\d+:\s*(\d+) KiB", text)
                )
            if total:
                vram.append(total / 1048576)
            m = re.search(r"^VmRSS:\s*(\d+) kB", s.get("process_status", ""), re.M)
            if m:
                rss.append(int(m[1]) / 1048576)
        energy = {}
        for key, label in meta.get("sensor_labels", {}).items():
            if "/energy" not in key:
                continue
            joules = seconds = 0
            for run in ids:
                ss = [s for s in tele if s["run"] == run and key in s["sensors"]]
                if len(ss) >= 2:
                    delta = ss[-1]["sensors"][key] - ss[0]["sensors"][key]
                    if delta >= 0:
                        joules += delta / 1e6
                        seconds += ss[-1]["unix"] - ss[0]["unix"]
            if seconds:
                energy[label] = {
                    "sampled_joules": joules,
                    "sampled_seconds": seconds,
                    "mean_watts": joules / seconds,
                }
        summary.append(
            {
                "model": model,
                "workload": workload,
                "n": len(group),
                "prompt_tokens": group[0]["usage"].get("prompt_tokens"),
                "backend": json.loads(
                    (
                        root
                        / f'{list(dict.fromkeys(r["model"] for r in rows)).index(model)+1}-loaded-health.json'
                    ).read_text()
                )["pipeline_backend"],
                "output_tokens_median": statistics.median(
                    r["usage"].get("completion_tokens", 0) for r in group
                ),
                "ttft_median_s": median("ttft_s"),
                "ttft_min_s": min(
                    r["ttft_s"] for r in group if r.get("ttft_s") is not None
                ),
                "first_visible_content_median_s": median("first_visible_content_s"),
                "first_reasoning_median_s": median("first_reasoning_s"),
                "visible_content_chars_median": median("visible_content_chars"),
                "reasoning_chars_median": median("reasoning_chars"),
                "thinking_text_in_content_runs": sum(
                    r["thinking_text_in_content"] for r in group
                ),
                "exact_reply_successes": sum(r["exact_reply_ok"] for r in group)
                if workload == "latency"
                else None,
                "ttft_max_s": max(
                    r["ttft_s"] for r in group if r.get("ttft_s") is not None
                ),
                "elapsed_median_s": median("elapsed_s"),
                "end_to_end_tokens_per_s": median("output_tokens_per_s"),
                "decode_proxy_tokens_per_s": median("decode_proxy_tokens_per_s")
                if workload != "latency"
                else None,
                "peak_process_vram_gib": max(vram) if vram else None,
                "peak_process_rss_gib": max(rss) if rss else None,
                "peak_sensor_c": max(temps) if temps else None,
                "energy_domains": energy,
            }
        )
result = {
    "summary": summary,
    "reloads": [r for r in rows if r["phase"] == "reload"],
    "errors": [r for r in rows if "error" in r or not r.get("done")],
    "completed_request_count": len(rows),
    "expected_request_count": 39,
    "limitations": [
        "Single reload sample per model; existing compilation and OS caches retained.",
        "Gemma models were smoke-tested on this nightly before benchmarking; Qwen was not. Reload timings therefore have different possible compilation-cache histories and are not comparable clean cold starts.",
        "Three warm repeats per workload; ranges describe those observations, not population confidence intervals.",
        "Exact repeated prompts may benefit from runtime caches; no assertion of cache-free prefill speed.",
        "TTFT measures first visible streamed text or reasoning, not an internal first token. Chunk buffering can inflate TTFT and decode proxy.",
        "Output token counts come from server retokenization. Decode proxy is not a native decoder measurement and is omitted for short replies.",
        "Long-input latency is not isolated prefill throughput. No pure-prefill tokens/s claim is made.",
        "GPU resident memory and process RSS are sampled about once per second; brief peaks can be missed.",
        "Energy domains are driver hwmon counters, not whole-system wall power. Estimates exclude unsampled request boundaries and idle subtraction.",
        "No independent accuracy, image, multi-user concurrency, or full 8K-context benchmark.",
        "A 256-token cap includes any generated reasoning. Inspect saved reasoning and visible responses before comparing answer completeness.",
        "Qwen emits plain-text Thinking Process planning in the normal content channel and exhausts the token cap. Its timing is throughput of that planning text, not time to a completed requested answer. first_visible_content means API content, not a final-answer boundary.",
        "Telemetry runs alongside the benchmark and can add small CPU overhead. No other B50 jobs were deliberately started.",
        "Use the same model artifacts, quantization and software where possible for a hardware comparison; otherwise compare complete stacks.",
    ],
}
(root / "summary.json").write_text(json.dumps(result, indent=2))
with (root / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]) if summary else ["model"])
    writer.writeheader()
    writer.writerows(
        {**s, "energy_domains": json.dumps(s["energy_domains"])} for s in summary
    )
columns = [
    "model",
    "workload",
    "n",
    "prompt_tokens",
    "output_tokens_median",
    "ttft_median_s",
    "elapsed_median_s",
    "end_to_end_tokens_per_s",
    "peak_process_vram_gib",
    "peak_sensor_c",
]


def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


table = "<tr>" + "".join("<th>" + html.escape(c) + "</th>" for c in columns) + "</tr>"
table += "".join(
    "<tr>"
    + "".join("<td>" + html.escape(fmt(s[c])) + "</td>" for c in columns)
    + "</tr>"
    for s in summary
)
energy_table = "<tr><th>Model</th><th>Workload</th><th>Driver energy domain</th><th>Sampled mean W</th><th>Sampled peak process RSS GiB</th></tr>"
for s in summary:
    for domain, values in s["energy_domains"].items():
        energy_table += (
            "<tr>"
            + "".join(
                "<td>" + html.escape(fmt(v)) + "</td>"
                for v in [
                    s["model"],
                    s["workload"],
                    domain,
                    values["mean_watts"],
                    s["peak_process_rss_gib"],
                ]
            )
            + "</tr>"
        )
reload_table = (
    "<tr><th>Model</th><th>Reload TTFT s</th><th>Reload total s</th></tr>"
    + "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(fmt(v)) + "</td>"
            for v in [r["model"], r.get("ttft_s"), r.get("elapsed_s")]
        )
        + "</tr>"
        for r in result["reloads"]
    )
)
doc = (
    """<!doctype html><meta charset="utf-8"><title>Arc Pro B50 — three-model inference benchmark</title>
<style>body{font:16px system-ui;max-width:1500px;margin:40px auto;padding:0 24px;color:#17252b}table{border-collapse:collapse;font-size:13px}td,th{padding:10px;border:1px solid #ccd5d8;text-align:left}th{background:#eaf1f3}li{margin:8px 0}pre{white-space:pre-wrap;background:#f1f4f5;padding:18px}h1{font-size:30px}</style>
<h1>Intel Arc Pro B50: three-model inference benchmark</h1><p>Measured October 18, 2026. Illama 2 / OpenVINO 2026.5 nightly (October 15). Text-only, single-user, greedy decoding, seed 42. Timings are local to the LXC, excluding LAN latency.</p><p><strong>Qwen qualification:</strong> Qwen outputs plain-text planning (Thinking Process) in the normal content channel and reaches the token cap before answering. Its rates below measure generated planning text. They do not establish equal answer quality or time to a completed answer.</p>
<h2>Warm measurements</h2><p>All rates are output tokens per second, using each model's own tokenizer. TTFT and elapsed values are seconds. Memory is GiB. Values are medians of three requests; raw trials and observed ranges are in summary.json and results.jsonl.</p><table>"""
    + table
    + "</table><h2>Reload measurements</h2><p>One request per model after unload, with existing compilation cache retained.</p><table>"
    + reload_table
    + "</table><h2>Sampled GPU energy counters and process memory</h2><p>Driver-reported energy, without idle subtraction; partial sampled windows only. Energy domains may overlap and must not be added together. This is not wall-outlet power.</p><table>"
    + energy_table
    + "</table><h2>Interpretation limits</h2><ul>"
    + "".join("<li>" + html.escape(x) + "</li>" for x in result["limitations"])
    + "</ul><h2>Protocol</h2><pre>"
    + html.escape((root / "protocol.json").read_text())
    + "</pre><h2>Errors</h2><pre>"
    + html.escape(json.dumps(result["errors"], indent=2))
    + "</pre>"
)
(root / "report.html").write_text(doc, encoding="utf-8")
print(json.dumps(result, indent=2))
