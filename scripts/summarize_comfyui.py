"""Summarize saved ComfyUI benchmark jobs and sampled GPU memory."""

import argparse
import csv
import html
import json
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    root = pathlib.Path(args.directory)
    rows = [json.loads(x) for x in (root / "results.jsonl").read_text().splitlines()]
    telemetry = [
        json.loads(x) for x in (root / "telemetry.jsonl").read_text().splitlines()
    ]
    summary = []
    for model, size in dict.fromkeys((r["model"], r["resolution"]) for r in rows):
        group = [r for r in rows if r["model"] == model and r["resolution"] == size]
        warm = [r for r in group if r["rep"] > 0 and r["success"]]
        ids = {r["id"] for r in warm or group}
        samples = [s for s in telemetry if s["id"] in ids]
        values = [r["wall_s"] for r in warm]
        server_times = []
        for r in warm:
            messages = dict(r.get("history", {}).get("status", {}).get("messages", []))
            begin = messages.get("execution_start", {}).get("timestamp")
            end = messages.get("execution_success", {}).get("timestamp")
            if begin is not None and end is not None:
                server_times.append((end - begin) / 1000)
        errors = []
        for r in group:
            for name, detail in (
                r.get("history", {}).get("status", {}).get("messages", [])
            ):
                if name == "execution_error":
                    errors.append(
                        {
                            "node": detail.get("node_type"),
                            "type": detail.get("exception_type"),
                            "message": detail.get("exception_message"),
                        }
                    )
        temps = [
            v / 1000
            for s in samples
            for k, v in s.get("sensors", {}).items()
            if "/temp" in k and v > 0
        ]
        summary.append(
            {
                "model": model,
                "file": group[0]["file"],
                "resolution": size,
                "successful_jobs": sum(r["success"] for r in group),
                "attempted_jobs": len(group),
                "warm_repeats": len(warm),
                "warm_wall_median_s": statistics.median(values) if values else None,
                "warm_wall_min_s": min(values) if values else None,
                "warm_wall_max_s": max(values) if values else None,
                "server_execution_median_s": statistics.median(server_times)
                if server_times
                else None,
                "first_job_s": group[0]["wall_s"],
                "peak_process_vram_gib": max(
                    (s.get("process_vram_bytes", 0) / 2**30 for s in samples), default=0
                ),
                "peak_process_rss_gib": max(
                    (s.get("process_rss_bytes", 0) / 2**30 for s in samples), default=0
                ),
                "peak_sensor_c": max(temps) if temps else None,
                "errors": errors,
                "quality_note": "Eight-step BF16 sample visibly degraded; runtime success is not a validated quality recipe."
                if model == "zimage-bf16"
                else (
                    "Format mismatch, not demonstrated OOM."
                    if errors
                    else "Representative 1024 image visually checked; no quantitative quality score."
                ),
            }
        )
    result = {
        "date": "2026-10-18",
        "attempted_jobs": len(rows),
        "successful_jobs": sum(r["success"] for r in rows),
        "summary": summary,
        "limitations": [
            "Batch 1; 512/1024 square; warm seeds 42,43,44 after seed-41 warmup.",
            "Z-Image uses 8 steps, cfg 1, euler/simple, shift 3; SDXL Lightning uses 4 steps, cfg 1, euler/sgm_uniform. These are different pipelines, not equal-work throughput comparisons.",
            "Changing seeds forces denoising; graph/model/conditioning caches remain enabled.",
            "Dynamic VRAM and CPU offload remain enabled with 3GB reserve. Success is not proof the full pipeline is simultaneously GPU-resident.",
            "VRAM/RSS sampled every approximately 0.5s; brief peaks can be missed. RSS includes resident/mapped memory and is not total system RAM requirement.",
            "Client timings include up to approximately 250ms history polling overhead; server execution timings are also recorded.",
            "Existing INT8 artifact fails a loader format assertion before sampling, not a VRAM OOM.",
            "No model files were downloaded. LoRAs, VAEs and adapters are components, not independent text-to-image pipelines.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    columns = [
        "model",
        "resolution",
        "warm_repeats",
        "warm_wall_median_s",
        "server_execution_median_s",
        "peak_process_vram_gib",
        "peak_process_rss_gib",
        "quality_note",
    ]
    table = (
        "<tr>" + "".join("<th>" + html.escape(k) + "</th>" for k in columns) + "</tr>"
    )
    for row in summary:
        table += (
            "<tr>"
            + "".join(
                "<td>"
                + html.escape(
                    f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k])
                )
                + "</td>"
                for k in columns
            )
            + "</tr>"
        )
    images = "".join(
        '<figure><img style="max-width:100%;width:480px" src="previews/'
        + html.escape(p.name)
        + '"><figcaption>'
        + html.escape(p.name)
        + "</figcaption></figure>"
        for p in sorted((root / "previews").glob("*1024*"))
    )
    (root / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>ComfyUI B50 measurements</title><style>body{font:16px system-ui;margin:32px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:10px}figure{display:inline-block;max-width:30%;vertical-align:top}</style><h1>ComfyUI on Arc Pro B50 — October 18, 2026</h1><p>ComfyUI 0.39.0; torch 2.14.0+xpu; source-built Intel Omni 0.2.0-b2. Existing models only.</p><table>'
        + table
        + "</table><h2>Representative outputs</h2>"
        + images
        + "<h2>Limits</h2><ul>"
        + "".join("<li>" + html.escape(x) + "</li>" for x in result["limitations"])
        + "</ul>",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
