# B50 Model & Platform Notes

Findings from running illama on an **Intel Arc Pro B50 16 GB**, inside a
**Proxmox LXC**. Living document — append entries with date + runtime build
string so future bisects are possible.

**Current reproducible baseline (October 18, 2026):** see the
[dated three-model benchmark](benchmarks/2026-10-18-b50.md) and the
[B50 quickstart](arc-b50-quickstart.md). The runner tested there uses OpenVINO
2026.5 nightly builds dated October 15. Older results and model verdicts below
are historical observations, not newly validated coverage or comparable trials.
Do not merge their throughput figures with the new benchmark protocol, infer
quality equivalence, or treat a past export failure as a current upstream fact.

> **⚠️ Read the "Platform gotchas" section first.** Between 2026-05-10 and
> 2026-08-11 every benchmark in this doc was silently measured on the
> **Arrow Lake iGPU**, not the B50, because the LXC was passing through the
> wrong DRM render node. Several conclusions were wrong as a result and have
> been corrected below.

**TL;DR (2026-08-11; runtime pins refreshed 2026-10-04 → OpenVINO / GenAI 2026.4.1)**

- illama on openvino-genai is the right backbone, and on the **real B50** it
  is **~2× faster than Ollama-Vulkan** for the same model. No sidecar needed.
- Qwen3.5-9B: **32.7 tps** via illama/openvino-genai (AWQ INT4) vs
  **15.3 tps** via Ollama-Vulkan (Q4_K_M GGUF).
- Live runner is on **openvino-genai 2026.4.1.0** / **openvino 2026.4.1**
  (was 2026.3.0 in Aug). Runtime bumps are rarely a large tps lever —
  speculative decoding still is. Always pin `ILLAMA_DEVICE=GPU.1` on this host.
- **The iGPU beats the B50 on Gemma-4 E4B** (43.9 vs 38.2 tps) — small MoE
  is bandwidth-bound, not compute-bound.
- The earlier "llama.cpp/Ollama is 5–7× faster" conclusion was an artifact of
  comparing our iGPU numbers against other people's B70 numbers. It is wrong.
- **DFlash is supported** as of openvino-genai PR #3938 (2026-07-22), which
  added it to the continuous-batching pipeline. An earlier note here said
  otherwise; that was true in May, not now. Untested on this hardware.
- **EAGLE-3 speculative decoding is the big win: 2.52x on Qwen3-8B**
  (37.9 -> 95.6 tps), 1.80x on Qwen3-1.7B. The speedup grows with model
  size. Recipe: export target and drafter as a matched pair with
  optimum-intel 2.1.0, link via model.json, and let illama disable prefix
  caching (incompatible with speculative decoding).
- **Recommended config: `Qwen3-8B-int4-ov` + `Qwen3-8B-eagle3-ov` at
  95.6 tps** - 2.93x the 32.6 tps of the Qwen3.5-9B setup, at comparable
  capability.
- Qwen3.5 cannot be exported in-house: optimum-intel imports
  `Qwen3_5DynamicCache`, which exists in no transformers version its own
  pin permits. Upstream bug, no local workaround.
- Status: the throughput question is **answered**. Remaining watch items are
  Vulkan/Mesa `xe` maturity, the Qwen3.5 export bug, and DFlash (supported
  since 2026-07-22, untested here).

---

## Platform gotchas (the expensive lessons)

### 1. Proxmox LXC was passing the wrong GPU

The LXC config had a section commented `# --- INTEL ARC B50 (Xe) ---` that
actually passed through the **Arrow Lake iGPU** DRM nodes. Always bind by
**PCI address**, because `cardN` / `renderDN` reorder across boots.

When an Arrow Lake iGPU and an Arc dGPU are both visible, OpenVINO numbers
them `GPU.0` (iGPU) and `GPU.1` (dGPU, name contains `0xe212` on Battlemage
G21). Illama must use `ILLAMA_DEVICE=GPU.1`. DRM `cardN` / `renderDN` reorder
across boots, so bind the LXC by PCI address, then re-check. An example
mapping from one such host (do not copy the card numbers):

| Device | card | render | PCI | Kernel driver |
|---|---|---|---|---|
| Arrow Lake iGPU | `card0` | `renderD128` | `00:02.0` | `i915` |
| **Arc Pro B50 (BMG G21)** | `card1` | `renderD129` | `04:00.0` | `xe` |

Verify the mapping (Proxmox host and inside the LXC):

```bash
for r in /dev/dri/renderD*; do
  minor=${r##*/renderD}
  pci=$(readlink /sys/dev/char/226:$minor/device 2>/dev/null | sed 's|.*/||')
  echo "$r -> $pci -> $(lspci -s "$pci" 2>/dev/null | sed 's/^.*: //')"
done
```

When editing the LXC config, mount the DRM nodes that currently map to the
B50 PCI address (`04:00.0` / `8086:e212` on this class of card). Example
pattern (replace the minors after running the loop above):

```
# Example only — replace 226:X with the live B50 card/render minors
lxc.cgroup2.devices.allow: c 226:<B50_card_minor> rwm
lxc.cgroup2.devices.allow: c 226:<B50_render_minor> rwm
lxc.mount.entry: /dev/dri/cardY dev/dri/cardY none bind,optional,create=file
lxc.mount.entry: /dev/dri/renderDZ dev/dri/renderDZ none bind,optional,create=file
```

**Symptom to watch for:** OpenVINO reports a single `GPU` device named
`Intel(R) Graphics (iGPU)` instead of two devices. Check with:

```bash
docker exec illama-service /opt/illama-venv/bin/python -c \
  'import openvino as ov; c=ov.Core(); print(c.available_devices); [print(" ",d,":",c.get_property(d,"FULL_DEVICE_NAME")) for d in c.available_devices]'
```

Correct output on this host:

```
['CPU', 'GPU.0', 'GPU.1']
  GPU.0 : Intel(R) Graphics (iGPU)              <- Arrow Lake — do not use
  GPU.1 : Intel(R) Graphics [0xe212] (dGPU)     <- B50 — Illama pin
```

`0xe212` is the Battlemage G21 PCI device ID. Illama must use
`ILLAMA_DEVICE=GPU.1` whenever an iGPU is also present.

### 2. illama defaults to the wrong GPU when two are present

`ILLAMA_DEVICE=GPU` resolves to `GPU.0`, which is the iGPU here. Pin it
explicitly in `deploy/.env`:

```
ILLAMA_DEVICE=GPU.1
```

### 3. Ollama couldn't see the B50 — group permissions

Inside the LXC, `/dev/dri/renderD*` are owned `root:kvm` (the host's `render`
gid maps to `kvm` inside the container). The `ollama` system user was in
`video` + `render`, **not** `kvm`, so Mesa silently skipped both Intel GPUs
and Ollama fell back to another GPU, llvmpipe, or CPU.

```bash
usermod -aG kvm ollama && systemctl restart ollama
```

After that, Ollama enumerates the B50 automatically — **no
`GGML_VK_VISIBLE_DEVICES` override needed.** In fact setting that variable
made things worse: ggml's Vulkan index order differs from `vulkaninfo`'s, so
`=1` selected llvmpipe (crash: `pre-allocated tensor (cache_k_l3) in a buffer
(Vulkan0) that cannot run the operation`) and `=0` filtered out everything.

Ground truth for which device Ollama picked (not nvtop — nvtop's device
numbering is its own):

```bash
journalctl -u ollama.service --no-pager | grep -A3 'device_info:' | tail -4
```

Want to see:
```
- Vulkan0 : Intel(R) Arc(tm) Pro B50 Graphics (BMG G21) (16304 MiB, 14288 MiB free)
```

### 4. A B50 SR-IOV virtual function steals VRAM from Illama and ComfyUI

Xe partitions LMEM when a VF is created. One auto-provisioned VF took ~15.8 GiB and left the PF ~128 MiB. `/system_stats` and similar “total = free” readouts still showed 16 GB. Keep `sriov_numvfs=0` on the B50 PF (`04:00.0`) while this CT owns the card. Unload Illama before a heavy ComfyUI job (and the reverse); they share the same 16 GB.

### 5. Vulkan needs the Intel client driver stack

`vulkaninfo` must list the B50 before Ollama can use it. Mesa 26.0.3 on this
host does enumerate `xe`-bound Battlemage correctly:

```bash
vulkaninfo --summary 2>/dev/null | grep -E '^GPU[0-9]:|deviceName' | paste - - | column -t
```

---

## Benchmarks

Prompt: `"Write a 300-word story."`, `max_tokens=400`, `temperature=0.7`.
Thinking mode disabled for Ollama (`"think": false`) since Qwen3.5 otherwise
burns the whole budget on `Thinking Process:` preamble.

### Corrected — same model, same box, two backends

| Backend | Model / format | Device | tps |
|---|---|---|---|
| **illama / openvino-genai** | `Qwen3.5-9B-ov-awq` (AWQ INT4) | **B50** | **32.7** |
| Ollama / Vulkan | `qwen3.5:9b` (Q4_K_M GGUF) | B50 | 15.3 |

**openvino-genai is 2.1× faster than Ollama-Vulkan on this hardware.**
Likely because Vulkan doesn't reach Battlemage's XMX units while OpenVINO
does. SYCL might narrow it, but it would have to beat 32.7 tps to matter.

### iGPU vs B50 (why the old numbers were misleading)

| Model | tps on ARL iGPU | tps on B50 |
|---|---|---|
| `Qwen3.5-9B-ov-awq` | 5.4 | **32.7** |
| `Qwen3.5-9B-ov-int4` | 5.5 | — |
| `gemma-4-E4B-it-int4-ov` | **43.9** | 38.2 * |

\* **The iGPU beats the B50 on Gemma-4 E4B.** Not a measurement error — 29.6
was the cold-cache reading, 38.2 the warm one, and both are below the iGPU's
43.9. A 4B-active MoE is bandwidth-bound rather than compute-bound, and
Arrow Lake's iGPU shares system DDR5 with no PCIe hop. The B50's advantage
is compute, which this workload doesn't stress. Bigger/denser models
(Qwen3.5-9B: 5.4 → 32.7) scale the way you'd expect.

Practical consequence: **there is no single "best device" setting.** If you
run a lot of small MoE models, the iGPU is competitive and leaves the B50
free for something else.

### 2026.3.0 upgrade — no measurable change

| Model | on `dev20260509` | on `2026.3.0.0` stable |
|---|---|---|
| `Qwen3.5-9B-ov-awq` | 32.7 | **32.6** |

PR #3801 (Qwen3.5 PagedAttention, merged 2026-05-12) plus three months of
Battlemage kernel work produced **no throughput change** for this model.
Useful negative result — it means the current 32.7 tps is not
kernel-limited in the way we assumed, and further openvino-genai bumps are
unlikely to move it. Speculative decoding is the remaining lever.

---

## Model verdicts

| Model | Quant | On-disk | Loads | Verdict |
|---|---|---|---|---|
| `OpenVINO/gemma-4-E4B-it-int4-ov` | INT4 | 3.6 GB | ✅ | Small, fast, official Intel conversion |
| `OpenVINO/gemma-4-E4B-it-int8-ov` | INT8 | ~7 GB | ✅ | Higher quality, still fits easily |
| `OpenVINO/LFM2-24B-A2B-int4-ov` | INT4 | 12.5 GB | ✅ | Newest official Intel 24B-class MoE |
| `circulus/Qwen3.5-9B-ov-awq` | AWQ INT4 | 6.5 GB | ✅ | **32.7 tps on B50** — good daily driver |
| `rippertnt/Qwen3.5-9B-ov-int4` | INT4 | 6.5 GB | ✅ | Equivalent to AWQ (5.5 vs 5.4 tps on iGPU) |
| `circulus/gemma-4-26B-A4B-it-ov-awq` | AWQ INT4 | 16.5 GB | ❌ | Converter emitted bad IR — `reshape_shape_inference.hpp:370` |
| `OpenVINO/gemma-4-31B-it-int4-ov` | INT4 | 18.6 GB | ❌ | Weights exceed 16 GB VRAM |
| `circulus/*-DFlash-*` | INT4 | ~0.7 GB | ⚠️ | **Drafter only — not a chat model.** See below |
| any `Qwen3.6-*` | — | — | ❌ | `Unsupported 'qwen3_6' VLM model type` (as of dev20260509) |

### AWQ vs INT4

Same file size, same VRAM, same speed. AWQ is INT4 produced by a smarter
algorithm (activation-aware scaling), typically 0.5–2% better on benchmarks.
Measured identical here (5.4 vs 5.5 tps on iGPU), so pick AWQ when both exist
but don't go hunting for it.

---

## DFlash models — do not pull

`circulus/Qwen3.5-9B-DFlash-ov-int4` and siblings are **speculative-decoding
drafters, not standalone models**. From its `config.json`:

```json
"architectures": ["DFlashDraftModel"],
"num_hidden_layers": 6,          // the drafter
"num_target_layers": 32,         // the model it drafts FOR
"dflash_config": { "block_size": 16, "target_layer_ids": [1,5,9,...,29] }
```

695 MB is the *entire* repo — the pull is not truncated. Three blockers:

1. **openvino-genai has no DFlash implementation.** OpenVINO core can execute
   the drafter's tensors (it's an ordinary 6-layer transformer IR), but
   nothing implements the block-diffusion drafting + hidden-state
   verification loop. Engine without a transmission. DFlash is currently a
   vLLM / SGLang / NVIDIA-side feature.
2. **This export is machine-specific and broken.** `auto_map` points at
   `/home/<export-user>/.local/lib/python3.12/site-packages/...` — paths that
   don't exist anywhere else.
3. **Stub tokenizer.** `tokenizer.json` is 1.63 kB; the real vocab lives at
   `z-lab/Qwen3.5-9B-DFlash` per `openvino_config.json`.

illama currently accepts these as valid models because
`_model_dir_has_artifact()` only checks for `openvino_model.xml`. Planned
guard below.

---

## Speculative decoding WORKS — 1.8× measured (2026-08-13)

| Model | Drafter | tps | Speedup |
|---|---|---|---|
| `Qwen3-1.7B-int4-ov` | none | 115.7 | — |
| `Qwen3-1.7B-int4-ov` | `Qwen3-1.7B-eagle3-ov` | **208.4** | **1.80×** |
| `Qwen3-8B-int4-ov` | none | 37.9 | — |
| `Qwen3-8B-int4-ov` | `Qwen3-8B-eagle3-ov` | **95.6** | **2.52×** |
| `Qwen3-14B` | `Qwen3-14B_eagle3` | **51.1** | ~2.4× (est. baseline ~21) |

Intel reported ~2× on CPU; the B50 GPU does better, and the speedup *grows*
with model size (1.80× at 1.7B, 2.52× at 8B). That direction makes sense:
the larger the target, the more expensive each verified token is relative to
the drafter's fixed cost, so a good acceptance rate saves proportionally
more.

**Retracts the earlier conclusion that speculative decoding was unreachable.**

### Practical consequence for this host

Verified end-to-end after a clean rebuild on openvino-genai
`2026.4.0.0.dev20260813`, identical prompt and 400-token budget for every
model, unloading between each:

| Model | Size | tps | Role |
|---|---|---|---|
| `Qwen3-1.7B-int4-ov` + eagle3 | 2.0 GB | **196.5** | instant / trivial tasks |
| `Qwen3-8B-int4-ov` + eagle3 | 5.6 GB | **96.0** | fast text |
| `Qwen3-14B` + eagle3 | 9.0 GB | **72.4** | **recommended default** - only 25% behind the 8B with far more capability |
| `gemma-4-E4B-it-int4-ov` | 6.1 GB | **42.0** | vision |
| `gemma-4-12B-it-qat-int4-ov` | 7.7 GB | **24.0** | vision, higher quality |

**Measure with a fixed token budget.** The same Qwen3-14B reads 51.1 tps via
`illama run` (which allows 4096 tokens and generated 3,764) and 72.4 tps at
400 tokens - throughput decays as the KV cache grows, so numbers are only
comparable at equal output length.
| ~~`Qwen3.5-9B-ov-awq`~~ | 6.1 GB | 32.6 | superseded - slower than every text option and than gemma-4-E4B for vision |

The eagle3 speedup is consistent across three model sizes (1.80x at 1.7B,
2.52x at 8B, ~2.4x at 14B), so it can be assumed for any text-only Qwen3
model with an AngelSlim drafter.

### Multimodal models cannot use it

Every gemma-4 OpenVINO export is VLM-shaped - `openvino_vision_embeddings_model.xml`
is present in E2B, E4B, 12B, 26B and 31B alike - so they hit the same
`inputs_embeds` wall as Qwen3.5. The gemma-4 eagle3 heads that exist
(`BCCard`, `RedHatAI`, `thoughtworks`) are published in vLLM `speculators`
format, not something optimum-intel can export.

**For multimodal models the only lever is quantisation choice.** gemma-4-E4B
at int4 is already the fast option; the qat-int4 12B is the quality option.

### The recipe that works

Export **both** models yourself with optimum-intel 2.1.0 — matched pair
matters, and the target needs no special flags:

```bash
# target - plain export
docker run --rm -v /opt/illama/models:/data -e HF_HOME=/data/.cache \
  illama-builder:latest \
  --model Qwen/Qwen3-1.7B --weight-format int4 /data/Qwen3-1.7B-int4-ov

# eagle3 drafter - needs --task and --trust-remote-code
docker run --rm -v /opt/illama/models:/data -e HF_HOME=/data/.cache \
  --entrypoint optimum-cli illama-builder:latest \
  export openvino --model AngelSlim/Qwen3-1.7B_eagle3 \
  --task text-generation-with-past --trust-remote-code \
  /data/Qwen3-1.7B-eagle3-ov
```

Then link in `models/<target>/model.json`:

```json
{ "overrides": { "draft_model": "Qwen3-1.7B-eagle3-ov",
                 "num_assistant_tokens": 4 } }
```

Drafter sources: **AngelSlim** publishes the eagle3 heads Intel documents
(`Qwen3-1.7B_eagle3`, `Qwen3-4B_eagle3`, `Qwen3-8B_eagle3`,
`Qwen3-14B_eagle3`, `Qwen3-32B_eagle3`, `Qwen3-VL-*_eagle3`). Prefer these
over third-party OpenVINO re-exports.

### Qwen3.5 speculative decoding: closed, and why

Four attempts, four distinct failures. Recording all of them because each
ruled out a different hypothesis.

| Drafter | Error | Means |
|---|---|---|
| `circulus/*-DFlash-ov-int4` | `DFlash speculative decoding is not supported for models with embeddings` | Runtime refuses DFlash on any multimodal target. Not fixable locally. |
| `circulus/*-eagle3-ov-int4` | `Model references undeclared parameters: beam_idx, attention_mask` | circulus's eagle3 IR is malformed - the graph uses those inputs without declaring them. Their bug. |
| `yuyijiong/Qwen3.5-9B-Eagle3` exported `--task text-generation-with-past` | `Port for tensor name inputs_embeds was not found` | Pipeline **constructs** (backend=continuous, no fallback) but the draft exposes `input_ids`, not `inputs_embeds`. |
| same, exported without `--task` | identical | optimum infers the same task; the source checkpoint is a text-path llama head, so no flag changes this. |

**Conclusion.** EAGLE-3 on a VLM target is *not* categorically blocked - the
pipeline constructs fine, which disproves the earlier "VLM blocks
everything" reading. What a VLM target requires is a draft head **trained
and exported for the VL structure**, so it accepts `inputs_embeds` rather
than `input_ids`. Intel's working example (optimum-intel PR #1679) uses
`AngelSlim/Qwen3-VL-4B-Instruct_eagle3` - a VL-specific checkpoint.

No such head exists publicly for Qwen3.5. Nothing to try until one appears.

**What would unblock it:** AngelSlim (or anyone) publishing a
`Qwen3.5-*-VL_eagle3` head, or circulus fixing their eagle3 export.
`illama pull --auto-draft` already knows the circulus naming convention, so
a fixed export would work immediately.

### Both Qwen3.5 and gemma-4 are blocked by optimum-intel's transformers cap

optimum-intel 2.1.0 (and main, 2.2.0.dev0) pin `transformers<5.6`. Two
current model families need something newer, so neither can be converted
in-house:

| Family | Failure | Detail |
|---|---|---|
| Qwen3.5 | `ImportError: cannot import name 'Qwen3_5DynamicCache'` | optimum-intel's patcher imports a class present in no permitted transformers. transformers 5.5.4 exposes only `['Cache', 'DynamicCache']`. |
| gemma-4 | `model type 'gemma4_unified' ... Transformers does not recognize this architecture` | `google/gemma-4-12B-it` is `gemma4_unified`; transformers 5.5.4 has no such model type. |

Both unblock the same way - optimum-intel raising its transformers ceiling.
There is no local workaround: forcing a newer transformers violates
optimum-intel's own pin and breaks its exporters.

Practical consequence: for these families, use **pre-converted** OpenVINO
exports (the `OpenVINO/` and `circulus/` orgs both publish them) and accept
that speculative decoding is unavailable, since drafters cannot be paired
with a multimodal target anyway.

### Qwen3.5 in-house export is blocked upstream

`optimum-cli export openvino --model Qwen/Qwen3.5-9B` fails with

    ImportError: cannot import name 'Qwen3_5DynamicCache' from
    'transformers.models.qwen3_5.modeling_qwen3_5'

optimum-intel 2.1.0 *and* main (2.2.0.dev0) both pin `transformers<5.6` and
both import `Qwen3_5DynamicCache` in `model_patcher.py`. transformers 5.5.4 -
the highest version that pin allows - exposes only `['Cache',
'DynamicCache']` in that module. The referenced class exists in **no
transformers version optimum-intel permits**, so this is an upstream bug,
not a local misconfiguration. Nothing to work around here; use the Qwen3
family instead until it is fixed.

### Gotchas found the hard way

- **Prefix caching breaks speculative decoding.** Both enabled →
  `Check 'content_length <= prompt_ids.size() + m_generated_ids.size()'
  failed at sequence_group.cpp:32` at first generate(). illama now disables
  prefix caching automatically whenever a draft model is linked.
- **The target needs no special export.** An earlier conclusion here — that
  EAGLE-3 required a target carrying aux hidden states — was wrong. Intel's
  optimum-intel PR #1588 exports the main model plain.
- **Matched pair matters.** The failed attempt used `circulus/Qwen3.5-9B-ov-awq`
  (third-party AWQ re-export) as target against a `circulus` eagle3 head and
  got `unregistered_parameters` at `model.cpp:264`. Exporting both halves
  from the same stock checkpoint avoids this. Worth re-testing Qwen3.5 now
  that the prefix-caching bug is also known.
- **Builder dependencies:** `datasets` (data-aware INT4 calibration is the
  2.1.0 default), plus `einops` / `sentencepiece` / `protobuf` for
  `--trust-remote-code` modeling files. Missing any of them fails the export
  late, after the trace has already run.
- **Don't install openvino-genai in the builder.** It's a runtime, not an
  export dependency, and the runner's nightly version has no PyPI equivalent.

---

## Acceleration attempts — measured results (2026-08-13)

Everything below was measured on the real B50 with the same prompt
(`"Write a 300-word story."`, `max_tokens=400`, `temperature=0.7`).

| Change | tps | Verdict |
|---|---|---|
| baseline, stateful/SDPA, openvino-genai 2026.3.0.0 | **32.6** | — |
| runtime bump to 2026.4.0.0-3359 | 32.6 | no change |
| continuous batching (PagedAttention), cache 2 GB | 32.2 | **~1% slower** |
| + EAGLE-3 drafter (`circulus/Qwen3.5-9B-eagle3-ov-int4`) | 32.1 | **failed**, fell back |
| + classic drafter (`rippertnt/Qwen3.5-0.8B-ov-int4`) | 31.1 | **failed**, fell back |

**Conclusion at the time: 32.6 tps looked like the ceiling.** That was
premature - see "Speculative decoding WORKS" above. Every failure in this
table had prefix caching enabled, which we later found breaks speculative
decoding outright. The Qwen3.5 rows are worth re-running now that both the
prefix-caching conflict and the matched-pair requirement are understood.

### Why PagedAttention didn't help

Continuous batching optimises *throughput under concurrency* — many
simultaneous requests amortising the paging bookkeeping. illama is a
single-user, one-request-at-a-time server, so there is nothing to amortise
against and the bookkeeping is pure overhead. Keep `pipeline_backend: auto`
anyway: the ~1% cost is noise, and it is the only backend where speculative
decoding could ever work.

Note the cache-size quirk: a bare `VLMPipeline(..., scheduler_config=s)` probe
rejected 2/4/6 GB and needed 8 GB, but illama succeeds at 2 GB. The difference
is `enable_prefix_caching=True`, which changes the block accounting.

### Why speculative decoding is unreachable for Qwen3.5

Two independent blockers, both confirmed by error:

1. **EAGLE-3 needs a purpose-built target export.** Its config declares
   `eagle_aux_hidden_state_layer_ids: [1, 15, 28]` and
   `use_aux_hidden_state: true` — the draft head consumes *intermediate
   hidden states* of the target. A stock export doesn't surface those, so
   construction dies with
   `Check 'unregistered_parameters.str().empty()' failed at
   src/core/src/model.cpp:264`. `circulus` published drafters but no matching
   aux-enabled targets. DFlash has the same dependency
   (`target_layer_ids: [1,5,9,...,29]`) and would fail identically.

2. **`ov_genai.draft_model()` requires a text-only IR.** It loads
   `openvino_model.xml` specifically; a VLM-shaped export fails with
   `Check 'util::directory_exists(path) || util::file_exists(path)' failed at
   src/frontends/common/src/frontend.cpp:117`.

   This is fatal for the whole family: **Qwen3.5 is natively multimodal, so
   every OpenVINO conversion of every size is VLM-shaped.** There is no
   compatible drafter — not the 0.8B, not the 2B, none of them.

illama rejects VLM-shaped drafters up front with an explanatory warning
rather than letting them fail opaquely in the frontend.

**What would unblock it:** a text-only (`--task text-generation-with-past`)
Qwen3.5 export for both target and drafter, converted in-house with
optimum-intel 2.1.0. Untested. Would also lose vision capability on the
target.

---

## Speculative decoding — the supported path

openvino-genai **does** implement classic draft-model speculative decoding:

```python
draft = ov_genai.draft_model(draft_dir, "GPU")
sched = ov_genai.SchedulerConfig(); sched.cache_size = 2   # GB
pipe = ov_genai.LLMPipeline(target_dir, "GPU",
                            scheduler_config=sched, draft_model=draft)
cfg.num_assistant_tokens = 5    # usual sweet spot
```

Hard requirement: **draft and target must share the same tokenizer/vocab.**

Official Intel FastDraft pairs exist only for three families:

| Draft | Target family |
|---|---|
| `OpenVINO/Phi-4-mini-FastDraft-120M-int8-ov` | Phi-4 |
| `OpenVINO/Phi-3-mini-FastDraft-50M-int8-ov` | Phi-3-mini |
| `OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov` | Llama-3.1-8B |

**No Qwen drafters, and no EAGLE-3 exports** in the `OpenVINO/*` org, despite
2026.3 adding EAGLE-3 runtime support.

**Qwen workaround:** the Qwen3.5 size ladder shares one tokenizer, so a small
sibling can draft for a big one — pure OpenVINO, no DFlash, no llama.cpp:

```
Target: Qwen3.5-9B-ov-awq
Draft:  rippertnt/Qwen3.5-0.8B-ov-int4    (~0.5 GB)
   or:  circulus/Qwen3.5-2B-ov-int4       (~1.5 GB)
```

Expected 1.4–2× on decode when acceptance rate is good — unproven on this
hardware, worth measuring.

---

## Version status (2026-08-11)

| Package | Pinned in repo | Running in container | Latest stable |
|---|---|---|---|
| openvino | nightly `--pre`, unpinned | **2026.3.0**-22451 | 2026.3.0 |
| openvino-genai | nightly `--pre`, unpinned | **2026.3.0.0**-3277 | 2026.3.0.0 |
| optimum-intel | `2.0.0` | (builder only) | **2.1.0** (2026-08-05) |
| Ollama | — | v0.32.9 | bundles llama.cpp **b10353** |
| LM Studio | — | — | bundles llama.cpp **b10252** (≈100 builds behind Ollama) |

Verify what's actually running:

```bash
docker exec illama-service /opt/illama-venv/bin/python -c \
  'import openvino_genai as g, openvino as ov; print("genai:", g.__version__); print("openvino:", ov.__version__)'
```

The runner already runs **2026.3.0 stable** — the unpinned `--pre` install
resolved to it once 2026.3 shipped. That's convenient but not reproducible:
the next rebuild will silently pull whatever is newest at that moment.
Pinning explicit versions is still worth doing for build reproducibility,
but it is **not** a performance change (see the upgrade table above).

`optimum-intel 2.1.0` matters only for the builder image — its
`transformers<5.6,>=4.51` constraint unblocks converting Qwen3.5 in-house
instead of relying on community IRs.

---

## Waiting on

Deliberate pause. Nothing here blocks daily use — performance is stable and
good. Revisit when any of these move:

- [ ] **Mesa / Vulkan `xe` maturity** — may close the 2× gap vs openvino-genai
- [x] ~~openvino-genai Battlemage kernel tuning~~ — 2026.3.0 measured, **no
      change** (32.7 → 32.6 tps). Further runtime bumps unlikely to help.
- [ ] **Qwen-family drafters for OpenVINO** — none exist yet; now the main
      remaining lever for Qwen3.5 throughput
- [ ] **EAGLE-3 exports in the `OpenVINO/*` org** — runtime support shipped in 2026.3, models haven't
- [ ] **DFlash in openvino-genai** — not on any roadmap; don't hold your breath
- [ ] **`qwen3_6` model type support** — currently unsupported
- [ ] **llama.cpp SYCL** — would need to beat 32.7 tps to be worth a sidecar

---

## Planned illama changes

| Version | Change | Status |
|---|---|---|
| 2.5.0 | Pin runner + builder to stable OpenVINO 2026.3.0 / optimum-intel 2.1.0 | container already runs 2026.3.0 via unpinned `--pre`; pin for reproducibility only, no perf gain |
| 2.5.0 | `_validate_pulled_model()` — reject drafters, warn on non-portable `auto_map`, stub tokenizers, and size/param mismatch | planned |
| 2.6.0 | Speculative decoding: `model.json` `draft_model` field, `illama link/unlink`, `--draft` on pull | planned |
| 2.4.1 | `finish_chat()` in `_vlm_generate` — fixes VLM state leak across requests | pending |
| 2.4.1 | `reasoning_effort` field to disable thinking mode per-request | pending |
| 2.4.1 | Extend `_LOAD_ERROR_USER_PATTERNS` with shape-inference errors | pending |

---

## Operational reference

### Disk usage on this host

```
/opt/illama/cache    19 GB    GPU compile artifacts — safe to clear, regenerates
/opt/illama/models   41 GB    the registry
```

Clearing the cache costs ~10 s per model on next load:
```bash
rm -rf /opt/illama/cache/*
```

### Backup (config only — models are re-pullable and untouched by rebuilds)

```bash
TS=$(date +%Y%m%d-%H%M)
ls -1 /opt/illama/models > /root/illama-models-$TS.txt
tar czf /root/illama-cfg-$TS.tar.gz -C /opt illama \
    --exclude='illama/models' --exclude='illama/cache'
docker tag illama-runner:latest illama-runner:pre-upgrade
```

`docker build` never touches `/opt/illama/models` — it's a host directory
bind-mounted read-only into the container.

### Ports

The default Illama port is `11434`. These notes use `11435` so Illama can sit
beside another server. Other workloads share the B50's16GB VRAM; stop them
before measuring Illama. Image generation has its own
[ComfyUI repository](https://github.com/mkhomutskyi/comfyui-intel-arc).
Private operator notes are not published.

```bash
curl -s -X POST http://localhost:11435/v1/unload
```

### Benchmark recipe

```bash
# illama
curl -s -X POST http://localhost:11435/v1/unload >/dev/null
curl -s http://localhost:11435/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<name>","messages":[{"role":"user","content":"Write a 300-word story."}],
       "max_tokens":400,"temperature":0.7}' \
  | python3 -m json.tool | grep -E 'tokens_per_second|completion_tokens'

# Ollama (think:false matters — Qwen3.5 otherwise spends the budget reasoning)
curl -s http://localhost:11434/api/generate \
  -d '{"model":"qwen3.5:9b","prompt":"Write a 300-word story.","stream":false,
       "think":false,"options":{"num_predict":400}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(round(d['eval_count']/(d['eval_duration']/1e9),1),'tps')"
```

### Gotchas that cost time

- `illama logs` runs `docker logs -f` — piping it to `tail` hangs. Use
  `docker logs --tail N illama-service`.
- `docker exec ... python - <<'PY'` needs `-i` or stdin is dropped silently.
- f-strings with escaped quotes inside `{}` (`f"{d[\"x\"]}"`) fail on older
  Python. Use `python3 -m json.tool | grep` or a heredoc instead.
- `curl -d` without `-H 'Content-Type: application/json'` gets form-encoded
  and fails Pydantic validation with a confusing root-level error.

