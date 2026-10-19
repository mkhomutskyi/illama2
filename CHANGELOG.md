# Changelog

All notable changes to this project will be documented in this file.

## Publication maintenance — 2026-10-19

- Aligned direct builder/runner build defaults and the deployment example with
  the current Illama 2.6.0 release; stable and optional nightly runtime pins are
  unchanged.
- Extended the public-data check to personal email addresses and commit metadata.
- Retained the latest dated B50/ComfyUI measurements, reproducible setup guides
  and third-party license notices. Private operator records remain excluded.

## Post-2.6.0 maintenance — 2026-10-18

- Added a B50 quickstart, optional pinned OpenVINO 2026.5 nightly profile, and
  dated three-model inference measurements with exact protocol and limitations.
- Recorded the independent ComfyUI XPU configuration, existing-weight image
  benchmarks, BF16 quality diagnostic, and INT8 loader incompatibility.
- Added a portable ComfyUI recipe; a network-disabled, non-privileged container
  passed Intel acceptance and generated a 1024 FP8 image on the B50.
- Fixed the CLI echo of configured API keys, existing lint failures and missing
  CI test dependencies. Added a bounded credential-pattern check in CI.
- Clarified that Docker health status alone does not restart a live unhealthy
  container. Private operator details and raw evidence remain excluded.

## 2.4.0 - 2026-05-10

### Added

- **Tool calling and reasoning extraction** in the chat API.
  - The runner now parses Hermes-3 style `<tool_call>{...}</tool_call>` blocks
    out of the model output and surfaces them as OpenAI-shaped `tool_calls[]`
    on `/v1/chat/completions` and Ollama-shaped `message.tool_calls[]` on
    `/api/chat`. Streaming on both surfaces emits one structured chunk per
    tool call as soon as `</tool_call>` closes.
  - `<think>...</think>` blocks (DeepSeek-R1, Qwen 3 with `enable_thinking`,
    GPT-OSS) are extracted and surfaced as `reasoning_content` on the OpenAI
    surface and `thinking` on the Ollama surface.
  - `tools` and `tool_choice` accepted on `ChatCompletionRequest`; `tools`
    accepted on `OllamaChatRequest`. The list is passed to the tokenizer's
    chat template (`apply_chat_template(messages, tools=...)`), which Hermes-
    trained checkpoints render as a tool schema in the system prompt.
  - `finish_reason` switches to `"tool_calls"` whenever the model emitted
    one or more tool calls, matching OpenAI semantics; client SDKs use this
    flag to decide whether to dispatch the tool.
  - This is what OpenVINO Model Server does with `--tool_parser hermes3
    --reasoning_parser qwen3`. We do it in 270 lines of Python so we don't
    take OVMS as a dependency.

### Module layout

- New `runner/_parsers.py` (`StreamingParser`, `parse_full_output`,
  `to_ollama_tool_call`). Stateful, hold-back-buffered, safe across
  arbitrary chunk boundaries. ~22 unit tests in `tests/test_parsers.py`.

## Unreleased

### Documentation

- Pinned the published Dockerfiles to stable OpenVINO **2026.4.1**, GenAI **2026.4.1.0**, tokenizers **2026.4.1.0**, and optimum-intel **2.2.0** (transformers **4.57.6** on the builder). Dropped the nightly index.
- Documented third-party stacks and the Apache-2.0 license in `LICENSE`, `NOTICE`, and the README.

### Fixed

- **`GET /api/ps` `expires_at` now reports the real unload time.** It
  was always returning `now()` (the timestamp of the request itself),
  which meant Ollama-style dashboards showed every loaded model as
  "expires in 0 seconds" regardless of the actual idle window. The
  field now reports `last_request_time + idle_timeout`. When
  `idle_timeout` is `0` (model pinned to VRAM forever) it returns the
  `9999-12-31T23:59:59.000Z` sentinel that stock Ollama emits for
  `keep_alive=-1`, which existing dashboard clients render as
  "Forever".
- **Idle watchdog poll interval now scales with the idle timeout.**
  Previously hard-coded to `time.sleep(60)`, which meant a 15s idle
  timeout could drift to 75s+ before the model actually unloaded. The
  new `_watchdog_sleep_interval` helper picks a poll period that
  re-checks `last_request_time` ~4x per idle window, floored at 1s and
  capped at 30s, so a 15s timeout fires within ~20s wallclock and a
  1-hour timeout still wakes the thread only twice an hour.

### Changed

- **Runner bumped to a pinned OpenVINO 2026.2 nightly to enable Gemma-4 VLM support today.**
  `runner/Dockerfile` now installs `openvino==2026.2.0.dev20260501` and
  `openvino-genai==2026.2.0.0.dev20260501` from Intel's nightly index
  (`https://storage.openvinotoolkit.org/simple/wheels/nightly/`) — both
  built 2026-05-01, the day after [PR #3644](https://github.com/openvinotoolkit/openvino.genai/pull/3644)
  (Gemma-4 / `gemma4` VLM model type) merged on 2026-04-30. Stable
  `2026.1.x` does not recognise `gemma4` and fails fast with
  `Unsupported 'gemma4' VLM model type`. **This pin is temporary** and
  will be replaced with stable `openvino==2026.2.0` /
  `openvino-genai==2026.2.0.0` once Intel ships the 2026.2 release on
  PyPI (~late May / early June 2026); the dev wheels are pinned to a
  specific build so that `illama build` is reproducible until then.
  The `builder/` image stays on stable `2026.1.0` — `optimum-intel`
  HF→OV conversion of Gemma-4 still requires extra work, so the
  supported path is `illama pull OpenVINO/gemma-4-E4B-it-int4-ov`
  (pre-converted from the `OpenVINO/` org on Hugging Face). VERSION
  bumped to `2.3.0`.
- **Model-load failures are now classified before triggering a restart.**
  When OpenVINO GenAI rejects a model with a config-level error
  (`Unsupported 'gemma4' VLM model type`, missing tokenizer, etc.) the
  runner now returns a clean **HTTP 400** with code
  `model_incompatible_with_runtime` and the actual reason in the error
  envelope, and **stays alive**. Previously every load failure went
  through the Phase 2 graceful-exit + Docker restart path, which is
  pointless for a runtime-doesn't-know-this-architecture error
  (restarting the GPU won't add new model types to the C++ switch
  statement) and recycled the container for nothing. GPU-state errors
  still take the restart path.
- **OpenVINO error text is surfaced to the API client.** All three
  load-failure paths (400 user error, 500 return_error, 503 restarting)
  now include a build-path-stripped, single-line summary of the
  underlying exception, so operators no longer need to read
  `docker logs` to find out why a load failed. Example:
  `Model 'gemma-4-E4B-it-int4-ov' cannot be loaded by this OpenVINO
  runtime: Unsupported 'gemma4' VLM model type. ... Try a different
  model or upgrade openvino-genai.`

### Added

- **Ready-to-go Docker Compose deployment under [`deploy/`](deploy/).**
  [`deploy/docker-compose.yml`](deploy/docker-compose.yml) mirrors `illama
  serve` exactly (same image, same `--device /dev/dri`, same `--group-add`
  for video/render, same volumes, same healthcheck), so `docker compose
  up -d` from `/opt/illama/deploy` is functionally equivalent to running
  the bash CLI. Every knob is parametrised in
  [`deploy/.env.example`](deploy/.env.example) — port, device, idle
  timeout, API key, host video/render GIDs, image tag, SSRF guard
  override. [`deploy/illama-runner.service`](deploy/illama-runner.service)
  wraps compose in a systemd unit for auto-start on boot. setup.sh now
  installs all three files into `$ILLAMA_HOME/deploy/` and seeds a
  `.env` copy on first install.
- **`illama pull` accepts a full HuggingFace URL.** The canonical
  `org/name` form still works, but the input is now normalised first, so
  any of these reach `snapshot_download` as `OpenVINO/Phi-4-int4-ov`:
  `https://huggingface.co/OpenVINO/Phi-4-int4-ov`,
  `https://huggingface.co/OpenVINO/Phi-4-int4-ov/tree/main`,
  `https://huggingface.co/OpenVINO/Phi-4-int4-ov/blob/main/README.md`,
  `hf://OpenVINO/Phi-4-int4-ov`. Trailing slashes, query strings, and
  fragments are stripped. Lets users paste straight from the browser
  address bar without hand-editing the URL into a repo id.

### Changed

- **`setup.sh` no longer silently overwrites a customised live config.**
  On upgrade, if `$ILLAMA_HOME/config/server.json` differs from the
  shipped one, the new defaults are written to `server.json.new`
  alongside the live file (with a `diff -u` hint), so operator
  customisations survive `git pull && ./setup.sh`. Fresh installs still
  get the shipped defaults outright.
- **Shipped `config/server.json` no longer pins `max_input_tokens` to 0.**
  Removing the key from the shipped config lets the Phase 2 code default
  (8192 input tokens) take effect on fresh installs, matching the
  documented behaviour. Existing installs are unaffected — their live
  config is preserved per the previous bullet, so an explicit
  `"max_input_tokens": 0` continues to mean "disabled".
- **`runner/server.py` is no longer a single 3.1k-line file.** Pure helpers
  (`_b64decode_lenient`, `_DATA_URL_RE`, `is_auth_exempt_path`,
  `make_openai_error`, `make_stream_error_event`, `sse_encode`,
  `build_chat_stream_chunk`, `build_completion_stream_chunk`,
  `build_usage_payload`, `_parse_idle_timeout`, `_get_config_int`,
  `_get_config_bool`, `normalize_load_failure_policy`) moved to
  [`runner/_helpers.py`](runner/_helpers.py). Pydantic request/response
  models (`Message`, `ChatCompletionRequest`, `CompletionRequest`,
  `ImageURL`, `ContentPart{Text,Image}`, `StreamOptions`, `ModelInfo`,
  `ModelsResponse`) moved to [`runner/_schemas.py`](runner/_schemas.py).
  `server.py` re-imports every public name so existing callers and tests
  (`server.Message`, `server._b64decode_lenient`, ...) keep working
  unchanged. server.py drops to ~2.95k lines and the new modules can be
  unit-tested without touching OpenVINO or the FastAPI app — see
  `tests/test_helpers_module.py` for the standalone-import sanity suite.
  The Dockerfile now does `COPY *.py /app/` and `setup.sh` copies
  `runner/*.py` so all three files ship together.

### Added

- **GitHub Actions CI (`.github/workflows/ci.yml`).** Three jobs run on
  every push to `master` and on every PR: `shellcheck` (errors-only on
  `illama` and `setup.sh`), `ruff` (lint `runner/`, `scripts/`, `tests/`,
  pinned to ruff 0.7.4), and `pytest` (the 60-test smoke suite). The
  workflow deliberately does NOT install OpenVINO, openvino-genai, PIL,
  numpy, or requests — `tests/conftest.py` stubs them into `sys.modules`,
  so the suite stays fast (~0.25s) and runs on stock Linux runners. End-
  to-end tests that need the real Intel runtime continue to live on the
  deployment host (`scripts/benchmark_api.py`, `illama bench`).
- **`tests/` pytest smoke suite (60 tests, ~0.25s) for the pure-Python
  helpers in `runner/server.py`.** Covers the SSRF guard
  (`_assert_image_url_is_safe`), whitespace-tolerant base64 decoding
  (`_b64decode_lenient`, hoisted from inside `_decode_image_to_tensor` so
  it is testable), multimodal content flattening (`_message_text`),
  config parsing (`_get_config_bool`, `_parse_idle_timeout`), the
  data-URL regex, the auth-exempt path set, the OpenAI error shape, and
  version resolution. CI does not need OpenVINO, PIL, numpy, or
  requests installed - `tests/conftest.py` injects minimal stubs into
  `sys.modules` before importing the server module.
- **`illama gc`** reclaims disk by pruning the HuggingFace download cache
  (`${REGISTRY}/.cache`) and any model directories under `${REGISTRY}` that
  are missing the OpenVINO artifacts (a partial pull/conversion left a
  broken tree behind). Skips the currently-loaded model. Flags:
  `--dry-run` (preview only), `--force` (skip prompt), `--keep-cache`.
- **`illama doctor`** runs a battery of read-only host/runtime health
  checks — docker daemon, /dev/dri presence, video/render group
  membership, registry layout, runner image presence, container running,
  `/illama/health` reachable, broken model directories — and prints a
  pass/warn/fail summary with concrete next-step advice. Useful as a
  standalone diagnostic outside `setup.sh`.
- **Time-to-first-token (TTFT) metrics.** Both streaming workers
  (OpenAI SSE and Ollama NDJSON) now capture the wall-clock from request
  start to the first emitted token and feed it through `_record_success`.
  `/illama/stats` exposes `avg_ttft_seconds`, `ttft_samples`, and
  `overall_tokens_per_second_after_first` (decode TPS, excluding TTFT)
  globally and per-model. `/metrics` adds
  `illama_total_first_token_time_seconds`,
  `illama_total_first_token_samples`,
  `illama_model_first_token_time_seconds_total{model=…}`, and
  `illama_model_first_token_samples_total{model=…}`. Lets ops dashboards
  separate prompt-processing latency from per-token decode speed when
  comparing models.
- **SSRF guard for VLM image_url fetches.** `_assert_image_url_is_safe`
  resolves the host's IPv4/v6 addresses and rejects URLs that point at
  loopback (`127.0.0.0/8`), link-local (`169.254.0.0/16`, including the
  cloud metadata service `169.254.169.254`), private RFC1918 ranges,
  multicast, and reserved blocks before any HTTP round trip. Operators in
  trusted private networks can disable the check with
  `ILLAMA_ALLOW_PRIVATE_IMAGE_URLS=1`.
- **Single source of truth for the project version (`VERSION` file).** Read at
  the top of [runner/server.py](runner/server.py) (env > file > fallback)
  and at startup of [illama](illama) (installed-path file > repo-path file >
  "unknown"). The `VERSION` file is copied into `$ILLAMA_HOME/` by
  `setup.sh` and into the runner image build context. Bumping a release is
  now a single edit to `VERSION` plus a `CHANGELOG.md` entry; the FastAPI
  metadata, `/`, `/api/version`, and `illama version` all pick it up.
- **Pinned image rollback tags.** Both `setup.sh` and the lazy builder build
  in `illama pull` now emit two tags per image: `:latest` (the rolling
  pointer used by `illama serve` and `cmd_pull`) and `:$ILLAMA_VERSION`
  (an immutable rollback target). After a bad upgrade an operator can
  revert with `docker tag illama-runner:2.2.0 illama-runner:latest`
  instead of rebuilding from a previous git ref.
- **`.dockerignore` for both build contexts** (`runner/.dockerignore`,
  `builder/.dockerignore`) so stray editor temp files, `__pycache__`, and
  `.git` are not pulled into `docker build` contexts. Makes builds smaller,
  faster, and more deterministic.
- **`ARG ILLAMA_VERSION` build argument** in both Dockerfiles, baked into
  `ENV ILLAMA_VERSION` and into OCI image labels
  (`org.opencontainers.image.version`).
- **`illama search` for browsing the HuggingFace OpenVINO model index.**
  Queries `https://huggingface.co/api/models?filter=openvino` (passes
  `HF_TOKEN` as a Bearer token if set), prints a numbered table of
  trending repos with download / like counts and last-modified age, and
  offers a one-step pull prompt that dispatches to the existing
  `cmd_pull` (so the Phase E pre-converted-vs-convert routing keeps
  working). Flags: `--vlm` (filters to `image-text-to-text`),
  `--limit N` (default 20, capped at 100 by HF), and
  `--sort trending|downloads|likes` (default trending). The CLI maps
  these to the API's actual sort fields - `trendingScore`, `downloads`,
  `likes` with `direction=-1`.
- **Vision-Language Model (VLM) image input.** Both the OpenAI and the Ollama
  API surfaces now accept images on chat / generate requests, and OpenVINO
  GenAI's `VLMPipeline` is invoked correctly with `generation_config`,
  `images`, and `streamer` plumbed through. The `Message.content` schema
  accepts the multimodal content-array form (`[{"type":"text",...},
  {"type":"image_url","image_url":{"url":"data:..."|"https://..."}}]`)
  alongside the legacy plain-string form. The Ollama `images` field on chat
  messages and generate requests is also supported (raw base64 strings).
  Image sources can be `data:image/...;base64,...` URLs, `http(s)://` URLs
  fetched with a 10s timeout and 20 MB size cap, or raw base64. Decoding
  goes via Pillow into `numpy` (HWC, uint8, RGB) and then `ov.Tensor`,
  which is what OpenVINO GenAI expects.
- New runtime deps in [runner/Dockerfile](runner/Dockerfile): `pillow==11.0.0`,
  `requests==2.32.3`, `numpy==2.1.3`.

### Changed

- **Both Dockerfiles now install into a dedicated venv at
  `/opt/illama-venv` instead of `pip3 install --break-system-packages`.**
  Ubuntu 24.04 marks the system Python externally-managed; the venv
  approach is the supported path. PATH is set so the existing
  `python3 server.py` CMD and `optimum-cli` ENTRYPOINT keep working with
  no other changes.
- **`MAX_INPUT_TOKENS` default is now 8192 (was 0/disabled).** Previously,
  an unconfigured deployment had no input-token cap, so a single client
  could send a 200K-character prompt and pin the GPU. The new default fits
  every chat model we ship (all support >=8K context). Operators with
  larger contexts override via `config/server.json`
  (`"max_input_tokens": 32768`) or per-model `model.json`. Setting `0`
  still disables the check.
- **`/v1/unload` no longer holds `model_lock` during the 2s GPU-driver
  reclaim sleep.** `unload_current_model_locked` now returns a flag
  indicating the caller should sleep AFTER releasing the lock; the
  manual-unload endpoint reads that flag, releases the lock, then sleeps.
  Concurrent requests that just want to read state (e.g. `/health`) no
  longer block for two seconds during a manual unload.
- **`os._exit(1)` on a fatal GPU load failure is now scheduled with a
  short grace period instead of called inline.** The current request now
  receives a clean OpenAI-style 503 with `code=model_load_failed_restarting`
  before the process exits and Docker reaps the container. Tunable via
  `ILLAMA_FATAL_EXIT_GRACE_SECONDS` (default 1.5s).

### Fixed

- **`illama run` non-streaming was always returning "Error: Invalid JSON
  response".** The `make_request` helper piped the curl response into
  `python3 - <<'PY' ... PY`, but bash's heredoc redirection wins over the
  pipe. `python3 -` then read the heredoc as its script (correctly) and
  `json.load(sys.stdin)` saw an empty stdin (incorrectly). The response is
  now passed via the `ILLAMA_API_RESPONSE` env var so the heredoc and the
  payload no longer fight over stdin. Pre-existing bug, unrelated to the
  recent VLM/search work, but caught while reviewing.
- **VLM image input now tolerates whitespace in base64 payloads.**
  `_decode_image_to_tensor` was calling `base64.b64decode(..., validate=True)`,
  which rejects line-wrapped base64 (`base64 < img.png` wraps at 76 cols by
  default, and many clients copy-paste with newlines). Now it strips ASCII
  whitespace before decoding, matching what every other OpenAI/Ollama-
  compatible server does.
- **VLM `GenerationConfig` is no longer dropped on the floor.** The four
  generation call sites (OpenAI streaming + non-streaming, Ollama streaming
  + non-streaming) now route through a single `_vlm_generate` helper that
  forwards `generation_config`, `images`, and `streamer` to
  `VLMPipeline.generate`. Previously the VLM path called
  `pipeline.generate(prompt)` with none of the request's decoding settings.
- **`/api/generate` no longer hardcodes `is_vlm=False`.** It now respects
  the loaded model's actual type and accepts image input via the `images`
  field on the request body.



- **`illama pull` no longer leaves corrupt half-trees on retry.** A pull that
  errored mid-download used to leave `${REGISTRY}/<model>/` behind without a
  `openvino_model.xml`. The next pull would skip the overwrite-prompt (because
  the existence check required both directory AND model file) and let the new
  download interleave into the half-tree. The pull command now detects partial
  directories at the start of the run and cleans them up automatically, and
  cleans up after a failed download too.
- **`illama pull` validates the repo id and the derived model name** before
  doing any work, so typos like `pull /foo` or `pull foo bar` produce a clear
  error instead of a confusing Docker mount failure or silent path traversal.
- **`illama pull` heuristic for pre-converted repos no longer false-positives
  on `phi-3-overview`-style names.** The previous `*"-ov"*` substring match
  flagged any repo containing `-ov`. The check now requires `-ov` / `_ov`
  to sit on a token boundary (or the literal substring `openvino`, or an
  `OpenVINO/` prefix), so `microsoft/phi-3-overview` correctly routes to
  the conversion path.
- **`illama pull` for pre-converted models reuses the builder image when
  available**, instead of always spinning up a fresh `python:3.12-slim` and
  running `pip install -q huggingface_hub` (which failed offline and added
  tens of seconds of network/setup time per pull). When the builder image
  isn't cached locally, the original ephemeral path is still used so users
  who only ever pull pre-converted models don't need to materialise the
  ~5 GB builder.
- **`illama rm` now rejects path-traversal-style model names** (`../foo`,
  `/etc/passwd`, etc.) up front. The function previously did
  `sudo rm -rf "${REGISTRY}/${model}"` without validating `$model`, which
  meant a name like `..` would trigger `sudo rm -rf` against the registry
  root.
- **`illama rm` refuses to delete a model that is currently loaded by the
  running server** unless `--force` is passed. With `--force`, the model is
  unloaded via `POST /v1/unload` first so the runtime stops holding open
  files we're about to delete and `/v1/models` doesn't keep advertising a
  model that no longer exists on disk.

- **`/illama/stats` and `/metrics` no longer hold `stats_lock` across disk I/O.**
  Both endpoints used to wrap the entire handler body in `with stats_lock:`,
  which meant every Prometheus scrape and every dashboard refresh would
  serialize against the inference fast path's `_record_success` /
  `_record_error` calls. They now go through a `_snapshot_inference_stats()`
  helper that copies the totals and per-model dict under the lock and
  returns; the handlers then read `current_model_name` /
  `active_generations` under `model_lock` (where they actually live) and
  call `get_effective_request_limits` (which reads `model.json` from disk)
  with no locks held.
- **`/metrics` per-model cardinality is now bounded.** Previously every
  model name ever loaded became a permanent set of `illama_model_*_total`
  labels, growing Prometheus cardinality unboundedly. The endpoint now
  emits at most `MAX_METRICS_PER_MODEL_LABELS=50` rows, sorted by recent
  activity, plus two new gauges so operators can see the truncation:
  `illama_metrics_models_total` (records held) and
  `illama_metrics_models_emitted` (rows shipped this scrape). Records are
  still kept in `per_model_stats` for `/illama/stats` inspection.
- **Stop-sequence assignment failure now logs a debug message** instead of
  silently swallowing the exception in `apply_stop_sequences`. The
  fall-through behavior is unchanged.

- **Streaming worker no longer deadlocks when a client disconnects mid-response.**
  Both the OpenAI SSE path (`/v1/chat/completions`, `/v1/completions`) and the
  Ollama NDJSON path (`/api/chat`, `/api/generate`) used a bounded
  `queue.Queue(maxsize=64)` and a blocking `events.put(...)` from the worker
  thread. If the async consumer returned (e.g. browser tab closed) while the
  queue was full, the worker would block on `put()` forever, leak its pipeline
  reference, and leave `active_generations` permanently bumped, which in turn
  disabled idle unload and left the GPU effectively pinned. The producer now
  uses a cancel-event-aware `_enqueue` helper with timeout-based puts and a
  hard 30s deadline; the streamer returns `STOP` to OpenVINO GenAI on the next
  token after the consumer cancels, and the worker's `finally` block always
  calls `end_generation()`.
- **Ollama `/api/chat` and `/api/generate` now honor `allow_stream`.** The
  validation helper that 400s `stream=true` when the server config has
  `allow_stream: false` was previously only invoked from `/v1/*`. Ollama
  clients silently bypassed it because their default is `stream=True`.
- **Model switch no longer holds `model_lock` for an extra 2 seconds.**
  `unload_current_model_locked` had a defensive 2s sleep "for the GPU driver
  to reclaim" that was kept while the lock was held, blocking every other
  request waiting on the lock. The model switch path now skips that sleep
  because the immediately-following pipeline construction already takes
  several seconds and gives the driver plenty of reclaim time. Manual unload
  (`POST /v1/unload`) keeps the sleep, since no follow-up load is expected.
- **`/health` and `/illama/health` now read `current_model_name` and
  `active_generations` under `model_lock`** to avoid the brief inconsistent
  reads that were possible during a model switch.

### Documentation

- README "Components" section corrected: removed the stale "~300 lines of Python" claim
  (the runner is ~2.4k lines), expanded the endpoints summary to cover the full
  OpenAI `/v1/*` and Ollama `/api/*` surfaces shipped in 2.2.0, and removed the
  duplicate CLI command table that disagreed with the canonical "Commands"
  section further down.
- "OpenVINO Dependencies" table updated to match the actually-pinned versions in
  the Dockerfiles: `openvino` 2026.1.0, `openvino-genai` 2026.1.0.0,
  `transformers` 4.57.6, `huggingface_hub` 0.35.3, `accelerate` 1.13.0,
  `fastapi` 0.136.1, `uvicorn[standard]` 0.46.0, `pydantic` 2.13.3.
- New "Known Limitations" section documenting VLM `GenerationConfig` drop,
  `/v1/completions` + `/api/generate` VLM assumption, `response_format` and
  `presence_penalty` pass-through behavior, and unbounded `/metrics` cardinality.
- New "Backend Alternatives (May 2026 snapshot)" section comparing OpenVINO
  GenAI 2026.1, `llama.cpp` SYCL/Vulkan, OpenVINO's new `llama.cpp` backend,
  IPEX-LLM (archived by Intel on 2026-01-28 - explicit "do not migrate"),
  and vLLM. Recommended path stays OpenVINO GenAI 2026.1.
- Architecture directory tree now lists the `cache/` volume and the persisted
  `illama_stats.json` snapshot.
- "Next Priority" rewritten to reflect what's still ahead (streaming
  disconnect-resilience, Ollama `allow_stream` enforcement, `/metrics`
  cardinality cap, CLI pull/rm hardening) rather than items already done.

### CLI

- `illama` script header version comment and `--help` banner updated from 2.1
  to 2.2.0 to match the value `illama version` already prints.

## 2.2.0 - 2026-05-03

### Added

- **Ollama-compatible API** under `/api/*`. Targets the subset of
  https://github.com/ollama/ollama/blob/main/docs/api.md that real
  Ollama-targeting clients (Continue.dev, LangChain `ChatOllama`,
  OpenWebUI Ollama backend, etc.) actually consume:
  - `GET /api/version`
  - `GET /api/tags`
  - `GET /api/ps`
  - `POST /api/show`
  - `POST /api/chat` (NDJSON streaming, nanosecond timing fields)
  - `POST /api/generate` (NDJSON streaming)
  This makes illama a drop-in Ollama replacement for those clients while
  keeping the existing `/v1/*` OpenAI-compatible surface intact.
- **OpenAI request fields:** `seed` (mapped to `GenerationConfig.rng_seed`),
  `response_format={"type":"json_object"}` (best-effort prompt prefix because
  OpenVINO GenAI lacks constrained sampling), and `frequency_penalty` is now
  mapped to `GenerationConfig.repetition_penalty` instead of being silently
  dropped.
- **Stats persistence.** `/illama/stats` counters and per-model history are
  now snapshotted to `${ILLAMA_CACHE_DIR}/illama_stats.json` (configurable via
  `ILLAMA_STATS_PATH`). Loaded on startup, saved on shutdown, plus a 5-minute
  periodic snapshot so a hard crash still preserves recent counters.
- **Docker `HEALTHCHECK`** on the runner image (`/health` every 30 s with a
  15 s start-period). It reports unhealthy status; `--restart unless-stopped`
  recovers from process exits. An unhealthy but live process needs external
  operator/orchestrator restart handling.
- **Lazy builder image.** `setup.sh` no longer pre-builds `illama-builder`.
  `illama pull <non-OpenVINO repo>` builds it on first use. Hosts that only
  ever pull pre-converted `OpenVINO/*-int4-ov` repos save ~5 GB of disk and
  one image-build pass.
- **CLI commands:**
  - `illama logs [N]` — tails the runner's docker logs (default 100 lines, follows).
  - `illama bench [args...]` — wraps `scripts/benchmark_api.py` against the
    running server, auto-starting it if needed.
  - `illama prune builder` — removes the conversion image to reclaim ~5 GB.

### Removed

- `bench.sh` — superseded by `illama bench` (which calls the safer,
  feature-richer `scripts/benchmark_api.py`).

### Changed

- `setup.sh` now creates `${ILLAMA_HOME}/scripts` and copies
  `scripts/*.py` so the new `illama bench` works after install.

## 2.1.0 - 2026-05-03

### Fixed

- **GPU model cache was effectively disabled.** The runner Dockerfile set
  `OV_GPU_CACHE_MODEL=1` and mounted `/cache`, but `server.py` never passed a
  `CACHE_DIR` to the OpenVINO pipeline, so compile artefacts went to a default
  location instead of the persistent volume. The pipeline is now constructed
  with `CACHE_DIR=$ILLAMA_CACHE_DIR` (default `/cache`), making subsequent
  loads of the same model on the same GPU substantially faster. The legacy
  `OV_GPU_CACHE_MODEL=1` env var was removed.
- **Bearer-token comparison was timing-attack-vulnerable.** Replaced
  `token == API_KEY` with `secrets.compare_digest`.
- **Prometheus label escaping was incomplete.** Model names containing newlines
  could corrupt `/metrics` output; backslash, double-quote and newline are now
  all escaped per the Prometheus exposition format.
- **SSE event queue was unbounded.** A stalled client could grow runner memory
  while the worker thread kept emitting tokens. The queue is now capped at 64
  events and the worker blocks on `put`.
- **CLI shell-into-Python injection paths.** `illama show`, `illama pull`, and
  `illama run` (single-shot and streaming) no longer splice values like model
  names, repo IDs, prompts or temp file paths into Python source or curl `-d`
  JSON. Inputs are passed via `sys.argv`, env vars or stdin heredocs.
- **`illama serve` deadlock on stale container.** If `illama-service` had
  exited (e.g. crash, OOM), `docker run` would fail with a name conflict. The
  CLI now removes any stopped same-name container before starting a new one.
  Container name matching is also anchored (`^name$`) to avoid false positives.

### Changed

- Dependency bumps:
  - openvino `2026.0.0` → `2026.1.0`
  - openvino-genai `2026.0.0.0` → `2026.1.0.0`
  - transformers `4.55.4` → `4.57.6` (still within optimum-intel 1.27.0's
    `<4.58,>=4.45` window)
  - accelerate `1.12.0` → `1.13.0`
  - fastapi `0.129.0` → `0.136.1`
  - uvicorn[standard] `0.40.0` → `0.46.0`
  - pydantic `2.12.5` → `2.13.3`
  - huggingface_hub stays at `0.35.3` (transformers 4.57.x still requires
    `<1.0`); nncf stays at `2.19.0` pending a 3.x compatibility test pass.
- `load_model` now returns `(pipeline, is_vlm)` so request handlers no longer
  read the `current_model_is_vlm` global outside the model lock.
- `unload_current_model_locked` calls `gc.collect()` once instead of three
  times.
- README has a new "Deployment (Ubuntu Docker)" section with fresh-install,
  upgrade, smoke-test, and rollback steps.

## 2.0.0 - 2026-02-27

- Made model idle offload fully configurable via environment:
  - `ILLAMA_IDLE_TIMEOUT` (supports values like `300`, `300s`, `5m`).
  - Alias `ILLAMA_IDLE_TIMEOUT_SEC` is also honored.
- Updated runner startup logging to show the effective idle timeout, models directory, and device.
- Tightened OpenAI-compatible endpoints in `runner/server.py`:
  - `/v1/chat/completions` and `/v1/completions` now accept additional OpenAI-style parameters (`top_p`, penalties, `stop`) and return richer `usage` information.
  - Both endpoints now measure latency and approximate token counts for responses.
- Added built-in inference statistics and health endpoints:
  - `/illama/health` exposes uptime, device, idle timeout, models directory, and currently loaded model.
  - `/illama/stats` exposes aggregated request counts, token counts, latencies, and per-model statistics.
- Extended the CLI `illama run` command to:
  - Support both interactive chat and single-shot mode (`illama run <model> "prompt"` or `illama run <model> -p "prompt"`).
  - Surface basic generation statistics (token counts and tokens/sec) after a single-shot request.
- Documentation improvements in `README.md`:
  - Clarified idle timeout behavior as an internal server watchdog (no external `watchdog.sh` required).
  - Documented the new monitoring endpoints and the extended semantics of `ILLAMA_IDLE_TIMEOUT`.
  - Added an explicit “Upgrade & Backup” section showing how to back up `/opt/illama` (including all models), redeploy a new version with `setup.sh`, and roll back safely if needed.

