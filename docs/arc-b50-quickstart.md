# Run Illama 2 on an Arc Pro B50

Validated on October 18, 2026: Arc Pro B50 16 GB, Ubuntu 26.04.1 LXC on
Proxmox, `xe` host driver, Docker runner based on Ubuntu 24.04. Native Ubuntu
with Docker uses the same runner; Proxmox-specific device setup is optional.

## 1. Verify the GPU before installing models

The host kernel must recognize the B50 and expose its render device:

```bash
lspci -nn | grep -Ei 'VGA|3D|display'
ls -l /dev/dri
```

The measured B50 identifies as `8086:e212`. Do not copy another machine's
`cardN`, render-node minor, group IDs, or `GPU.1` blindly. Device ordering can
change after reboot. For Proxmox passthrough examples and PCI mapping, see
[platform notes](b50-model-notes.md#platform-gotchas-the-expensive-lessons).
Keep SR-IOV VFs disabled while the physical function needs all 16 GB.

## 2. Build from source

Install Docker, its Compose plugin, Python 3, and curl using your distribution's
supported packages. Then:

```bash
git clone https://github.com/mkhomutskyi/illama2.git
cd illama2
bash setup.sh
```

The default runner and builder use the stable OpenVINO 2026.4.1 family.
The installer creates `/opt/illama` and the `illama` command. Building images
requires network access; inference uses your local downloaded model artifacts.

Discover the OpenVINO device names from the built image:

```bash
docker run --rm --device /dev/dri:/dev/dri --entrypoint python \
  illama-runner:2.6.0 -c \
  'from openvino import Core; c=Core(); print([(d,c.get_property(d,"FULL_DEVICE_NAME")) for d in c.available_devices])'
```

Select the device whose name identifies the B50/dGPU. In the measured
multi-GPU system this was `GPU.1`; the iGPU was `GPU.0`.

## 3. Configure and start

Edit `/opt/illama/deploy/.env`, using the device and numeric groups from your
machine. Example values, not automatic detection:

```dotenv
ILLAMA_DEVICE=GPU.1
ILLAMA_PORT=11434
ILLAMA_RUNNER_TAG=2.6.0
HOST_VIDEO_GID=44
HOST_RENDER_GID=110
```

Check the group owning the actual B50 render node with `stat -c '%g'` and
replace `HOST_RENDER_GID` accordingly; LXC group mappings can differ.
Set `ILLAMA_API_KEY` in the private `.env` if clients beyond a trusted local
machine can reach the service. Keep the file out of Git. Restrict the service
to your intended network; authentication does not replace network controls.

```bash
cd /opt/illama/deploy
docker compose up -d
curl -fsS http://localhost:11434/illama/health
```

When auth is enabled, supply the bearer token using your client configuration.
Do not paste it into public examples, screenshots, issue reports, or logs.
CLI `illama serve` reads shell environment; it does not source Compose's `.env`.
Use Compose consistently for the example above.

## 4. Use a model you are licensed to run

```bash
illama pull OpenVINO/Phi-4-int4-ov
illama run Phi-4-int4-ov -p 'Explain what a GPU does in two sentences.'
```

Pulling downloads weights. Some models require accepting their terms and
supplying `HF_TOKEN` privately. Existing OpenVINO conversions can be loaded
directly; other supported Transformers models use the separate builder.
Not every model architecture or export works. Disk size alone does not prove
that a model, KV cache, and activations fit in 16 GB VRAM.

## Optional: reproduce the October 18 nightly profile

Keep the stable image for rollback and build the dated overlay from this repo:

```bash
docker build -f runner/Dockerfile.nightly \
  -t illama-runner:nightly-20261015 runner
```

Set `ILLAMA_RUNNER_TAG=nightly-20261015` in the private deployment `.env`, then
run `docker compose up -d`. This updates inference only. It does not claim
validation of model conversion on a nightly builder. To roll back, select
`ILLAMA_RUNNER_TAG=2.6.0` and recreate the runner. Retain model and cache volumes.

See the [dated measurements and limitations](benchmarks/2026-10-18-b50.md).
Those results include both Gemma models on the stateful backend and Qwen on
the continuous backend; they are whole-stack measurements.

## Sharing the GPU with ComfyUI

Unload one workload before starting the other:

```bash
curl -fsS -X POST http://localhost:11434/v1/unload
```

The [separate ComfyUI project](https://github.com/mkhomutskyi/comfyui-intel-arc) describes the independently
pinned image stack. Installing Illama does not install ComfyUI.
