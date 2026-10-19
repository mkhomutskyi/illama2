# ComfyUI on Intel Arc Pro B50 — October 18, 2026

ComfyUI uses a separate PyTorch XPU stack from Illama's OpenVINO runner.
Do not install one project's runtime into the other's container. Unload the
other workload before measuring; both share the B50's 16 GB dedicated VRAM.

## Recorded working configuration

| Component | Measured pin |
|---|---|
| Local image | `llm-scaler-omni:0.2.0-b2-comfyui-bmg` |
| Image ID | `sha256:1d99f381e7652cf756aa8c12ad7f2579bb1c92f71e109982e46a533b997cafad` |
| Intel llm-scaler source | `49c94a32bed69fbabd824debc4b136a6d15cae6d` |
| ComfyUI | 0.39.0, `b0b743566f65daafc423b4fea8a2fbda94b3384a` |
| Frontend / workflow templates | 1.53.10 / 0.11.76 |
| PyTorch / torchvision / torchaudio | 2.14.0+xpu / 0.29.0+xpu / 2.11.0+xpu |
| Comfy Kitchen / AIMDO XPU runtime | 0.2.37 / 0.5.5 |
| Native kernel | `omni_xpu_kernel==0.2.0b2+torch214.bmg` |
| Device | `xpu:0 Intel(R) Arc(TM) Pro B50 Graphics` |
| Memory policy | Dynamic VRAM, 3 GB reserve, batch 1 in these tests |

Models are mounted read-only at `/models/host` and discovered through the
image's `/llm/configs/comfyui_host_models.yaml`. Input, output and user data
are separate writable volumes. The image owns `/llm/ComfyUI`; no host checkout
is overlaid there. Local optional Impact Pack/Subpack and IPAdapter nodes
are separate read-only additions and are not needed for the baseline graphs.

The measured operator deployment uses privileged mode and an unconfined
AppArmor profile. That is a recorded local exception, not a requirement or
recommendation for a portable deployment. The [portable recipe](../examples/comfyui-b50/compose.yaml)
omits those flags, binds the UI to localhost, and uses no startup package
installation. Its deployment validation status is documented below.

Validation on October 18 used a separate container with **privileged=false**,
no extra security options, and **network=none**. Intel's image acceptance check
passed, and the saved 1024 FP8 workflow generated an image in 19.83 seconds
(one initial-load observation). The temporary container was stopped and removed;
the original service and output files were preserved. This validates the image,
device access and offline workflow; firewall/port binding still depends on the
operator's own deployment.

## Build and validate the Intel image

The image is locally built; do not replace it with a similarly named Hub tag.
Intel's current source documents 0.2.0-b2 as source-only. See the
[official build guide](https://github.com/intel/llm-scaler/blob/main/omni/docs/IMAGE_BUILD.md).

```bash
git clone https://github.com/intel/llm-scaler.git
cd llm-scaler
git checkout 49c94a32bed69fbabd824debc4b136a6d15cae6d
cd omni
MAX_JOBS=4 OMNI_IMAGE_REPOSITORY=llm-scaler-omni XPU_TARGET=bmg bash build.sh

docker run --rm --device=/dev/dri \
  llm-scaler-omni:0.2.0-b2-comfyui-bmg \
  /llm/entrypoints/validate_comfyui_image.sh
```

The acceptance check must identify the B50 and the BMG target. Building the
image requires network access and build space; it does not bundle model weights.
Model licenses remain separate from this repository's Apache-2.0 instructions.
ComfyUI itself is not vendored or relicensed here.

## Portable deployment

Prepare existing models under `COMFY_DATA/models`, plus `input`, `output` and
`user` directories. Copy `.env.example` beside the example Compose file to
`.env`, then set local paths and actual GPU group IDs:

```bash
cd examples/comfyui-b50
cp .env.example .env
# Edit .env. Determine GIDs from your actual /dev/dri card/render nodes.
docker compose config --quiet
docker compose up -d
curl -fsS http://localhost:8188/system_stats
```

Check the actual device name; `ZE_AFFINITY_MASK=0` is not universally the B50.
Keep SR-IOV VFs off while the physical function needs the full 16 GB. Dynamic
VRAM may offload/stage weights through system memory: successful inference is
not proof the whole pipeline fits simultaneously in dedicated VRAM.

The portable recipe deliberately disables Hugging Face network fetches. It
expects model files that the operator has already obtained and is licensed to
use. It does not install optional custom nodes or download missing weights.

## Can this stack be updated?

Checked October 18: upstream ComfyUI's latest release is **0.41.0**, while this
validated Intel source pins **0.39.0**. Intel main was three commits ahead of
the measured source, but its comparison changed SGLang kernels, not the Omni
ComfyUI integration. A newer standalone ComfyUI version is not evidence of a
compatible replacement for this image's Torch ABI, Kitchen, AIMDO, and BMG
native binaries.

No ComfyUI runtime update was deployed during these measurements. Keep this
pin until a separately built candidate passes Intel's acceptance script,
dependency checks, GPU identity check and the same saved workflows. Preserve
the old image and data volumes for rollback. Do not use `pip install -U torch`
or overlay stock ComfyUI into this running image.

Sources: [Intel Omni](https://github.com/intel/llm-scaler/tree/main/omni),
[ComfyUI releases](https://github.com/Comfy-Org/ComfyUI/releases),
[Intel source comparison](https://github.com/intel/llm-scaler/compare/49c94a32bed69fbabd824debc4b136a6d15cae6d...49d2f8d074397c91f138f933e7d7d6ff033f1581).

## Existing models and measurements

See [the dated image benchmark](benchmarks/2026-10-18-comfyui-b50.md), its
saved workflows, and the [benchmark client](../scripts/benchmark_comfyui.py).
The client submits only explicit local loader filenames, never cloud API nodes.
It refuses an occupied queue and unloads Illama before starting.

LoRAs, IPAdapters, CLIP vision encoders, VAEs and motion modules are components,
not standalone generators. This machine has a Wan VAE and an AnimateDiff motion
module, but no corresponding complete video base-model pipeline in the inspected
store. Do not describe those files as a tested video model or a VRAM failure.
