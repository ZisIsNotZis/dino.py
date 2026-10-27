**English** · [简体中文](README.zh-CN.md)

<p align="center">
  <img src="assets/logo.svg" width="120" height="120" alt="dino.py logo">
</p>

<h1 align="center">dino.py</h1>

<p align="center"><strong>DINOv2 and DINOv3 vision features on ggml — no PyTorch at inference.</strong></p>

<p align="center">Point it at a GGUF checkpoint and run a DINO ViT entirely through a hand-built ggml graph: NumPy images in, a global CLS descriptor and a dense patch-feature grid out.</p>

This is an experimental reference implementation, not a packaged application. The scripts are small enough to read in one sitting, and the inference path depends only on the `ggml` Python bindings, NumPy, Pillow and scikit-learn.

**Dependencies**

| Part of the repo | Needs |
| --- | --- |
| Inference (`gg.py`, `dino2.py`, `dino3.py`, `cutvid.py`) | `ggml` bindings providing the `ggml` and `ggml.utils` modules, `numpy`, `pillow`, `scikit-learn` |
| `cut.py` | a `dinov2` executable on `PATH` |
| Conversion helpers (`convert_*.py`, `llama_quantize.py`) | `torch`, `transformers`, `gguf` |

## Quick start

### 1. Install

You need Python 3.12, the `ggml` Python FFI bindings, and a GGUF model.

```bash
git clone https://github.com/ZisIsNotZis/dino.py
cd dino.py
python3.12 -m venv .venv
. .venv/bin/activate
pip install numpy pillow scikit-learn
```

Install the `ggml` bindings for your platform separately; they are not vendored here and are not on PyPI under a name this repository can assume. `gg.py` only needs the compiled bindings' Python modules `ggml` and `ggml.utils` to be importable. No model weights or lockfile are included.

### 2. Point at a model

`dino2.py` and `dino3.py` ship with local example paths in their `run()` functions. Change the `model=` default, or pass a path when calling `run` directly:

```python
cls, grid = run(image[None], model="/path/to/dinov2-small-f16.gguf")
```

Set `gpu=True` for the CUDA backend. The CPU path uses 28 threads by default; edit that in `gg.init` if your machine differs.

### 3. Run

```bash
python dino2.py photo.jpg   # DINOv2
python dino3.py photo.jpg   # DINOv3
```

Each input gets raw float32 feature files next to it, named `<image><dino2|dino3>.<dims>f`, where `<dims>` is the comma-joined array shape. The shape suffix is how `cut.py` rebuilds the feature file name it loads back; `cutvid.py` instead calls `run()` directly. Both scripts accept several images in one call. Set `VIS=1` to also write a PCA visualization as JPEG:

```bash
VIS=1 python dino2.py photo.jpg
```

### Use it from Python

```python
from PIL import Image
import numpy as np
from dino2 import run

image = np.asarray(Image.open("photo.jpg").convert("RGB"))
cls, grid = run(image[None], model="/path/to/dinov2-small-f16.gguf")

print(cls.shape)    # global descriptor; width follows the checkpoint
print(grid.shape)   # per-patch features for dense tasks
```

## What it can do now

- **DINOv2 ViT inference** — register tokens, fused or split QKV weights, LayerScale, GELU MLP and final LayerNorm, built as an explicit ggml graph.
- **DINOv3 ViT inference** — 2D RoPE, SwiGLU, register tokens and the DINOv3 normalization layout.
- **CPU or CUDA** — choose the ggml backend per run; no PyTorch in the inference path.
- **Dense and global features** — one forward pass yields the CLS descriptor and the patch grid used by the crop and video tools.
- **PCA visualizations** — `VIS=1` turns patch features into an RGB saliency image for a quick sanity check.
- **Checkpoint helpers** — convert DINOv3 checkpoints to Hugging Face format, convert HF ViTs to GGUF, and requantize GGUF files to f16.
- **Two applications** — `cut.py` for a PCA-saliency smart crop and `cutvid.py` for keyframe picking from video via FFmpeg/QSV.

Both prototypes assume their local helper setup: `cut.py` calls a `dinov2` executable on `PATH`, and `cutvid.py` needs FFmpeg with Intel QSV support.

## Architecture

```text
dino.py/
├── gg.py                          # Tensor/Graph DSL over the ggml FFI
├── dino2.py                       # DINOv2 ViT graph + image CLI
├── dino3.py                       # DINOv3 ViT graph + image CLI
├── cut.py                         # smart-crop prototype
├── cutvid.py                      # video keyframe prototype
├── convert_dinov3_vit_to_hf.py    # official DINOv3 ViT -> HF
├── convert_dinov3_convnext_to_hf.py  # official DINOv3 ConvNeXt -> HF
├── convert_dinov3_vit_hf_to_gguf.py  # HF DINOv2 ViT -> GGUF
├── convert_hf_to_gguf.py          # generic HF model -> GGUF
├── llama_quantize.py              # requantize an existing GGUF to f16
├── CONTRIBUTING.md                # scope, verification, PR rules
└── assets/logo.svg
```

The flow is deliberately direct:

```text
GGUF weights -> gg.py loader -> explicit ggml graph -> ggml backend -> NumPy arrays
```

`gg.py` wraps tensors, views, matmuls, normalization, attention, 2D RoPE and graph allocation. `dino2.py` and `dino3.py` contain only the model-specific wiring, which makes them useful as a reference when porting another ViT to ggml.

The conversion helpers are separate from inference and require PyTorch, Transformers and `gguf`. `convert_dinov3_vit_hf_to_gguf.py` also expects an external `dinov2_inference` module for GGML dtype mappings.

## How to develop

- There is no build step, test suite or CI. Validate a change by running a script on a real image and inspecting the output shapes and the `VIS=1` visualization.
- The code requires Python 3.12: it uses PEP 695 generics and PEP 701 f-string syntax, so older interpreters will fail to parse it.
- `NOIMP=1` swaps in-place ggml operations for out-of-place ones, which is useful on backends that do not support them.
- `gg.py` logs graph construction at `INFO`; raise or lower the level in the `basicConfig` call when debugging.
- Model paths are intentionally simple defaults, not a configuration system. Edit them or pass `model=` explicitly.

## How to contribute

Small, focused patches are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for scope, verification expectations and PR rules. Open an issue with:

- the checkpoint or GGUF family you are using,
- the backend (CPU or CUDA) and Python version,
- the command you ran and the output or traceback.

There is no test harness yet, so please include a reproducible command and the expected shapes. Keep the inference path free of new dependencies unless they are essential, and preserve the Apache-2.0 headers in the two upstream Hugging Face converters.

## Roadmap

**Implemented**

- [x] DINOv2 ViT inference from GGUF on ggml
- [x] DINOv3 ViT inference with 2D RoPE and SwiGLU
- [x] CPU and CUDA backend selection
- [x] CLS descriptors and patch-feature grids from one forward pass
- [x] PCA feature visualization
- [x] Smart-crop and video keyframe prototypes
- [x] GGUF conversion and f16 requantization helpers

**Planned**

- [ ] A real CLI/config path instead of editing model defaults in source
- [ ] A package layout and pinned, documented runtime environment
- [ ] Feature-parity tests against the reference PyTorch implementation
- [ ] A clean dependency story for the DINOv3-to-GGUF converter
- [ ] More DINOv3 backbones and broader GGUF coverage

The long-term idea is a small, model-agnostic vision runner: give it a GGUF file and an image, get usable global and dense features without a framework runtime. Cropping, keyframe selection and retrieval should then be thin layers on top of the same graph.

## License

The repository's own code is [AGPL-3.0-only](LICENSE), copyright © ZisIsNotZis. The upstream Hugging Face converters `convert_dinov3_convnext_to_hf.py` and `convert_dinov3_vit_to_hf.py` retain their Apache-2.0 copyright and license headers; the repository license does not replace those notices.
