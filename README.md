# dino.py — DINOv2/DINOv3 feature extraction straight on ggml

A tiny, dependency-light visual-feature extractor: it loads a DINOv2 / DINOv3
ViT from a GGUF file and runs the forward pass as a hand-built `ggml` compute
graph through the `ggml` Python FFI bindings — no PyTorch at inference time.
Plus two small applications built on top: a saliency/PCA-based "smart crop"
tool and a keyframe-picking video cropper.

The scripts are an experimental archive, not a packaged application. `dino2.py`
and `dino3.py` expect local GGUF model files at hard-coded paths in their `run`
functions; edit those paths before running them with image paths as arguments.
They require Python 3.12, `ggml` Python bindings, NumPy, Pillow, and
scikit-learn. The converters additionally depend on PyTorch, Transformers and
GGUF tooling; `cutvid.py` uses FFmpeg with Intel QSV support. No models or
dependency lockfile are included.

The repository's own code is under [AGPL-3.0-only](LICENSE). The two upstream
Hugging Face converters `convert_dinov3_convnext_to_hf.py` and
`convert_dinov3_vit_to_hf.py` retain their Apache-2.0 copyright and license
headers; the repository license does not replace those notices.
