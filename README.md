# dino.py — DINOv2/DINOv3 feature extraction straight on ggml

A tiny, dependency-light visual-feature extractor: it loads a DINOv2 / DINOv3
ViT from a GGUF file and runs the forward pass as a hand-built `ggml` compute
graph through the `ggml` Python FFI bindings — no PyTorch at inference time.
Plus two small applications built on top: a saliency/PCA-based "smart crop"
tool and a keyframe-picking video cropper.
