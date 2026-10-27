# Contributing

`dino.py` is a small experimental reference implementation, not a packaged
application. The most useful contribution is a working demonstration of
something that currently fails: another DINO checkpoint, another ggml backend,
another image or video shape.

## Scope

In scope:

- correctness fixes in `gg.py`, `dino2.py` and `dino3.py` (graph wiring, dtype and
  shape handling, checkpoint key layouts),
- support for more DINOv2/DINOv3 checkpoints that convert to GGUF,
- fixes to the conversion helpers,
- documentation that removes a real stumbling block.

Out of scope: turning the scripts into a framework, adding a dependency the
inference path does not need, or reformatting files you are not otherwise
changing.

## Before you open a PR

There is no build step and no test suite, so describe how you verified the
change. A useful report contains:

- the checkpoint or GGUF file family you used, and how you produced the GGUF if
  you converted it,
- backend (CPU or CUDA) and Python version (3.12 is required — the code uses
  PEP 695 generics and PEP 701 f-strings),
- the exact command you ran,
- the shapes you got from `run()` and, for visual changes, the `VIS=1`
  visualization.

Sanity check on a real image:

```bash
python dino2.py photo.jpg
VIS=1 python dino2.py photo.jpg
```

`run()` should return a CLS descriptor and a patch grid whose spatial extent
matches the image size divided by the patch size. If in-place ggml operations
are unsupported on your backend, `NOIMP=1` swaps them for out-of-place ones.

## PR expectations

- One change per PR, with a message in Conventional Commits form
  (`fix:`, `feat:`, `docs:`, `chore:`).
- Keep the inference path free of new dependencies unless they are essential.
- Preserve the Apache-2.0 headers in `convert_dinov3_vit_to_hf.py` and
  `convert_dinov3_convnext_to_hf.py`; they carry upstream Hugging Face code and
  the repository's AGPL-3.0 license does not replace those notices.
- Do not commit model weights, GGUF files, or generated feature/visualization
  files; `.gitignore` covers the usual names.

## Issues

Please include the checkpoint family, backend, Python version, the command you
ran, and the full traceback. "It does not work on my model" cannot be acted on;
a minimal reproduction with a public checkpoint can.
