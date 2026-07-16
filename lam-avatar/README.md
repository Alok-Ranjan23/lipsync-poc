# LAM Avatar Creation POC

This POC evaluates the official companion avatar generator for LAM-A2E:

```text
one portrait photo → LAM → animatable Gaussian-head avatar → LAM ZIP export
```

LAM-A2E then consumes speech audio separately and produces ARKit expression
coefficients that drive the exported ZIP through the official LAM WebGL
renderer.

## Why this is the current primary candidate

LAM is the only evaluated open-source project that already connects all of
these stages:

- one-image avatar generation;
- an animatable avatar representation;
- an export format used by the official LAM WebGL renderer;
- an Audio2Expression companion model using ARKit facial coefficients.

Mesh/rig projects such as AniGen and MeshForge are useful alternatives to
evaluate later, but their GLB output still needs ARKit-52 facial blendshape
authoring, validation, and a renderer integration before it can use LAM-A2E.

## Setup

The upstream LAM install script targets CUDA 12.1 and PyTorch 2.3. Run this
only on a dedicated compatible NVIDIA GPU host. Do not use a shared GPU host
without confirming its capacity.

```bash
cd /home/alokr/code/lipsync/lipsync-poc/lam-avatar
LAM_PYTHON=python3.10 bash setup_gpu.sh
```

The setup clones the official LAM source, creates an isolated virtual
environment, and downloads the public LAM-20K assets and model weights.

## Create and export an avatar

```bash
cd /home/alokr/code/lipsync/lipsync-poc/lam-avatar
BLENDER_PATH=/absolute/path/to/blender bash run_gradio.sh
```

Open the local Gradio URL, upload one front-facing portrait, generate the
avatar, then use LAM's Avatar Export feature to produce the renderer ZIP.

`BLENDER_PATH` is required for ZIP export. The generated ZIP is a LAM
Gaussian-splat avatar asset, not a VRM/GLB file.

## Compatibility and scope

- The generated LAM ZIP works with
  [`aigc3d/LAM_WebRender`](https://github.com/aigc3d/LAM_WebRender), not a
  generic VRM renderer.
- This POC does not add a QVAC-owned renderer or photo-to-avatar production
  backend.

## License status

The upstream repositories used by this POC declare Apache-2.0:

- [`aigc3d/LAM`](https://github.com/aigc3d/LAM) source code;
- [`3DAIGC/LAM-20K`](https://huggingface.co/3DAIGC/LAM-20K) model weights;
- [`3DAIGC/LAM-assets`](https://huggingface.co/3DAIGC/LAM-assets) assets and
  face-blendshape model.

Apache-2.0 permits commercial use, modification, and distribution, provided
the required license, copyright, and NOTICE material is preserved. This
verification does not cover third-party dependencies, the separate renderer,
or rights in the input photo and generated likeness. Obtain appropriate user
consent and complete legal review before product distribution.

## Sources

- [LAM official repository](https://github.com/aigc3d/LAM)
- [LAM WebGL renderer](https://github.com/aigc3d/LAM_WebRender)
- [LAM Audio2Expression](https://github.com/aigc3d/LAM_Audio2Expression)
