# LongCat-Video-Avatar 1.5 POC

This POC evaluates the upstream direct talking-avatar path:

```text
reference image + speech audio + text prompt
  → vocal separation + Whisper-large-v3
  → LongCat Avatar 1.5 flow-matching video DiT
  → Wan VAE decode + MP4 audio mux
  → talking-avatar video
```

It is a **diffusion-family flow-matching video generator**, not a LAM-A2E
audio-to-ARKit coefficient model. The inference result is a rendered MP4, not
a VRM, rig, or reusable facial-animation sequence.

## Upstream fixture

The initial POC uses the repository's unmodified single-speaker fixture:

- Manifest: `LongCat-Video/assets/avatar/single_example_1.json`
- Reference image: `assets/avatar/single/man.png`
- Speech audio: `assets/avatar/single/man.mp3`
- Mode: Audio-Image-to-Video (`ai2v`)

The manifest also supplies the text prompt. It is intentionally used unchanged
to verify the upstream path before testing a custom image or audio file.

## Runtime architecture

| Component | POC choice |
|---|---|
| Avatar model | `meituan-longcat/LongCat-Video-Avatar-1.5` |
| Base model | `meituan-longcat/LongCat-Video` |
| Audio encoder | Whisper-large-v3 |
| Inference mode | distilled 8-step Avatar 1.5 |
| Transformer weights | upstream INT8 DiT (`--use_int8`) |
| Parallelism | 2 GPUs, context parallel size 2 |
| Initial resolution | 480p |
| Initial duration | 93 frames at 25 FPS, about 3.72 seconds |

For longer output, `LONGCAT_NUM_SEGMENTS` requests video-continuation
segments. Each later segment reuses 13 condition frames from the previous
segment while consuming the corresponding next audio window.

## Run

From this POC directory on the CUDA host:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_upstream_fixture.sh
```

The runner requires both checkpoint directories and uses the POC-local FFmpeg
binary. It writes the first result beneath
`outputs/upstream-single-fixture/`.

## Licensing and fixture use

The upstream repository says its weights are MIT-licensed. Its project page
also says that showcase images and audio derived from real videos are for
academic demonstration only. Treat the included fixture as POC-only and
perform separate source-asset, generated-content, privacy, consent, and
product-use review before shipping.

## Sources

- https://github.com/meituan-longcat/LongCat-Video
- https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5
- https://meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page/
