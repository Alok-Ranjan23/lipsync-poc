# EchoMimicV3 Flash POC

This runner evaluates the direct 2D pipeline:

```text
portrait image + speech audio → EchoMimicV3 Flash → talking video
```

It is separate from the LAM-A2E and LatentSync environments.

## GPU setup

Run on the CUDA server:

```bash
cd /home/dev/alok/lipsync-poc/echomimic-v3
CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
```

The setup downloads the Wan2.1-Fun 1.3B base, Chinese Wav2Vec2 audio encoder,
and EchoMimicV3 Flash weights into the gitignored `weights/` directory.

## Run

```bash
CUDA_VISIBLE_DEVICES=1 bash run_image_audio.sh \
  assets/demo1_first_frame.png \
  assets/demo1_audio.wav \
  ../outputs/echomimic_v3_demo1.mp4
```

The runner reports end-to-end time and writes an MP4 with an audio track.

## Automatic short/long selection

Use one entry point for either output length:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_image_audio_auto.sh \
  --mode auto \
  assets/demo2_first_frame.png \
  assets/demo2_audio.wav \
  ../outputs/echomimic_v3_demo2.mp4
```

`auto` reads the input audio duration at 25 FPS:

- 81 frames or fewer: EchoMimicV3 Flash, one 8-step clip.
- More than 81 frames: EchoMimicV3 Preview, chunked long-video inference.

Override the choice with `--mode short` or `--mode long`. Flash is faster but
needs a larger direct GPU allocation. Preview uses memory offloading and is
slower, but supports long audio.

## Long audio inference

`run_image_audio.sh` uses the Flash model and produces a short 81-frame clip
(about 3.2 seconds at 25 FPS). For audio longer than 138 frames, use the
Preview model's Long Video CFG path:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_image_audio_long.sh \
  assets/demo2_first_frame.png \
  assets/demo2_audio.wav \
  ../outputs/echomimic_v3_demo2_long.mp4
```

The long runner uses EchoMimicV3's `app_mm.py` memory-managed path. It processes
the audio in 81-frame chunks with 8-frame overlap and uses `mmgp` offloading
with a 75% GPU-memory budget by default. It needs the Preview model weights and
standard `wav2vec2-base-960h`, both downloaded by `setup_gpu.sh`.

## Public fixtures

Three public image/audio fixture pairs are available under `assets/`:

```text
demo1_first_frame.png + demo1_audio.wav
demo2_first_frame.png + demo2_audio.wav
demo3_first_frame.png + demo3_audio.wav
```

See [`assets/README.md`](assets/README.md) for provenance and test-only usage.

## License and scope

EchoMimicV3 documents Apache-2.0 model terms, but the Wan2.1-Fun base,
Wav2Vec2 encoder, test assets, and generated content still require separate
license and consent review. This POC produces direct 2D video; it does not
produce ARKit-52 coefficients, a LAM avatar ZIP, VRM, or a reusable 3D avatar.
