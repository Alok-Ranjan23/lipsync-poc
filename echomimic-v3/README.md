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
