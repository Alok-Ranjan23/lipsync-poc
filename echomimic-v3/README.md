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
  ../assets/sample_face.png \
  ../assets/latentsync/demo2_audio.wav \
  ../outputs/echomimic_v3_sample_face_demo2_audio.mp4
```

The runner reports end-to-end time and writes an MP4 with an audio track.

## License and scope

EchoMimicV3 documents Apache-2.0 model terms, but the Wan2.1-Fun base,
Wav2Vec2 encoder, test assets, and generated content still require separate
license and consent review. This POC produces direct 2D video; it does not
produce ARKit-52 coefficients, a LAM avatar ZIP, VRM, or a reusable 3D avatar.
