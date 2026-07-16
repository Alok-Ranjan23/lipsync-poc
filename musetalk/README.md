# MuseTalk POC

MuseTalk lip-syncs a still image or an input video to replacement speech audio.
The runner writes a talking-video MP4 to `outputs/musetalk_demo.mp4`.

## Setup

From this directory:

```bash
CUDA_VISIBLE_DEVICES=0 bash setup_gpu.sh
```

For CPU-only setup:

```bash
MUSETALK_VENV=.venv-cpu bash setup.sh
```

## Run the default POC

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
```

For the CPU environment:

```bash
MUSETALK_VENV=.venv-cpu bash run.sh
```

## Run public fixtures

The media files are in `../assets/musetalk-public/`; see that directory's
README for their sources and terms.

```bash
CUDA_VISIBLE_DEVICES=0 \
MUSETALK_CONFIG=fixtures/web_du_bois_librivox.yaml \
MUSETALK_OUTPUT=results/poc/v15/web_du_bois_cc0_librivox_one_minute_test.mp4 \
bash run.sh
```

Use `fixtures/oppenheimer_librivox.yaml` for the second still portrait, or
`fixtures/obama_librivox.yaml` for video-to-video lip-sync.

The fixture files must be re-downloaded after a clean checkout because their
binary media is intentionally gitignored.
