# Open-Source Lip-Sync POC

Research + working proof-of-concept for open-source lip-sync, with a focus on
**realistic, human-like** rendering (not cartoon).

Three runnable paths:

1. **`wav2lip/`** — a self-contained pipeline (ONNX Runtime, no PyTorch) that runs on this
   machine today. Proves the end-to-end plumbing. Quality is the classic Wav2Lip baseline (soft
   96×96 mouth patch); an optional GPEN enhancer (`--enhance gpen`) sharpens it.
   **Auto-detects GPU**: uses CUDA if `onnxruntime-gpu` is installed, otherwise falls back to
   CPU (as here in WSL2). No code change needed.
2. **`musetalk/`** — MuseTalk 1.5 (single-step latent inpainting), the **most realistic
   result we can produce locally**. Verified running on **CPU** here (~14 min for a 6 s clip);
   the *same code* uses a **GPU** automatically if you install a CUDA build of PyTorch (then it's
   real-time). See §3c.
3. **`latentsync/`** — GPU runner for **LatentSync 1.6**, the highest-quality diffusion model
   (multi-step, needs a real NVIDIA GPU). Run it on an SSH GPU box — see **§4**.

To run any of these on a GPU box over SSH (setup + run scripts), see **§4**.

**CPU vs GPU at a glance:**

| Model | Local CPU | GPU | Code change for GPU? |
|---|---|---|---|
| `wav2lip` | ✅ | ✅ | None — auto-detects; just add `onnxruntime-gpu` |
| `musetalk` | ✅ | ✅ | None — just install CUDA PyTorch |
| `latentsync` | ⚠️ very slow | ✅ | None — run on a GPU box (§4) |

---

## 1. Research summary (2026)

| Model | Repo | Approach | Realism | Speed | VRAM | License |
|---|---|---|---|---|---|---|
| **LatentSync 1.6** ⭐ | [`bytedance/LatentSync`](https://github.com/bytedance/LatentSync) | Audio-conditioned latent diffusion (Stable Diffusion) | **Best** — sharp textures, real teeth, strong identity | Slow (~0.1× realtime) | ~18–20 GB @ 512² | Apache-2.0 ✅ |
| **MuseTalk 1.5** | [`TMElyralab/MuseTalk`](https://github.com/TMElyralab/MuseTalk) | Latent-space inpainting of mouth | High; slight jitter | **Real-time** (30fps+ on V100) | ~8–12 GB | research |
| **Sonic** (CVPR'25) | [`jixiaozhong/Sonic`](https://github.com/jixiaozhong/sonic) | SVD, single **still image** → talking head | Very natural head motion | Slow | ~32 GB | research |
| **SadTalker** | [`OpenTalker/SadTalker`](https://github.com/OpenTalker/SadTalker) | Photo + audio → head motion | Moderate | Variable | ~12 GB | check |
| **Wav2Lip** | [`Rudrabha/Wav2Lip`](https://github.com/Rudrabha/Wav2Lip) | GAN, 96×96 mouth patch | Dated, blurry mouth | ~1× realtime; **CPU-capable via ONNX** | ~8 GB (GPU) / CPU | non-commercial weights |

**Recommendation for "human-like":** **LatentSync 1.6** — the 2026 consensus best local model for
realistic lip replacement on real footage; Apache-2.0 (commercially safe); actively maintained.

- Existing **video** + new audio → **LatentSync** (or MuseTalk for speed).
- Single **photo** + audio → **Sonic** / **SadTalker**.

### Input type: image vs. video (and why it matters)

With a **still image**, the head is frozen — only the mouth moves → it reads as a "talking
portrait." With a real **video**, you also inherit natural head/eye motion → much more
convincingly "a person talking."

| Model | Accepts | Best with | Why |
|---|---|---|---|
| Wav2Lip | image or video | either | mouth-only patch |
| MuseTalk | image or video | **video** | inpaints mouth; video adds real head motion |
| LatentSync | video (photo via static-video) | **video** | diffusion mouth replacement on real footage |
| Sonic / SadTalker | single image | image | generates head motion from a photo |

**Bottom line:** it's not that these *need* video — it's that they **reuse** the real motion in a
video, which is exactly what makes them look human. If you want realism from just a photo (with
head motion), that's **Sonic/SadTalker's** job instead.

---

## 2. Hardware reality (local dev machine)

The CPU baseline (§3, §3b, §3c) was built and measured on a **local laptop** with **no GPU**
(WSL2). The GPU path is §4. This section explains why the local defaults are CPU-oriented.

```
GPU: none        (/dev/dxg missing, no libcuda in WSL2)
CPU: 16 cores    RAM: 13 GB
```

Consequences that shaped this POC:

- **MuseTalk (diffusion-class, single-step) DOES run on this CPU** — verified, see §3c. It's the
  best local quality. Only *speed* suffers on CPU (minutes vs. seconds).
- **LatentSync (multi-step diffusion, 512²) is the one to keep on a GPU** — technically runnable on
  CPU but impractically slow, and at fp32 it risks exceeding 13 GB RAM. Run it on a GPU box (§4).
- **GGUF / llama.cpp is not applicable to lip-sync models.** `llama.cpp`/GGUF only run transformer
  LLM graphs; the ComfyUI-GGUF project explicitly can't quantize the **UNet/conv2d** graphs used by
  LatentSync and MuseTalk. There are no GGUF lip-sync models and no runtime to load one.
- **The only thing that runs locally on CPU is Wav2Lip via ONNX/OpenVINO** — which is exactly why
  the local baseline below uses it. General graph runtimes (ONNX Runtime, OpenVINO) support the conv
  ops these models need; `llama.cpp` does not.

### What quantization is realistic for MuseTalk

If the goal is "make MuseTalk smaller/faster," the real options are:

| Approach | Feasible? | Notes |
|---|---|---|
| GGUF / llama.cpp | ❌ | wrong architecture + no runtime |
| ONNX export + ORT INT8 | ✅ | community ONNX ports of MuseTalk parts exist |
| PyTorch quantization / `torch.compile` | ✅ | native |
| TensorRT / fp16 | ✅ | best on GPU |

So: **MuseTalk → GGUF = no** (it's a conv-UNet PyTorch pipeline). If you want it lighter, the path
is **ONNX + INT8** (or TensorRT/fp16 on GPU) — the same "general graph runtime" reasoning as why
Wav2Lip works in ONNX but not GGUF.

---

## 3. Local POC — Wav2Lip (ONNX, CPU/GPU auto)

Runs on **CPU by default** (auto-detects and uses a GPU if `onnxruntime-gpu` is installed).
Accepts a **face image or video** + **audio** and produces a lip-synced MP4. A still image is
treated as a repeated frame, so a single photo works.

### Setup — CPU or GPU (same code, `providers.py` auto-detects)

```bash
cd lipsync-poc/wav2lip
bash setup.sh                                # CPU  (onnxruntime)
# --- or, on an NVIDIA box --- (GPU: onnxruntime-gpu + cuDNN)
CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
```
Both create the repo-root `.venv` and fetch the ONNX weights (`wav2lip_gan.onnx`, `yunet.onnx`,
`sface.onnx`). Note: Wav2Lip is tiny, so CPU is already only seconds; GPU gain is marginal (and
`onnxruntime-gpu` may lack Blackwell/sm_120 kernels → falls back to CPU).

### Run

```bash
cd lipsync-poc/wav2lip
bash run.sh                                  # sample face+audio -> outputs/wav2lip_demo.mp4
# your own inputs (+ optional GPEN enhancer as 4th arg):
bash run.sh <face> <audio> outputs/out.mp4 gpen
```
`run.sh` prints `[device] ONNX providers ...` and `[TIME] end-to-end`, and auto-adds the cuDNN lib
path for the GPU install. (You can also call `python wav2lip/infer.py --face ... --audio ... --out ...`
directly.)

Output: `outputs/wav2lip_demo.mp4`. On a 16-core CPU, ~156 frames (6.3 s @ 25 fps) in ~13 s.

### Optional: sharpen the mouth with a face enhancer

Wav2Lip's mouth is 96² and looks soft. Add `--enhance gpen` to run a GPEN-BFR-512 face
restorer (ONNX, CPU) over the generated face so the mouth matches the rest:

```bash
python wav2lip/download_models.py --enhancer   # also fetches gpen_bfr_512.onnx
python wav2lip/infer.py \
  --face assets/sample_face.png --audio assets/sample_audio.wav \
  --out outputs/wav2lip_enhanced.mp4 --enhance gpen
```

Measured effect on the demo (via `evaluate.py`):

| Metric | Baseline | + GPEN | Note |
|---|---|---|---|
| mouth sharpness | 7.3 | **50.7** | ~7× crisper |
| mouth/face ratio | 0.13 | **0.36** | seam removed |
| CSIM (identity) | 0.95 | 0.92 | still same person |
| time (156 frames, CPU) | ~15 s | ~5m50s | ~2.2 s/frame |

Trade-off: much sharper, but ~25× slower on CPU (GPU makes this negligible). It still won't
match LatentSync's native 512² generation, but it noticeably closes the gap.

### Result (from `outputs/frame_check.png`)

A mid-clip frame shows the mouth actively synthesized to the audio — and the expected Wav2Lip
limitation: a soft, lower-resolution 96×96 patch around the mouth on the otherwise sharp face.
This is the "dated/cartoonier" baseline and the reason to use LatentSync for realism.

### What's in the pipeline

- `wav2lip/audio.py` — mel-spectrogram exactly matching Wav2Lip's training params.
- `wav2lip/infer.py` — YuNet face detection → 96×96 crop (lower half masked) → ONNX
  generator → paste-back → ffmpeg audio mux.
- `wav2lip/download_models.py` — fetch ONNX weights.
- `wav2lip/make_sample_audio.py` — optional offline Piper TTS to (re)generate the demo audio.

### Models used (all fetched programmatically)

- `wav2lip_gan.onnx` — HuggingFace `wanesoft/faceswap_pack` (145 MB).
- `yunet.onnx` — OpenCV Zoo YuNet face detector.
- Piper voice `en_US-lessac-medium` — HuggingFace `rhasspy/piper-voices` (demo audio only).

---

## 3b. Measuring output quality

Lip-sync quality is evaluated on three independent axes:

| Axis | Standard metric | Needs ground truth? |
|---|---|---|
| **Sync accuracy** | **LSE-C** (↑) / **LSE-D** (↓) via SyncNet | No |
| **Visual realism** | FID, FVD, SSIM (↑), PSNR (↑), LPIPS (↓) | Yes (except FID) |
| **Identity** | **CSIM** (↑, ArcFace cosine) | No |
| **Overall** | **MOS** (human 1–5) | No |

`wav2lip/evaluate.py` computes the **reference-free** ones offline on CPU
(CSIM + mouth/face sharpness):

```bash
python wav2lip/evaluate.py --video outputs/wav2lip_demo.mp4 --source assets/sample_face.png
```

Sample result on the Wav2Lip CPU demo:

```
CSIM (identity)  : mean=0.950     -> identity well preserved
mouth sharpness  : 7.3
face sharpness   : 54.7
mouth/face ratio : 0.13           -> mouth ~7.5x blurrier than the face
```

Interpretation: identity is great, but the **mouth is far softer than the face** —
the quantified reason Wav2Lip doesn't look like a real human talking. LatentSync
fixes this by generating the whole region at 512².

For the **standard sync score (LSE-C/LSE-D)**, run SyncNet (PyTorch, GPU) on the
output: https://github.com/joonson/syncnet_python

## 3c. Local MuseTalk (diffusion-class, more realistic)

MuseTalk 1.5 (256² latent inpainting) is the **most realistic result we can produce locally**.
It's more human-like than Wav2Lip because it regenerates the mouth at **256²** and leaves the
rest of the face untouched. It runs on **CPU** (verified, ~14 min for a 6 s clip) and
auto-uses a **GPU** if a CUDA build of PyTorch is installed (then it's near real-time). It's
CPU-feasible because it's **single-step** (one UNet pass per frame), unlike LatentSync's
multi-step diffusion.

Setup lives in `musetalk/` — an isolated Python 3.10 env (via `uv`) with CPU PyTorch + the
mmlab stack (this is a heavier, PyTorch pipeline, not ONNX).

### Setup (one-time)

```bash
# Prereq: uv  ->  curl -LsSf https://astral.sh/uv/install.sh | sh
cd lipsync-poc/musetalk
bash setup.sh          # CPU env  (Python 3.10 + CPU torch + mmlab, ~4 GB weights)
# --- or, on an NVIDIA box (GPU) --- (see §4; mmcv needs the CUDA toolkit on Blackwell)
CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
```

Both `setup.sh` (CPU) and `setup_gpu.sh` (GPU) create the venv at `musetalk/.venv`, so the **same
`run.sh` works for either** — it prints `[device] CPU` or `[device] CUDA - <gpu>` at the start.

### Run

```bash
cd lipsync-poc/musetalk
bash run.sh            # writes lipsync-poc/outputs/musetalk_demo.mp4  (+ [TIME] and [device])
```

`run.sh` handles ffmpeg (bundled) and the known harmless `save_dir_full` error the repo throws
*after* the mp4 is written for still-image input. To use your own inputs, edit
`musetalk/MuseTalk/configs/inference/poc.yaml` (`video_path` accepts an image or a video,
`audio_path` a wav).

### How MuseTalk produces the output (step-by-step)

1. **Landmarks** — DWPose (mmpose) finds the face and key points in each input frame; a still
   image is treated as one frame reused for the whole audio.
2. **Audio → features** — the wav is encoded by **Whisper** into per-frame audio features.
3. **Encode the face** — the face crop is compressed into a latent by the **VAE**, with the
   mouth region masked out.
4. **Inpaint the mouth (1 step)** — the **UNet** fills the masked latent, conditioned on the
   Whisper audio features, so the mouth matches the sound. One forward pass per frame (no
   multi-step denoising) — this is why it's CPU-viable.
5. **Decode + blend** — the VAE decodes the latent back to pixels; face-parsing blends the new
   mouth back into the original frame (so hair/eyes/background stay pixel-identical).
6. **Assemble** — frames are encoded to video and the original audio is muxed in → the mp4.

One-line mental model: *encode the face, erase the mouth, let the UNet repaint it to match each
slice of audio, decode, and blend it back — keeping everything else from the source.*

### Local method comparison (same face + audio, all on CPU)

| Method | Mouth sharpness | Face sharpness | CSIM | Realism | Time (6.3s clip) |
|---|---|---|---|---|---|
| Wav2Lip | 7.3 | 54.7* | 0.95 | soft mouth patch | ~15 s |
| Wav2Lip + GPEN | 50.7 | 142* | 0.92 | sharper, some hallucination | ~6 min |
| **MuseTalk 1.5** | 18.5 | 199 | 0.92 | **most natural, no seam** | ~14 min |

\* Wav2Lip re-pastes a resized crop, lowering whole-face sharpness; MuseTalk leaves the rest of
the face at native resolution (hence 199), only regenerating the mouth — which is why it looks
the most natural despite a moderate absolute mouth-sharpness number.

Takeaway: **the realistic path does not strictly require a GPU.** MuseTalk gives the best local
result; the GPU only matters for *speed* (seconds vs. minutes) and for the heavier multi-step
LatentSync.

## 4. Running on a GPU box over SSH (e.g. RTX 5090 / CUDA 13.2)

On a real NVIDIA box the models run far faster and with fewer surprises. Setup + run scripts are
provided; pick the **free GPU** with `CUDA_VISIBLE_DEVICES`.

> **Blackwell note (RTX 50-series):** the 5090 is `sm_120` and needs **PyTorch ≥2.7 built for
> CUDA 12.8 (`cu128`)** — older torch has no kernels for it. The GPU setup scripts install `cu128`.

```bash
git clone https://github.com/<you>/lipsync-poc.git && cd lipsync-poc
nvidia-smi                     # find a free GPU (note its index)

# --- LatentSync (best fit for a big GPU; no mmcv) ---
cd latentsync
CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh
CUDA_VISIBLE_DEVICES=1 bash run.sh        # -> ../outputs/latentsync_out.mp4

# --- Wav2Lip (tiny; CPU is fine, or onnxruntime-gpu) ---
cd ../ && python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt           # or: pip install onnxruntime-gpu (instead of onnxruntime) for CUDA
python wav2lip/download_models.py
python wav2lip/infer.py --face assets/sample_face.png --audio assets/sample_audio.wav --out outputs/wav2lip_demo.mp4

# --- MuseTalk (GPU; mmcv on Blackwell may need a source build - see script warning) ---
cd musetalk
CUDA_VISIBLE_DEVICES=1 bash setup_gpu.sh  # if mmcv fails on Blackwell, use ./setup.sh (CPU) instead
CUDA_VISIBLE_DEVICES=1 bash run.sh
```

Reliability on a fresh Blackwell box: **LatentSync = clean**, **Wav2Lip = clean**, **MuseTalk =
possible but `mmcv` is finicky** (needs the CUDA toolkit to compile against `cu128` torch).

LatentSync is **video → video**; for a photo, use the commented photo→static-video block in
`latentsync/run.sh` (or prefer Sonic/SadTalker for photo→talking-head with head motion).

---

## 5. Suggested next steps

**Finish the comparison**
- Run all three on the GPU box (§4) and collect timing + `CSIM/mouth_sharp/face_sharp` into one
  table (GPU-speed vs. quality, apples-to-apples with the local CPU runs).
- Add **SyncNet LSE-C / LSE-D** (the one missing, industry-standard *lip-sync accuracy* metric;
  needs PyTorch + GPU) so ranking isn't based on sharpness/identity alone.

**Improve realism**
- Drive with a **real short video** instead of the still photo — this is the biggest jump, and it's
  what lets MuseTalk/LatentSync show natural head motion (a static image only moves the mouth).
- For **photo-only + head motion**, evaluate **Sonic / SadTalker** (image → talking-head).

**Pick per use case**
- Best local / near real-time → **MuseTalk**.
- Max quality on real footage → **LatentSync 1.6**.
- Lightweight / CPU-only / fast dubbing → **Wav2Lip (+GPEN)**.

**Toward production**
- Decide GPU hosting for the diffusion models (the shared SSH box / RunPod / Lambda).
- Wrap the chosen model in a small batch/HTTP API; add queueing for multi-request use.
- Speed up MuseTalk with **ONNX + INT8 / TensorRT / fp16** (see §2) if throughput matters.

## Repo layout

```
lipsync-poc/
├── README.md
├── doc/                architecture + flow write-ups per model
│   ├── wav2lip_arch.md
│   ├── musetalk_arch.md
│   └── latentsync_arch.md
├── latentsync/         GPU runner for LatentSync (setup_gpu.sh + run.sh; clone gitignored)
├── requirements.txt
├── assets/            sample_face.png, sample_audio.wav
├── models/            downloaded ONNX weights (gitignored)
├── outputs/           generated videos
├── wav2lip/            Wav2Lip pipeline (ONNX; CPU/GPU auto)
│   ├── setup.sh             CPU env + weights
│   ├── setup_gpu.sh         GPU env (onnxruntime-gpu + cuDNN)
│   ├── run.sh               run inference (handles cuDNN path) -> outputs/
│   ├── audio.py
│   ├── infer.py
│   ├── providers.py         ONNX CPU/GPU provider auto-detect
│   ├── enhancer.py          GPEN face restorer (--enhance gpen)
│   ├── evaluate.py          reference-free metrics (CSIM + sharpness)
│   ├── compare.py           one-command comparison across outputs
│   ├── download_models.py
│   └── make_sample_audio.py
└── musetalk/            local MuseTalk (py3.10 venv + cloned repo)
    ├── setup.sh             one-time CPU env + weights setup
    ├── setup_gpu.sh         GPU env (cu128 torch; mmcv may need source build on Blackwell)
    ├── run.sh               run inference -> outputs/musetalk_demo.mp4
    ├── .venv/               (gitignored)
    └── MuseTalk/            cloned repo + models (gitignored) + configs/inference/poc.yaml
```
