# LatentSync 1.6 — Architecture & Flow

How the **LatentSync** path works end-to-end, with our inputs:
`assets/sample_face.png` + `assets/sample_audio.wav`.

LatentSync is an **end-to-end audio-conditioned latent-*diffusion*** model (built on Stable
Diffusion + AnimateDiff-style temporal layers). Unlike MuseTalk's single pass, it runs a
**multi-step denoising loop** (e.g. 20 steps) per generation — heavier and slower, but it produces
the sharpest, most photorealistic **512×512** mouth and the best temporal consistency. It is
fundamentally a **video → video** model.

---

## 1. Components

| Component | Role | Checkpoint |
|---|---|---|
| **VAE (SD)** | pixel frames ⇄ latent | (SD VAE) |
| **U-Net + temporal layers** | denoise latents, conditioned on audio; smooth across frames | `latentsync_unet.pt` |
| **Whisper** | audio → embeddings (cross-attention conditioning) | `whisper/tiny.pt` |
| **SyncNet / LPIPS / TREPA** | **training-only** losses (lip-sync, perceptual, temporal) | `stable_syncnet.pt` |
| **ffmpeg** | encode + mux audio | system |

### How the U-Net is conditioned (diffusion)

```
noised latent (z_t) ┐
reference frames    ├─ channel-concat ─► U-Net (+ temporal attn) ─► predicted noise ─► z_{t-1}
masked frames       ┘                         ▲                         │
Whisper audio ──── cross-attention ───────────┘        repeat for N steps (t: T→0)
```

- The reference frame and the **masked frame** (mouth region hidden) are **channel-concatenated**
  with the **noised latent** and fed to the U-Net.
- **Whisper** audio embeddings condition the U-Net via **cross-attention**.
- **Diffusion loop:** starting from noise, the U-Net iteratively denoises over N steps
  (`--inference_steps`, e.g. 20) to produce a clean latent. `--guidance_scale` trades off audio
  adherence vs. natural motion.
- **Temporal layers** (AnimateDiff-style) attend across frames so the mouth/motion is smooth over
  time (no flicker). Training adds SyncNet (sync), LPIPS (perceptual) and TREPA (temporal) losses.

---

## 2. Flow for our asset

> LatentSync expects a **video**. Our asset is a still image, so it must first become a video.

```
sample_face.png ──(loop, 25 fps, scale 512)──► static_face.mp4     sample_audio.wav
                                                     │                    │
                                              per-frame VAE encode    Whisper encode
                                              + mouth-masked latent   audio embeddings
                                                     └──────────┬──────────┘
                                                                ▼
                                        DIFFUSION U-Net (N steps/frame, + temporal attn)
                                        noise → denoise conditioned on audio+reference+mask
                                                                │
                                                        VAE decode → 512×512 frames
                                                                │
                                                                ▼
                                          ffmpeg → H.264 + AAC(original audio) → out.mp4
```

### Step-by-step

1. **(Prep) image → video.** `sample_face.png` is looped for the audio's duration at 25 fps and
   scaled to 512 → `static_face.mp4` (the notebook's optional "photo → static video" cell). With a
   real driving video you skip this and use its frames directly.
2. **Audio features.** `sample_audio.wav` → Whisper → audio embeddings, aligned per frame.
3. **VAE encode (per frame).** Each frame is encoded to a latent; a **mouth-masked** version is also
   prepared. Reference + masked latents become the U-Net's spatial conditioning.
4. **Diffusion denoising (per frame).** From a noised latent, the U-Net runs **N denoising steps**,
   each conditioned on the audio (cross-attention) + reference + mask. **Temporal layers** keep
   consecutive frames coherent. `guidance_scale` controls lip expressiveness.
5. **VAE decode.** The clean latent is decoded to a **512×512** frame with the audio-matched mouth.
6. **Assemble.** Frames → ffmpeg → video + the original `sample_audio.wav` → `latentsync_out.mp4`.

---

## 3. How the pieces connect

- **Whisper → U-Net (cross-attention):** the audio pathway that drives lip motion.
- **VAE encode → U-Net (N steps) → VAE decode:** all generation is in latent space; diffusion is
  the iterative core (the expensive part).
- **Reference + masked frames → U-Net input:** tell the model *whose* face and *where* to paint.
- **Temporal layers:** connect frames to each other for smooth motion — the main reason LatentSync
  shines on **real video** (it preserves/smooths natural head motion).
- **SyncNet/LPIPS/TREPA:** training-only; absent at inference.

---

## 4. Why it looks the way it does

- **Strength:** 512² latent diffusion + temporal layers → sharpest, most photorealistic mouth and
  best temporal stability of the three. Best FID/sync in benchmarks.
- **Cost:** multi-step diffusion × frames → **slow**; wants a GPU (fp16). On CPU it's impractically
  slow and can OOM at fp32 (why it lives in the Colab notebook, not the local path).
- **Input:** truly **video → video**. On a static image the head is frozen (mouth-only motion) and
  results are weaker than on real footage — for photo→talking-head with motion, use Sonic/SadTalker.

---

## Cross-model summary

| | Wav2Lip | MuseTalk | LatentSync |
|---|---|---|---|
| Family | GAN (pixel) | latent inpainting (SD VAE+UNet) | latent **diffusion** (SD+temporal) |
| Steps/frame | 1 (feed-forward) | 1 (single inpaint pass) | **N** (denoising loop) |
| Mouth res | 96² | 256² | 512² |
| Audio encoder | conv audio encoder | Whisper | Whisper |
| Runtime | ONNX (CPU/GPU) | PyTorch (CPU/GPU) | PyTorch (GPU) |
| Best input | image/video | video (image ok) | video |
