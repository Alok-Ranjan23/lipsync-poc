# MuseTalk 1.5 — Architecture & Flow

How the **MuseTalk** path works end-to-end, with our exact inputs:
`assets/sample_face.png` (still portrait) + `assets/sample_audio.wav` (~6 s TTS speech).

MuseTalk is a **latent-space inpainting** model. It uses a Stable-Diffusion-style **VAE + U-Net**,
but crucially it runs the U-Net **once per frame** (a single inpainting pass, *not* an iterative
diffusion denoising loop). That single-step design is why it's much faster than LatentSync and can
run on CPU, while still regenerating the mouth at **256×256** (sharper/more human than Wav2Lip).

---

## 1. Components

| Component | Role | Weight file |
|---|---|---|
| **DWPose (RTMPose, via mmpose/mmcv)** | face landmarks → bbox/crop | `dwpose/dw-ll_ucoco_384.pth` |
| **Whisper (tiny)** | audio → per-frame audio features | `whisper/pytorch_model.bin` |
| **VAE (SD ft-mse)** | pixel face ⇄ latent | `sd-vae/diffusion_pytorch_model.bin` |
| **U-Net (SD-based)** | inpaint masked latent, conditioned on audio | `musetalkV15/unet.pth` |
| **Face parsing (BiSeNet)** | segmentation mask for seamless blending | `face-parse-bisent/79999_iter.pth` |
| **ffmpeg** | encode video + mux audio | system / imageio |

All PyTorch (not ONNX). This is why the local env needs `torch` + the `mmlab` stack.

### How the U-Net is conditioned

```
                 masked-face latent  ┐
                 reference latent    ├─ channel-concat ─► U-Net ──► inpainted latent
Whisper audio features ───────────── cross-attention ──►┘   (ONE forward pass)
```

- The face crop is VAE-encoded twice: once as a **reference** and once with the **mouth region
  masked**. These latents are concatenated as the U-Net's spatial input.
- **Whisper** audio features are injected via **cross-attention** — this is what makes the mouth
  match the sound.
- The U-Net predicts the **inpainted latent** in a single step (no noise schedule / no 20-step
  denoising loop). Trained with perceptual + GAN + sync losses (v1.5).

---

## 2. Flow for our asset (still image + audio)

```
sample_face.png                         sample_audio.wav
      │                                        │
 DWPose landmarks                        Whisper (tiny)
      │                                        │
 face bbox → crop 256×256               per-frame audio features
      │                                        │
 VAE encode:                                   │
  • reference latent                           │
  • masked latent (mouth erased)               │
      └───────────────┬────────────────────────┘
                      ▼
        U-Net inpainting (1 step, per frame)
   concat(latents) + cross-attn(audio) → inpainted latent
                      │
                 VAE decode → 256×256 face with new mouth
                      │
         face-parsing mask → blend mouth into original frame
                      │
                      ▼
        write frames → ffmpeg → H.264 + AAC(original audio) → out.mp4
```

### Step-by-step

1. **Landmarks.** DWPose (an RTMPose model run through mmpose) detects the face and key points on
   `sample_face.png`, giving a stable bounding box / crop. A still image is treated as one frame;
   the number of output frames comes from the audio length (≈158 for our clip).
2. **Audio features.** `sample_audio.wav` is encoded by **Whisper-tiny** into a sequence of audio
   feature vectors, aligned per frame.
3. **VAE encode.** The face crop (256×256) is encoded by the SD VAE into a compact latent — once as
   the **reference**, and once with the **mouth/lower region masked** (the part to synthesize).
4. **U-Net inpaint (single step).** For each frame, the U-Net takes the concatenated
   `[masked latent | reference latent]` and, conditioned on that frame's **Whisper features via
   cross-attention**, predicts the completed latent — the mouth shaped to the audio. One forward
   pass per frame (this is the key speed difference vs LatentSync).
5. **VAE decode.** The latent is decoded back to a 256×256 face image with the new mouth.
6. **Parse + blend.** BiSeNet face-parsing produces a mask so only the mouth/jaw region is pasted
   back into the **original full-resolution frame** — the rest stays pixel-identical (why it looks
   natural and seam-free).
7. **Assemble.** Frames → ffmpeg → video, and the original `sample_audio.wav` is muxed in →
   `outputs/musetalk_demo.mp4`.

---

## 3. How the pieces connect

- **DWPose → VAE:** landmarks define the crop/alignment the VAE and U-Net operate on.
- **Whisper → U-Net (cross-attention):** the only audio pathway; it drives lip shape.
- **VAE encode → U-Net → VAE decode:** all generation happens in latent space; the VAE is the
  bridge to/from pixels.
- **Face parsing → blend:** keeps the edit localized to the mouth/jaw so identity and background
  are untouched.
- **Single-step U-Net:** unlike diffusion, there's no iterative denoising — one pass per frame,
  which is what makes it CPU-viable (~14 min for our clip) and near real-time on a GPU.

---

## 4. Why it looks the way it does

- **Strength:** 256² mouth + native-resolution surrounding face + parsing-based blend → the most
  natural of the three locally (no seam; identity preserved, CSIM ≈0.92 in our runs).
- **Limitation:** with a **still image** the head is frozen (only the mouth moves). Driving it with
  a real **video** gives natural head/eye motion. Heavier deps than Wav2Lip (torch + mmlab). Not
  ONNX/GGUF-able (conv-UNet pipeline) — quantization path is ONNX+INT8 / TensorRT.
