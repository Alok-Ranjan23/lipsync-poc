# Wav2Lip — Architecture & Flow

How the **Wav2Lip** path in this POC works end-to-end, with our exact inputs:
`assets/sample_face.png` (a still portrait) + `assets/sample_audio.wav` (~6 s TTS speech).

Wav2Lip is a **GAN-based, pixel-space mouth generator**. It does *not* generate a whole face —
it only repaints the **mouth region** of an existing face to match audio. Everything else (eyes,
hair, background) is copied from the input.

---

## 1. Components

| Component | Role | In our pipeline |
|---|---|---|
| **Mel extractor** | audio → mel spectrogram (Wav2Lip's exact params) | `wav2lip/audio.py` |
| **Face detector** | find the face box per frame | YuNet ONNX (`yunet.onnx`) |
| **Wav2Lip generator** | (audio + masked face) → new mouth | `wav2lip_gan.onnx` |
| **Blender** | feather the new mouth back into the frame | `_feather_mask` in `infer.py` |
| **GPEN restorer** *(optional)* | sharpen the 96×96 mouth | `gpen_bfr_512.onnx` |
| **ffmpeg** | encode video + mux original audio | `imageio-ffmpeg` |
| **SyncNet** | lip-sync discriminator — **training only**, not used at inference | — |

### The generator itself (the ONNX model)

A convolutional **encoder–decoder GAN** with two encoders that share one decoder:

```
audio (mel 1×80×16) ─► Audio Encoder (2D convs) ─┐
                                                 ├─► concat ─► Face Decoder (up-convs, U-Net skips) ─► mouth 3×96×96
face (6×96×96) ──────► Face Encoder (2D convs) ──┘
   = [ masked lower-half face (3ch) | reference face (3ch) ]
```

- **Audio Encoder** turns the mel chunk into an audio embedding (what the mouth *should say*).
- **Face Encoder** takes 6 channels: the face with its **lower half zeroed** (the mouth to fill)
  **+** the unmasked reference (identity/pose to preserve).
- The two embeddings are concatenated and the **Face Decoder** (transpose-convs with skip
  connections) produces the new **96×96** mouth region.
- Trained with: L1 reconstruction + **SyncNet** lip-sync loss + (in `wav2lip_gan`) a GAN visual
  loss. SyncNet is a pretrained audio–video sync scorer used *only* to supervise training.

---

## 2. Flow for our asset (still image + audio)

```
sample_audio.wav                          sample_face.png
      │                                          │
      ▼                                          ▼
 mel spectrogram                        read as 1 frame (still)
 (80 mel bins)                                   │
      │                                    YuNet face detect
 slice into 16-col chunks,                       │
 ONE chunk per output frame              crop face → resize 96×96
      │                                          │
      │                              build 6-ch = [mask lower half | reference]
      └───────────────┬──────────────────────────┘
                      ▼
        Wav2Lip generator  (per frame, batched)
         (mel chunk, 6-ch face) → 3×96×96 mouth
                      │
             resize mouth → face-box size
             feather-blend into the frame
             (optional) GPEN restore the crop
                      │
                      ▼
        pipe frames → ffmpeg → H.264 + AAC(original audio) → out.mp4
```

### Step-by-step

1. **Audio → mel.** `sample_audio.wav` is loaded at 16 kHz and converted to a mel spectrogram
   using Wav2Lip's exact settings (n_fft=800, hop=200, 80 mels, pre-emphasis, log + normalize).
2. **Mel → per-frame chunks.** At 25 fps, each output frame gets a 16-column mel slice
   (`≈0.2 s` of audio context). The number of frames = audio length × fps (≈158 for our clip).
3. **Face → frames.** The still `sample_face.png` becomes a single frame **reused** for every mel
   chunk (a real video would use its own frames instead).
4. **Detect + crop.** YuNet finds the face box; we crop tight (forehead→chin) and resize to 96×96.
5. **Mask + stack.** The crop's **lower half is zeroed** (mouth erased); this masked crop is
   concatenated channel-wise with the untouched reference crop → a `6×96×96` tensor.
6. **Generate.** For each frame, `(mel[1×80×16], face[6×96×96]) → mouth[3×96×96]`. The audio
   encoder + face encoder feed the shared decoder, which paints a mouth matching that audio slice.
7. **Blend back.** The `96×96` mouth is resized to the original face-box size and **feather-blended**
   in (soft edges → no rectangular seam). Optionally, **GPEN** restores the whole crop at 512 so
   the mouth's sharpness matches the rest of the face.
8. **Assemble.** Frames are streamed into ffmpeg, which encodes H.264 and muxes the original
   `sample_audio.wav` → `outputs/wav2lip_demo.mp4`.

---

## 3. How the pieces connect

- **YuNet → generator:** the detector decides *where* the mouth is; without a good box the mask
  lands wrong (e.g., on the neck).
- **Audio encoder + Face encoder → shared decoder:** audio says *what shape*, the reference says
  *whose mouth / what pose*. The mask forces the decoder to synthesize (not copy) the mouth.
- **Generator → blender:** the model only owns a 96×96 patch; the blender re-integrates it into
  the full-resolution frame.
- **(optional) GPEN → blender:** a separate restorer improves fidelity but never changes lip shape.
- **SyncNet:** disconnected at inference; it shaped the generator's weights during training.

---

## 4. Why it looks the way it does

- **Strength:** tiny, fast, CPU-friendly; strong identity preservation (rest of face untouched).
- **Limitation:** the mouth is generated at **96×96** and upscaled → visibly softer than the face
  (our metrics: mouth sharpness ≈7 vs face ≈55). With a still image there's **no head motion** —
  only the mouth moves → a "talking portrait." GPEN narrows the sharpness gap but not the
  resolution ceiling; MuseTalk/LatentSync address both.
