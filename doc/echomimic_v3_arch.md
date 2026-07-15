# EchoMimicV3 — Architecture & Flow

How the **EchoMimicV3** path works end-to-end with a portrait image and speech
audio.

EchoMimicV3 is an **audio-conditioned image-to-video latent generative model**.
It is built on the **Wan2.1-Fun Inpainting 1.3B** video foundation model and
generates pixels directly; it does **not** produce ARKit-52 coefficients, a
facial rig, VRM output, or a reusable 3D avatar.

---

## 1. Components

| Component | Role | POC checkpoint / implementation |
|---|---|---|
| **Wan VAE** | image/video pixels ⇄ compressed video latents | `Wan2.1-Fun-V1.1-1.3B-InP` |
| **Wan audio-mask 3D Transformer (DiT)** | iteratively generates video latents from image, audio, text, and task-mask conditions | EchoMimicV3 Flash or Preview transformer checkpoint |
| **Wav2Vec2** | speech audio → temporally aligned audio features | `chinese-wav2vec2-base` (Flash), `wav2vec2-base-960h` (Preview) |
| **CLIP image encoder** | reference portrait → image-condition features | included by the Wan base pipeline |
| **UMT5 text encoder** | optional prompt → text-condition features | included by the Wan base pipeline |
| **Scheduler / sampler** | performs the iterative latent generation trajectory | Flow UniPC (Flash); Preview uses its upstream long-video configuration |
| **MP4 writer / audio muxer** | writes generated frames with the supplied audio track | upstream EchoMimicV3 pipeline |

### How the transformer is conditioned

```text
reference image ─► VAE latent + image features ─┐
speech audio ────► Wav2Vec2 → aligned segments ─┼─► masked Wan 3D Transformer
text prompt ─────► UMT5 text features ──────────┤       ↓ iterative generation
task mask ───────► temporal/spatial mask ───────┘    generated video latent
                                                            ↓
                                                     Wan VAE decode
                                                            ↓
                                               frames + original audio → MP4
```

- The Wan VAE compresses the reference image and generated frames to latent
  space, so generation happens on compact video latents rather than pixels.
- The model uses a **masked-inpainting formulation**: the task is expressed as
  a 0/1 temporal-spatial mask concatenated with the latent input. For lip-sync,
  the reference image identity is retained while the video is generated to
  match speech.
- Wav2Vec2 produces speech features. EchoMimicV3 divides those features into
  time-local segments, aligns them to latent-frame time, and injects them with
  audio cross-attention. A face-region attention mask gives audio conditioning
  stronger influence on facial motion.
- Image, text, and audio conditions are fused with **Coupled-Decoupled
  Multi-Modal Cross Attention (CDCA)**. They have separate key/value
  projections but a shared transformer query, then are combined with
  timestep-dependent weights.
- The scheduler repeatedly updates the latent. This is generative video
  inference, unlike LAM-A2E's single audio-to-coefficient regression pass.

---

## 2. Flow for our asset

```text
portrait.png ──► image preprocessing ─► CLIP + Wan VAE features ─┐
                                                                    │
speech.wav ───► Wav2Vec2 ─► time-aligned audio segments ──────────┼─►
                                                                    │
prompt ───────► UMT5 text features ────────────────────────────────┤
                                                                    ▼
                                  Wan 3D Transformer + task mask + iterative sampler
                                                                    │
                                                         Wan VAE decode
                                                                    │
                                                       frames + audio → out.mp4
```

### Step-by-step

1. **Validate and load inputs.** The runner receives a portrait image, speech
   audio, an output `.mp4` path, and an optional text prompt.
2. **Create conditions.** The image is encoded for identity/appearance, audio
   is encoded with Wav2Vec2, and the prompt is encoded by UMT5. The task mask
   tells the Wan inpainting model which temporal-spatial content to generate.
3. **Align audio to video time.** Wav2Vec2 features are segmented to match the
   VAE's temporal downsampling and latent-frame positions. Each generated time
   region receives local speech context rather than one global audio vector.
4. **Generate the latent video.** The Wan 3D Transformer starts from its
   sampled latent state and follows the configured flow/diffusion-style
   scheduler for multiple steps. CDCA combines the image, audio, and text
   conditions at each transformer timestep.
5. **Decode and save.** The Wan VAE decodes the final latent video into frames.
   The upstream pipeline writes an MP4 containing those frames and the input
   audio track.

---

## 3. Flash and Preview paths in this POC

| | Flash (`run_image_audio.sh`) | Preview long-video (`run_image_audio_long.sh`) |
|---|---|---|
| Primary use | short talking clip | longer talking clip |
| Audio encoder | `chinese-wav2vec2-base` | `wav2vec2-base-960h` |
| Video extent | 81 frames (about 3.24 seconds at 25 FPS) | 81-frame chunks with 8-frame overlap |
| Resolution | 768×768 | upstream Preview configuration |
| Inference configuration | Flow UniPC, 8 steps, TeaCache enabled | upstream long-video CFG with memory-managed execution |
| Memory behavior | direct GPU allocation; faster, but needs more free VRAM | `mmgp` offloads model portions between GPU and system memory; slower |

`run_image_audio_auto.sh --mode auto` chooses Flash up to 81 frames and the
Preview long-video path for longer WAV inputs. `--mode short` and `--mode long`
override that selection.

### Long-video continuity

Preview cannot generate arbitrary-duration video in one fixed latent window.
It generates overlapping windows and uses EchoMimicV3's long-video CFG
calculation to blend guidance in the overlap. This reduces identity, colour,
and motion discontinuities at chunk boundaries, at the cost of repeated work
and longer wall-clock time.

---

## 4. Training-only features

These are described by the EchoMimicV3 paper but are not separate runtime
models loaded by our POC:

- **Soup-of-Tasks:** represents text-to-video, image-to-video,
  first/last-frame-to-video, and lip-sync as different masked-reconstruction
  tasks in one transformer.
- **Soup-of-Modals / CDCA:** trains the model to combine text, image, and
  audio conditions without using a separate model for each modality.
- **Negative DPO:** training refinement that penalizes undesirable generated
  samples.
- **Phase-aware Negative CFG:** inference guidance strategy for rejecting
  different artifact types at different generation phases.

---

## 5. What this path is and is not

- **Strength:** direct portrait + audio → talking MP4. It can synthesize face
  and body motion without a rigged 3D avatar, ARKit mapping, or external
  renderer.
- **Cost:** iterative video generation is GPU- and memory-intensive. Flash is
  short but fast; Preview enables long audio through chunking and offloading,
  not through a single unlimited generation.
- **Output:** a rendered 2D video, not reusable facial-animation data. Editing
  the avatar, retargeting to another rig, or using the result in a game engine
  needs a different asset/animation workflow.
- **Runtime scope of this POC:** Python, PyTorch, and CUDA. This is not a
  cross-platform QVAC native inference implementation.

---

## Sources

- [EchoMimicV3 paper](https://arxiv.org/abs/2507.03905)
- [EchoMimicV3 upstream repository](https://github.com/antgroup/echomimic_v3)
- Local POC runners: `echomimic-v3/run_image_audio.sh` and
  `echomimic-v3/run_preview_long_offload.py`
