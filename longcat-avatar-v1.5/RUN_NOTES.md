# LongCat-Video-Avatar 1.5 POC — Run Notes

## Scope

This is an isolated upstream-fixture POC for:

```text
image + speech audio + prompt → talking-avatar MP4
```

It uses LongCat Avatar 1.5 `ai2v` mode, not LAM-A2E coefficient inference.

## Environment

| Item | Value |
|---|---|
| Host | `qvac-dev-linux-x64` |
| Source revision | `6b3f4b8582a8bc3f20f795735f5383716c4ba794` |
| GPUs | 2 × NVIDIA RTX 5090, 32,607 MiB each |
| Initial GPU use | GPU 0: 2,277 MiB / 70%; GPU 1: 4,355 MiB / 78% |
| Host memory available | 55 GiB |
| Disk available before download | 435 GiB |
| Isolated Python | 3.11.15 |
| PyTorch | 2.7.1+cu128 |
| FlashAttention | 2.7.4.post1 |
| ONNX Runtime | 1.23.0 |
| Downloaded checkpoint storage | 99 GiB |

The POC uses upstream `assets/avatar/single_example_1.json` without changing
its image, audio, or prompt.

## Runner

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_upstream_fixture.sh
```

It runs the upstream `run_demo_avatar_single_audio_to_video.py` with:

```text
--stage_1=ai2v
--context_parallel_size=2
--use_distill
--model_type=avatar-v1.5
--use_int8
--resolution=480p
--num_segments=1
```

Expected output location:

```text
outputs/upstream-single-fixture/ai2v_demo_1.mp4
```

## Validation outcome

No MP4 was generated on this shared host.

1. The first run stopped before model loading because upstream
   `onnxruntime==1.16.3` requests an executable stack, which this host rejects.
   Updating the isolated POC environment to `onnxruntime==1.23.0` resolved that
   import failure.
2. The retry initialized two-rank context parallelism, loaded the five shared
   text-encoder shards, and began loading the INT8 DiT. It then made no
   progress for nearly 15 minutes. Concurrent GPU and process status commands
   also stopped responding.
3. The POC SSH session was terminated. No conclusion about a model bug or OOM
   is justified from this run; the evidence supports a shared-host resource
   stall while both GPUs were already under substantial utilization.

## Required next run conditions

- Reserve both GPUs for the POC; start only when background GPU utilization is
  near idle.
- Keep both GPU IDs visible: `CUDA_VISIBLE_DEVICES=0,1`.
- Preserve the completed checkpoint directories and isolated environment.
- Retry the same one-segment upstream fixture before testing long continuation
  or custom assets.

## Licensing and asset caution

The upstream repository states MIT terms for its model weights. The project
page separately limits showcase assets derived from real videos to academic
demonstration. The upstream fixture and any generated result are POC-only
until source-asset provenance, consent, privacy, and product distribution
review are complete.
