# FaceOff: Quadro P600 and Intel Arc B580

This checkout adds opt-in profiles to upstream commit
`38fe852cb05b40a3cf06e5c7424efed029bb7aa4` (FaceOff 2.12.0).
Upstream `config.yaml` and the CUDA-13 `uv.lock` remain unchanged. **Do not run
`uv sync` in either portable environment:** it would replace the compatible stack.

## Installation on Windows

Prerequisites: Python 3.12 registered with `py`, FFmpeg and FFprobe on PATH,
current compatible graphics driver, and Microsoft C++ Build Tools for
InsightFace's source build. No system CUDA toolkit is required for the P600
profile; the pinned Torch wheel supplies the CUDA/cuDNN DLLs. These scripts
install only into this checkout's virtual environments.
Select one profile per process and restart when changing profiles.

From this directory in PowerShell:

```powershell
.\scripts\setup-hardware.ps1 -Profile p600
.\.venv-p600\Scripts\python.exe scripts/download_models.py
.\scripts\run-hardware.ps1 -Profile p600 -Verify -OutputDir outputs/p600-validation
```

On the B580 machine, copy this source checkout (including the new files) and
optionally its `inswapper_128.onnx` and `models/buffalo_l` files. Do not copy
Windows virtual environments between machines. Install the Intel environment:

```powershell
.\scripts\setup-hardware.ps1 -Profile b580
.\.venv-b580\Scripts\python.exe scripts/download_models.py
.\scripts\run-hardware.ps1 -Profile b580 -Verify -OutputDir outputs/b580-validation
```

The B580 profile is prepared for later hardware validation. It requires a GPU
whose OpenVINO full device name includes `B580`, selects that device explicitly,
and fails if it is absent or ambiguous. If multiple B580s are installed, set
`runtime.openvino_device` in `profiles/b580.yaml` to the intended `GPU.N` value.
Also set `streaming.qsv_adapter` to the intended DXGI adapter index; these are
different numbering systems. The verifier records the DXGI adapter inventory.
An Intel integrated GPU cannot satisfy this profile by accident.

Some InsightFace/dependency distributions declare the CPU `onnxruntime` package
even when an alternative ORT distribution supplies the runtime. If installed,
the setup script removes the overlapping CPU distribution and reinstalls the
chosen GPU/OpenVINO wheel. In such environments, `pip check` can report a missing
CPU distribution despite a working GPU runtime. Do not install overlapping ORT
distributions to fix that metadata warning; verify actual sessions instead.

## Local processing

No browser or web server is started by either script. Outputs are local files.

```powershell
.\scripts\run-hardware.ps1 -Profile p600 -Source C:/Media/source.jpg -Target C:/Media/target.mp4
.\scripts\run-hardware.ps1 -Profile b580 -Source C:/Media/source.jpg -Target C:/Media/target.jpg
```

Images, GIFs and videos use the same processing pipeline. Provide a static WebP
as an image; convert animated WebP to GIF/video before using this local runner.
The runner processes the target's original dimensions and frame rate. For
review, provide a local 480p MP4; full-resolution delivery uses the original
target after approval. Hardware encoding failure stops the job.

## Supported paths and boundaries

| Stage | P600 profile | B580 profile |
| --- | --- | --- |
| Face analysis / inswapper | CUDA EP, Torch 2.6 cu126, ORT 1.20.2 | OpenVINO EP 1.23, OpenVINO 2025.3 |
| Frame handling | Host RGB; InsightFace BGR conversion at model boundary | Same; no CUDA calls for model execution |
| Decode | FFmpeg CUDA decode, host frames | FFmpeg host decode |
| H.264 output | Required `h264_nvenc`, explicit 6 Mb/s | Required `h264_qsv`, explicit 6 Mb/s |
| TensorRT / PyNvVideoCodec | Disabled | Not installed / disabled |
| Restoration / upscaling | Upstream CUDA implementations, opt-in; not part of swap-only verification | Rejected explicitly; XPU port not validated |

The Intel environment intentionally installs CPU Torch for surrounding utility
code; **face-model inference runs on the Intel GPU through OpenVINO**, not Torch.
The Intel path is not advertised as GPU-accelerated enhancement. Unsupported
enhancement requests fail instead of silently using CPU or an NVIDIA card.

GPU provider initialization is mandatory. Some graph nodes (for example shape
operations) may still run on CPU; the verifier records actual node placement
rather than claiming a CPU-free pipeline. Host-frame conversion and paste are
also CPU operations. There is no cross-vendor multi-GPU scheduling in these
profiles.

Conservative P600 limits are one frame per inference and two host frames per
streaming chunk. `onnx_mem_limit_mb` is a per-session allocator limit, not a
total VRAM cap. No universal throughput or maximum-resolution guarantee is
made. Keep enhancement off for initial tests.

## Verification evidence

`-Verify` uses the repository's existing sample faces and GIF, creates a two-second
480p/12fps motion fixture with audio, performs an image swap and video swap,
checks changed pixels, 24 output frames, retained audio, full decode, actual
hardware encoder diagnostics, and ORT node execution for detection, recognition
and swapping. It saves JSON, logs, raw ORT profiles, image output, MP4 output
and a before/after video frame locally. It exits nonzero on failure, including
missing GPU execution evidence. It does not test all media, resolutions,
long-video memory behavior or perceptual audio quality.

Read the generated `p600-verification.json` or `b580-verification.json` for the
actual result. A prepared profile or downloaded wheel alone does not establish
B580 compatibility. Validate on the B580 before relying on it.

The application's source MIT license does not replace the pretrained models'
licenses. See `NOTICE` for the upstream model restrictions.

## Local result on 2026-11-12

The P600 passed real image and 480p video swaps on Windows with driver 582.78,
Torch 2.6.0+cu126 and ORT 1.20.2. ORT traces recorded CUDA execution in detection,
recognition and swapping. FFmpeg diagnostics identified the Quadro P600 running
`h264_nvenc`. The 24-frame video retained audio and passed full-file decode;
its image/frame comparisons were visually inspected. The focused regression
suite passed all 90 tests, and `pip check` reported no broken requirements.
An additional GIF swap produced 80 frames at the original 260×229 dimensions
and passed a full-file decode.

An initial 768 MiB per-session arena was too small for the swapper. The final
profile uses 1024 MiB and disables cuDNN's maximum-workspace mode. This is a
tested starting configuration, not a guarantee for every workload.

The Intel Windows wheels were downloaded and its package requirements resolved
successfully. Backend isolation and adapter selection have regression tests.
**No B580 inference or encoding was tested here:** the installed cards are a
P600 and UHD 630. Run the B580 verification command on that machine.

The complete local report is `outputs/p600-validation/p600-verification.json`;
related logs, ORT traces, environment snapshot and review files are beside it.
