"""Local file processing and reproducible GPU validation; never starts a web UI."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def checked(command):
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise RuntimeError(f"Command failed: {command}\n{result.stderr[-4000:]}")
    return result.stdout


def run_job(source, target, output, enhance=False, restore_faces=False):
    import numpy as np
    from PIL import Image
    from processing.orchestrator import ProcessOptions, process_media

    suffix = target.suffix.lower()
    media_type = (
        "gif"
        if suffix == ".gif"
        else (
            "image" if suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp") else "video"
        )
    )
    options = ProcessOptions(
        source_image=np.array(Image.open(source).convert("RGB")),
        dest_path=str(target),
        media_type=media_type,
        output_dir=str(output),
        enhance=enhance,
        restore_faces=restore_faces,
        use_fp32=True,
        model_name="RealESRGAN_x2plus",
        outscale=2,
        tile_size=128,
    )
    result = process_media(options)
    path = next((Path(p) for p in result if p), None)
    if path is None or not path.is_file():
        raise RuntimeError(f"Processing produced no output: {result}")
    return path


def verify(profile, output, report):
    import numpy as np
    import onnxruntime as ort
    import torch
    from PIL import Image
    from utils.hardware_backend import intel_device
    from core.model_pool import get_model_pool
    from utils.config_manager import config

    report["versions"] = {
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "python": sys.version,
        "cuda": torch.version.cuda,
    }
    report["available_providers"] = ort.get_available_providers()
    if profile == "p600":
        report["gpu"] = checked(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv",
            ]
        ).strip()
        report["torch_architectures"] = torch.cuda.get_arch_list()
        # is_available alone doesn't prove executable kernels on Pascal.
        x = torch.arange(256, dtype=torch.float32, device="cuda:0")
        assert float((x * 2).sum().cpu()) == 65280.0
        torch.cuda.synchronize()
    else:
        import openvino as ov

        core = ov.Core()
        device = intel_device()
        report["gpu"] = core.get_property(device, "FULL_DEVICE_NAME")
        report["openvino"] = ov.__version__
        report["openvino_device"] = device

    os.environ["FACEOFF_ORT_PROFILE_DIR"] = str(output / "ort")
    source = ROOT / "assets" / "arnold.jpg"
    target = ROOT / "assets" / "elon.webp"
    report["fixture_sources"] = [
        str(source),
        str(target),
        str(ROOT / "assets/nacho-libre.gif"),
    ]
    started = time.perf_counter()
    image_path = run_job(source, target, output)
    report["image"] = str(image_path)
    before = np.array(Image.open(target).convert("RGB"))
    after = np.array(Image.open(image_path).convert("RGB"))
    difference = np.abs(before.astype(float) - after.astype(float))
    report["image_mean_absolute_change"] = float(difference.mean())
    if before.shape != after.shape or report["image_mean_absolute_change"] <= 0.1:
        raise RuntimeError("Image output unchanged or wrong dimensions")

    instance = get_model_pool().get_instance(0, use_tensorrt=False)
    source_faces = instance.get_faces(np.array(Image.open(source).convert("RGB")))
    if not source_faces:
        raise RuntimeError("Fixture source face was not detected")
    report["source_faces"] = len(source_faces)
    report["image_seconds_including_model_load"] = time.perf_counter() - started

    # Motion fixture plus a known tone exercises streaming, audio mux and encode.
    input_video = output / "input-480p.mp4"
    encoder = "h264_nvenc" if profile == "p600" else "h264_qsv"
    report["fixture_encoder"] = encoder
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "verbose",
        "-stream_loop",
        "-1",
        "-i",
        str(ROOT / "assets/nacho-libre.gif"),
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000",
        "-t",
        "2",
        "-vf",
        "scale=-2:480,fps=12",
        "-c:v",
        encoder,
        "-b:v",
        "6M",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        str(input_video),
    ]
    if profile == "b580":
        from utils.intel_video import dxgi_adapters, qsv_device_args

        report["dxgi_adapters"] = dxgi_adapters()
        command[1:1] = qsv_device_args()
        command[command.index("-vf") + 1] += ",format=nv12,hwupload=extra_hw_frames=16"
        command[command.index("-pix_fmt") + 1] = "qsv"
    proc = subprocess.run(command, capture_output=True, text=True, errors="replace")
    (output / "fixture-encode.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(f"{encoder} hardware encode failed: {proc.stderr[-3000:]}")
    started = time.perf_counter()
    video_path = run_job(source, input_video, output)
    report["video"] = str(video_path)
    report["video_seconds"] = time.perf_counter() - started
    metadata = json.loads(
        checked(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(video_path),
            ]
        )
    )
    report["video_streams"] = metadata["streams"]
    video = next(s for s in metadata["streams"] if s["codec_type"] == "video")
    if int(video["nb_read_frames"]) != 24 or video["height"] != 480:
        raise RuntimeError("Unexpected video frame count/dimensions")
    if not any(s["codec_type"] == "audio" for s in metadata["streams"]):
        raise RuntimeError("Output audio missing")
    checked(["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"])
    report["full_video_decode"] = "passed"
    # Check that streaming actually swapped frames, rather than copying input.
    frame_arrays = []
    for path in (input_video, video_path):
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        ).stdout
        frame_arrays.append(
            np.frombuffer(raw, dtype=np.uint8).reshape(
                24, video["height"], video["width"], 3
            )
        )
    changes = np.abs(
        frame_arrays[0].astype(float) - frame_arrays[1].astype(float)
    ).mean(axis=(1, 2, 3))
    report["video_frame_mean_changes"] = changes.tolist()
    if not np.any(changes > 1.0):
        raise RuntimeError("Streaming output appears unchanged")
    montage = np.concatenate([frame_arrays[0][12], frame_arrays[1][12]], axis=1)
    Image.fromarray(montage).save(output / "video-before-after.png")
    report["video_comparison"] = str(output / "video-before-after.png")
    encode_log = video_path.with_suffix(".ffmpeg.log")
    if not encode_log.is_file() or encoder not in encode_log.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError("Actual encoder diagnostic evidence is missing")
    report["encoder_evidence"] = str(encode_log)

    sessions = {
        **{task: model.session for task, model in instance.face_app.models.items()},
        "swapper": instance.swapper.session,
    }
    expected = (
        "CUDAExecutionProvider" if profile == "p600" else "OpenVINOExecutionProvider"
    )
    report["inference"] = {}
    for task, session in sessions.items():
        trace = session.end_profiling()
        events = json.loads(Path(trace).read_text(encoding="utf-8"))
        counts = collections.Counter(
            e.get("args", {}).get("provider")
            for e in events
            if e.get("args", {}).get("provider")
        )
        report["inference"][task] = {
            "providers": session.get_providers(),
            "node_events": dict(counts),
            "trace": trace,
        }
        if counts[expected] == 0:
            raise RuntimeError(f"No {expected} execution recorded for {task}")
    report["model_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.file_digest(
            p.open("rb"), "sha256"
        ).hexdigest()
        for p in [
            ROOT / config.inswapper_model_path,
            *sorted((ROOT / config.buffalo_model_path).glob("*.onnx")),
        ]
    }
    report["status"] = "passed"
    report["limits"] = [
        "Short repository-fixture tests only; no long-video benchmark.",
        "Intel enhancement/restoration intentionally unavailable.",
        "Audio stream and decode checked; no full perceptual listening claim.",
    ]


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=("p600", "b580"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local"))
    parser.add_argument("--enhance", action="store_true")
    parser.add_argument("--restore-faces", action="store_true")
    args = parser.parse_args()
    if not args.verify and (args.source is None or args.target is None):
        parser.error("--source and --target are required without --verify")
    os.environ["FACEOFF_PROFILE"] = args.profile
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "run.log", encoding="utf-8"),
        ],
    )
    from datetime import datetime, timezone

    report = {
        "profile": args.profile,
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }
    try:
        from utils.runtime_bootstrap import bootstrap

        bootstrap()
        if args.verify:
            verify(args.profile, output, report)
        else:
            path = run_job(
                args.source.resolve(),
                args.target.resolve(),
                output,
                args.enhance,
                args.restore_faces,
            )
            report.update(status="passed", output=str(path))
    except Exception as exc:
        report["error"] = str(exc)
        logging.exception("FaceOff failed")
    finally:
        destination = output / (
            f"{args.profile}-verification.json" if args.verify else "last-job.json"
        )
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report: {destination}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
