"""Cross-backend regression tests; Intel paths must not touch CUDA."""

import copy
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def intel(monkeypatch):
    from utils.config_manager import config
    from utils.hardware_backend import intel_device

    original = config._config
    monkeypatch.setattr(config, "_config", copy.deepcopy(original))
    config.update("runtime", "backend", value="openvino")
    config.update("runtime", "profile", value="b580")
    config.update("runtime", "require_gpu", value=True)
    config.update("runtime", "device_name_contains", value="B580")
    intel_device.cache_clear()
    yield config
    intel_device.cache_clear()


def test_intel_model_lifecycle_never_selects_cuda(intel, monkeypatch):
    import torch
    from core.model_pool import GPUModelInstance

    def forbidden(*args, **kwargs):
        raise AssertionError("Intel code called CUDA")

    for name in ("set_device", "is_available", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    app = MagicMock()
    swapper = MagicMock()
    app.get.return_value = ["face"]
    swapper.get.side_effect = lambda image, *args, **kwargs: image
    swapper.session.run.return_value = [np.ones((1, 3, 128, 128), np.float32)]
    swapper.input_names = ["target", "source"]
    monkeypatch.setattr(
        GPUModelInstance,
        "_init_face_analysis",
        lambda self: setattr(self, "face_app", app),
    )
    monkeypatch.setattr(GPUModelInstance, "_load_swapper", lambda *args: swapper)
    monkeypatch.setattr("core.model_pool.build_swapper_providers", lambda *args: [])
    instance = GPUModelInstance(0, "unused.onnx")
    rgb = np.full((8, 8, 3), [20, 40, 90], np.uint8)
    assert instance.get_faces(rgb) == ["face"]
    assert app.get.call_args.args[0][0, 0].tolist() == [90, 40, 20]
    assert np.array_equal(instance.swap_face(rgb, "target", "source"), rgb)
    assert instance.run_swapper_batch(
        np.zeros((1, 3, 128, 128)), np.zeros((1, 512))
    ).shape == (1, 3, 128, 128)
    with instance.acquire():
        pass
    instance.cleanup()


def test_intel_ignores_cuda_automatic_workload_flags(intel, monkeypatch):
    from processing.workload_profile import resolve_workload_profile

    monkeypatch.setattr(
        "torch.cuda.is_available", lambda: (_ for _ in ()).throw(AssertionError())
    )
    profile = resolve_workload_profile(
        media_type="video",
        enhance=False,
        enhancement_model="RealESRGAN",
        restore_faces=False,
        face_mappings=None,
    )
    assert profile.name == "intel_host_frames"
    assert not any(
        (
            profile.frame_retention,
            profile.paste_on_gpu,
            profile.zero_copy,
            profile.enhancement_chain,
            profile.use_nvcodec_decode,
        )
    )


def test_failed_gpu_session_is_not_accepted(intel):
    from utils.hardware_backend import validate_session

    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="required OpenVINOExecutionProvider"):
        validate_session(session, "swapper")


def test_intel_enhancement_rejected_before_processing(intel):
    from utils.hardware_backend import validate_features

    with pytest.raises(ValueError, match="face swapping only"):
        validate_features(enhance=True)


def test_intel_provider_has_explicit_gpu(intel, monkeypatch):
    from utils.onnx_providers import (
        build_face_analysis_providers,
        build_swapper_providers,
    )

    monkeypatch.setattr("utils.hardware_backend.intel_device", lambda: "GPU.1")
    for providers in (build_face_analysis_providers(0), build_swapper_providers(0)):
        assert providers[0] == ("OpenVINOExecutionProvider", {"device_type": "GPU.1"})
        assert not any("CUDA" in str(p) for p in providers)


def test_hardware_profile_overlays_without_rewriting_defaults(monkeypatch):
    from utils.config_manager import config

    original = config._config
    monkeypatch.setattr(config, "_config", copy.deepcopy(original))
    monkeypatch.setenv("FACEOFF_PROFILE", "p600")
    config.reload()
    assert config.batch_size == 1
    assert not config.tensorrt_enabled
    assert config.get("streaming", "video_codec") == "h264_nvenc"
    monkeypatch.setenv("FACEOFF_PROFILE", "b580")
    config.reload()
    assert config.get("runtime", "backend") == "openvino"
    assert config.get("streaming", "video_codec") == "h264_qsv"
    assert not config.streaming_nvenc_encode


def test_unknown_profile_fails_closed(monkeypatch):
    from utils.config_manager import config

    monkeypatch.setattr(config, "_config", copy.deepcopy(config._config))
    monkeypatch.setenv("FACEOFF_PROFILE", "typo")
    with pytest.raises(RuntimeError, match="requested hardware profile"):
        config.reload()


def test_nvenc_required_does_not_fall_back(monkeypatch, tmp_path):
    from utils import video_io
    from utils.config_manager import config

    monkeypatch.setattr(config, "_config", copy.deepcopy(config._config))
    config.update("streaming", "encoder_required", value=True)
    monkeypatch.setattr(video_io, "nvenc_available", lambda: False)
    with pytest.raises(RuntimeError, match="Required h264_nvenc"):
        video_io.StreamingVideoWriter(
            tmp_path / "out.mp4", 640, 480, 12, use_nvenc=True
        )


def test_qsv_encoder_never_uses_nvenc_or_crf(intel, monkeypatch, tmp_path):
    from utils import video_io

    intel.update("streaming", "video_bitrate", value="6M")
    proc = MagicMock()
    proc.stderr = None
    monkeypatch.setattr(video_io.subprocess, "Popen", lambda command, **kwargs: proc)
    monkeypatch.setattr(
        "utils.intel_video.qsv_device_args", lambda: ["-init_hw_device", "qsv=test"]
    )
    writer = video_io.StreamingVideoWriter(
        tmp_path / "out.mp4", 640, 480, 12, codec="h264_qsv"
    )
    assert "h264_qsv" in writer.command
    assert "h264_nvenc" not in writer.command
    assert "-crf" not in writer.command
    assert writer.command[writer.command.index("-b:v") + 1] == "6M"
    assert "qsv=test" in writer.command


def test_openvino_selects_arc_instead_of_integrated_gpu(intel, monkeypatch):
    from utils.hardware_backend import intel_device

    core = MagicMock()
    core.available_devices = ["CPU", "GPU.0", "GPU.1"]
    core.get_property.side_effect = lambda device, name: {
        "GPU.0": "Intel UHD Graphics 630",
        "GPU.1": "Intel Arc B580 Graphics",
    }[device]
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    assert intel_device() == "GPU.1"


def test_openvino_rejects_integrated_gpu_only(intel, monkeypatch):
    from utils.hardware_backend import intel_device

    core = MagicMock()
    core.available_devices = ["GPU.0"]
    core.get_property.return_value = "Intel UHD Graphics 630"
    monkeypatch.setitem(sys.modules, "openvino", SimpleNamespace(Core=lambda: core))
    with pytest.raises(RuntimeError, match="matching 'B580'"):
        intel_device()


def test_qsv_uses_dxgi_index_independently_of_openvino(intel, monkeypatch):
    from utils.intel_video import qsv_device_args

    qsv_device_args.cache_clear()
    monkeypatch.setattr(
        "utils.intel_video.dxgi_adapters",
        lambda: [
            {"index": 0, "vendor": 0x8086, "name": "Intel UHD Graphics 630"},
            {"index": 3, "vendor": 0x8086, "name": "Intel Arc B580 Graphics"},
        ],
    )
    try:
        assert "child_device=3" in qsv_device_args()[1]
    finally:
        qsv_device_args.cache_clear()
