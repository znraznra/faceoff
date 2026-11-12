"""Hardware identity and strict provider selection for opt-in portable profiles.

ONNX inference uses CUDA or OpenVINO. Intel frames stay in host memory;
CUDA-only frame retention and enhancement are deliberately excluded there.
"""

from functools import lru_cache

from utils.config_manager import config


def backend_name():
    name = config.get("runtime", "backend", default="cuda")
    if name not in ("cuda", "openvino"):
        raise ValueError(f"Unsupported inference backend: {name}")
    return name


def uses_cuda():
    return backend_name() == "cuda"


@lru_cache(maxsize=1)
def intel_device():
    import openvino as ov

    core = ov.Core()
    requested = config.get("runtime", "openvino_device", default="GPU")
    expected = config.get("runtime", "device_name_contains", default="")
    devices = [d for d in core.available_devices if d.startswith("GPU")]
    matches = [
        d
        for d in devices
        if expected.lower() in core.get_property(d, "FULL_DEVICE_NAME").lower()
    ]
    if requested != "GPU":
        matches = [d for d in matches if d == requested]
    if len(matches) != 1:
        found = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices}
        raise RuntimeError(
            f"Select exactly one Intel GPU matching {expected!r}; found {found}"
        )
    return matches[0]


def intel_providers():
    return [
        ("OpenVINOExecutionProvider", {"device_type": intel_device()}),
        "CPUExecutionProvider",
    ]


def validate_session(session, label):
    """Fail if the requested GPU provider failed initialization.

    This checks session registration, not individual node placement. The
    validation script records ORT node profiling to distinguish those claims.
    """
    if not config.get("runtime", "require_gpu", default=False):
        return
    expected = "CUDAExecutionProvider" if uses_cuda() else "OpenVINOExecutionProvider"
    if expected not in session.get_providers():
        raise RuntimeError(
            f"{label}: required {expected} unavailable; active={session.get_providers()}"
        )


def validate_features(enhance=False, restore_faces=False):
    if not uses_cuda() and (enhance or restore_faces):
        raise ValueError(
            "The Intel profile currently supports face swapping only. "
            "Restoration/upscaling require a separately validated XPU port."
        )
