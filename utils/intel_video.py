"""Resolve the B580's DXGI adapter for FFmpeg QSV on Windows.

OpenVINO GPU.N indices and DXGI indices are different namespaces. Resolve by
name/vendor independently; never assume either GPU 0 is the discrete Arc.
"""

import sys
from functools import lru_cache


def dxgi_adapters():
    if sys.platform != "win32":
        raise RuntimeError("The prepared B580 QSV profile currently targets Windows")
    import ctypes as c
    import uuid

    class Luid(c.Structure):
        _fields_ = [("low", c.c_uint32), ("high", c.c_int32)]

    class Desc(c.Structure):
        _fields_ = [
            ("description", c.c_wchar * 128),
            ("vendor", c.c_uint32),
            ("device", c.c_uint32),
            ("subsys", c.c_uint32),
            ("revision", c.c_uint32),
            ("video_memory", c.c_size_t),
            ("system_memory", c.c_size_t),
            ("shared_memory", c.c_size_t),
            ("luid", Luid),
            ("flags", c.c_uint32),
        ]

    def method(pointer, index, restype, *types):
        table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
        return c.WINFUNCTYPE(restype, c.c_void_p, *types)(table[index])

    factory = c.c_void_p()
    iid = (c.c_ubyte * 16).from_buffer_copy(
        uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le
    )
    dll = c.WinDLL("dxgi")
    create = dll.CreateDXGIFactory1
    create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
    create.restype = c.c_int32
    if create(c.byref(iid), c.byref(factory)) < 0:
        raise RuntimeError("Cannot enumerate DXGI adapters")
    result = []
    try:
        for index in range(32):
            adapter = c.c_void_p()
            hr = method(factory, 12, c.c_int32, c.c_uint32, c.POINTER(c.c_void_p))(
                factory, index, c.byref(adapter)
            )
            if hr & 0xFFFFFFFF == 0x887A0002:
                break
            if hr < 0:
                raise RuntimeError(f"DXGI adapter enumeration failed: {hr}")
            try:
                desc = Desc()
                if (
                    method(adapter, 10, c.c_int32, c.POINTER(Desc))(
                        adapter, c.byref(desc)
                    )
                    < 0
                ):
                    raise RuntimeError("Cannot read DXGI adapter identity")
                result.append(
                    {
                        "index": index,
                        "name": desc.description,
                        "vendor": desc.vendor,
                        "video_memory": desc.video_memory,
                    }
                )
            finally:
                method(adapter, 2, c.c_uint32)(adapter)
    finally:
        method(factory, 2, c.c_uint32)(factory)
    return result


@lru_cache(maxsize=1)
def qsv_device_args():
    from utils.config_manager import config

    expected = config.get("runtime", "device_name_contains", default="B580")
    matches = [
        a
        for a in dxgi_adapters()
        if a["vendor"] == 0x8086 and expected.lower() in a["name"].lower()
    ]
    requested = config.get("streaming", "qsv_adapter")
    if requested is not None:
        matches = [a for a in matches if a["index"] == int(requested)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Select exactly one Intel QSV adapter matching {expected!r}; found {matches}"
        )
    index = matches[0]["index"]
    return [
        "-init_hw_device",
        f"qsv=faceoff:hw,child_device={index},child_device_type=d3d11va",
        "-filter_hw_device",
        "faceoff",
    ]
