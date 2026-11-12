"""Install upstream BasicSR compatibility shims without launching Gradio."""

import os
import sys
from types import ModuleType


def bootstrap():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from torchvision.transforms.functional import rgb_to_grayscale

    module = ModuleType("torchvision.transforms.functional_tensor")
    module.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = module
    from utils.legacy_torch import install_legacy_torch_load_shim

    install_legacy_torch_load_shim()
