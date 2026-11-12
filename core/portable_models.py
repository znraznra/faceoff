"""Explicit sessions for the pinned InsightFace 0.7.3 model classes.

Its model_zoo.get_model discards sess_options; constructing these standard
model wrappers directly preserves memory, graph and profiling settings.
"""

from pathlib import Path

import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo.arcface_onnx import ArcFaceONNX
from insightface.model_zoo.retinaface import RetinaFace

from utils.hardware_backend import validate_session


class PortableFaceAnalysis(FaceAnalysis):
    def __init__(self, model_dir, providers, options_factory):
        self.model_dir = str(Path(model_dir).resolve())
        self.models = {}
        for task, filename, wrapper in (
            ("detection", "det_10g.onnx", RetinaFace),
            ("recognition", "w600k_r50.onnx", ArcFaceONNX),
        ):
            path = Path(self.model_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f"Required buffalo_l model missing: {path}")
            session = ort.InferenceSession(
                str(path), sess_options=options_factory(task), providers=providers
            )
            validate_session(session, task)
            self.models[task] = wrapper(model_file=str(path), session=session)
        self.det_model = self.models["detection"]
