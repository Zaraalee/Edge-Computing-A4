"""
common/registry.py
------------------
One place that knows every trained model: its checkpoint file, how to rebuild the network,
which dataset it was trained on and its input resolution. Used by infer.py, export_onnx.py,
onnx_infer.py and benchmark.py, so all of them treat the models in exactly the same way.
"""
import os
import torch
from common.utils import CKPT
from q1_distillation.models import build as build_q1, STUDENT_NAMES
from q2_nas.search_space import build_model as build_nas

STUDENT_MODELS = [f"{s}_{m}" for s in STUDENT_NAMES for m in ("baseline", "kd")]
ALL_MODELS = STUDENT_MODELS + ["nas_best"]
TITLES = {**{f"{s}_baseline": f"{STUDENT_NAMES[s]} (baseline)" for s in STUDENT_NAMES},
          **{f"{s}_kd": f"{STUDENT_NAMES[s]} (distilled)" for s in STUDENT_NAMES},
          "nas_best": "NAS best architecture", "teacher": "ResNet-50 teacher"}


def ckpt_path(name):
    if name == "teacher":
        return os.path.join(CKPT, "teacher_resnet50.pth")
    if name == "nas_best":
        return os.path.join(CKPT, "nas_best_model.pth")
    return os.path.join(CKPT, f"student_{name}.pth")


def load_model(name):
    """Returns (model in eval mode, info dict with dataset / resolution / checkpoint metadata)."""
    ck = torch.load(ckpt_path(name), map_location="cpu", weights_only=False)
    if name == "nas_best":
        model = build_nas(ck["config"], 10)
        info = {"dataset": "cifar10", "resolution": ck["config"]["resolution"], "num_classes": 10}
    else:
        model = build_q1(ck["arch"], 100)
        info = {"dataset": "cifar100", "resolution": 32, "num_classes": 100}
    model.load_state_dict(ck["state_dict"])
    info.update({k: v for k, v in ck.items() if k != "state_dict"})
    return model.eval(), info
