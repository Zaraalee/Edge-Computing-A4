"""
export_onnx.py   (Deliverable 10: ONNX deployment)
--------------------------------------------------
WHAT : exports the six student models and the NAS model to ONNX (onnx_models/*.onnx).
WHY  : ONNX Runtime runs ONNX files on a CPU, phone or edge board without PyTorch installed.
HOW  : torch.onnx.export (opset 17, dynamic batch size), then
         1. onnx.checker validates the graph,
         2. ONNX Runtime runs it on 64 test images and the outputs are compared with PyTorch
            (max absolute difference of the logits and agreement of the predicted classes).
       Every ONNX model takes the normal 32x32 CIFAR image. If the NAS model was searched at a
       different input resolution, the resize is built into the exported graph.

Run:  python export_onnx.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.data import load, preprocess
from common.registry import load_model, ALL_MODELS, TITLES, ckpt_path
from common.utils import ONNX_DIR, rel


class ResizeThenModel(nn.Module):
    """Takes a 32x32 image, resizes it to the model's input resolution, then classifies it."""

    def __init__(self, model, res):
        super().__init__()
        self.model, self.res = model, res

    def forward(self, x):
        return self.model(F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False))


data = {}
print(f"{'model':<28} {'ONNX file':<46} {'size':>8}   check vs PyTorch")
for name in ALL_MODELS:
    if not os.path.exists(ckpt_path(name)):
        print(f"{name:<28} (checkpoint missing - skipped)")
        continue
    model, info = load_model(name)
    ds, res = info["dataset"], info["resolution"]
    net = ResizeThenModel(model, res).eval() if res != 32 else model
    path = os.path.join(ONNX_DIR, f"{name}.onnx")
    torch.onnx.export(net, torch.randn(1, 3, 32, 32), path, input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(path))
    if ds not in data:
        data[ds] = load(ds)
    x = preprocess(data[ds][2][:64], ds)
    with torch.no_grad():
        ref = net(x).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    diff = np.abs(out - ref).max()
    agree = (out.argmax(1) == ref.argmax(1)).mean() * 100
    print(f"{name:<28} {rel(path):<46} {os.path.getsize(path) / 1e6:6.2f} MB   max |diff| {diff:.1e}, "
          f"same class on {agree:.0f}% of 64 images", flush=True)
print(f"\nAll ONNX models are in {rel(ONNX_DIR)}/")
