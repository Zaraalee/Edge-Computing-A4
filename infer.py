"""
infer.py   (Deliverable 9: unified PyTorch inference for all student models and the NAS model)
------------------------------------------------------------------------------------------------
Loads any trained model by name, applies the same preprocessing as in training and prints the
top-3 classes with their probabilities.

Models: mobilenetv2_baseline, mobilenetv2_kd, squeezenet_baseline, squeezenet_kd,
        efficientnet_lite0_baseline, efficientnet_lite0_kd   (CIFAR-100, 100 classes)
        nas_best                                             (CIFAR-10, 10 classes)
        all  (= every model above)

Examples
  python infer.py --model mobilenetv2_kd --index 0 1 2       # test images by index
  python infer.py --model nas_best --image my_photo.jpg      # your own image (resized)
  python infer.py --model all --testset                      # test accuracy of every model
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from common.data import load, preprocess
from common.registry import load_model, ALL_MODELS, TITLES
from common.train import evaluate_top5

ap = argparse.ArgumentParser()
ap.add_argument("--model", nargs="+", default=["mobilenetv2_kd"], choices=ALL_MODELS + ["all"])
ap.add_argument("--index", type=int, nargs="*", default=[])
ap.add_argument("--image", nargs="*", default=[])
ap.add_argument("--testset", action="store_true", help="accuracy on the whole test set")
args = ap.parse_args()
names = ALL_MODELS if "all" in args.model else args.model

data = {}
for name in names:
    model, info = load_model(name)
    ds, res = info["dataset"], info["resolution"]
    if ds not in data:
        data[ds] = load(ds)
    _, _, xte, yte, classes = data[ds]
    print(f"=== {TITLES[name]}  [{name}]  ({ds}, input {res}x{res}) ===")
    if args.testset:
        top1, top5 = evaluate_top5(model, xte, yte, ds, res)
        print(f"  test accuracy on {len(xte):,} images: top-1 {top1:.2f}%  top-5 {top5:.2f}%")
    imgs, truth, titles = [], [], []
    for i in args.index:
        imgs.append(xte[i]); truth.append(classes[yte[i]]); titles.append(f"test[{i}]")
    for p in args.image:
        imgs.append(torch.from_numpy(np.array(Image.open(p).convert("RGB").resize((32, 32)))))
        truth.append("?"); titles.append(os.path.basename(p))
    if imgs:
        x = preprocess(torch.stack(imgs), ds)
        if res != 32:
            x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
        with torch.no_grad():
            prob = F.softmax(model(x), 1)
        for k, p in enumerate(prob):
            top = p.topk(3)
            guesses = ", ".join(f"{classes[c]} {v * 100:.1f}%" for v, c in zip(top.values, top.indices))
            mark = "correct" if classes[top.indices[0]] == truth[k] else ("" if truth[k] == "?" else "wrong")
            print(f"  {titles[k]:>12} | true: {truth[k]:<14} | predicted: {classes[top.indices[0]]:<14} {mark:<7} | top-3: {guesses}")
