"""
onnx_infer.py   (Deliverable 10: CPU inference with ONNX Runtime)
-----------------------------------------------------------------
Runs the exported models with ONNX Runtime on the CPU. The model itself needs only numpy +
onnxruntime (+ Pillow for your own images); the CIFAR test set is read through common/data.py.

Examples
  python onnx_infer.py --model squeezenet_kd --index 0 1 2
  python onnx_infer.py --model nas_best --image my_photo.jpg
  python onnx_infer.py --benchmark          # every ONNX model on the full test set -> CSV
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import onnxruntime as ort
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.path.join(HERE, "onnx_models")
STATS = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
         "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))}
MODELS = ["mobilenetv2_baseline", "mobilenetv2_kd", "squeezenet_baseline", "squeezenet_kd",
          "efficientnet_lite0_baseline", "efficientnet_lite0_kd", "nas_best"]


def dataset_of(name):
    return "cifar10" if name == "nas_best" else "cifar100"


def preprocess(imgs_uint8_nhwc, ds):
    """uint8 NHWC -> normalised float32 NCHW, exactly as in training."""
    mean, std = (np.array(v, dtype=np.float32).reshape(1, 3, 1, 1) for v in STATS[ds])
    x = imgs_uint8_nhwc.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return ((x - mean) / std).astype(np.float32)


def session(name, threads=0):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(os.path.join(ONNX_DIR, name + ".onnx"), so, providers=["CPUExecutionProvider"])


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv2_kd", choices=MODELS)
    ap.add_argument("--index", type=int, nargs="*", default=[])
    ap.add_argument("--image", nargs="*", default=[])
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    from common.data import load                    # only used to read the CIFAR test sets
    cache = {}

    def test_set(ds):
        if ds not in cache:
            _, _, xte, yte, classes = load(ds)
            cache[ds] = (xte.numpy(), yte.numpy(), classes)
        return cache[ds]

    if args.benchmark:
        import pandas as pd
        rows = []
        for n in [m for m in MODELS if os.path.exists(os.path.join(ONNX_DIR, m + ".onnx"))]:
            ds = dataset_of(n)
            xte, yte, _ = test_set(ds)
            x_all = preprocess(xte, ds)
            s = session(n)
            t = time.perf_counter()
            logits = np.concatenate([s.run(None, {"input": x_all[i:i + 500]})[0] for i in range(0, len(x_all), 500)])
            total = time.perf_counter() - t
            top5 = np.argsort(-logits, 1)[:, :5]
            s1 = session(n, threads=1)                  # one image, one thread (edge setting)
            x1 = x_all[:1]
            for _ in range(20):
                s1.run(None, {"input": x1})
            t = time.perf_counter()
            for _ in range(200):
                s1.run(None, {"input": x1})
            lat = (time.perf_counter() - t) / 200 * 1000
            row = {"onnx_model": n, "dataset": ds, "top1_acc_%": round(float((top5[:, 0] == yte).mean() * 100), 2),
                   "top5_acc_%": round(float((top5 == yte[:, None]).any(1).mean() * 100), 2),
                   "size_MB": round(os.path.getsize(os.path.join(ONNX_DIR, n + ".onnx")) / 1e6, 2),
                   "latency_1img_ms": round(lat, 3), "throughput_img_per_s": round(len(yte) / total, 1)}
            rows.append(row)
            print(row, flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(HERE, "results", "onnx_benchmark.csv"), index=False)
        print("\n" + df.to_string(index=False))
        print("\nSaved results/onnx_benchmark.csv")
    else:
        ds = dataset_of(args.model)
        xte, yte, classes = test_set(ds)
        imgs, truth, titles = [], [], []
        for i in args.index or ([] if args.image else [0]):
            imgs.append(xte[i]); truth.append(classes[yte[i]]); titles.append(f"test[{i}]")
        for p in args.image:
            imgs.append(np.array(Image.open(p).convert("RGB").resize((32, 32)))); truth.append("?"); titles.append(p)
        prob = softmax(session(args.model).run(None, {"input": preprocess(np.stack(imgs), ds)})[0])
        print(f"ONNX Runtime {ort.__version__} | model: {args.model}.onnx | provider: CPU | dataset: {ds}")
        for k, p in enumerate(prob):
            top = p.argsort()[::-1][:3]
            print(f"{titles[k]:>12} | true: {truth[k]:<14} | predicted: {classes[top[0]]:<14} | top-3: "
                  + ", ".join(f"{classes[c]} {p[c] * 100:.1f}%" for c in top))
