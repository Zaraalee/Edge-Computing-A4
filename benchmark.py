"""
benchmark.py   (Deliverable 11: automated benchmarking of every trained model, PyTorch on CPU)
------------------------------------------------------------------------------------------------
For the teacher, the six students and the NAS model it measures, on the full test set:
  * top-1 / top-5 accuracy               * parameters, MACs per image, checkpoint size (MB)
  * latency of ONE image on ONE thread (median of 5 x 100 runs)   * throughput (batch 500)
  * peak inference memory: measured in a fresh process (Linux) + estimated from layer sizes
and stores every model's prediction for every test image.

Outputs (results/)
  q1_kd_benchmark.csv        teacher + 3 students x (baseline, distilled), with the KD gain
  q1_test_predictions.csv    10,000 CIFAR-100 test images: prediction of every Q1 model
  q2_nas_benchmark.csv       NAS model (+ its constraints)
  q2_test_predictions.csv    10,000 CIFAR-10 test images: prediction of the NAS model
  all_models_benchmark.csv   everything in one table
  q1_kd_comparison.png       baseline vs distilled accuracy chart

Run:  python benchmark.py
"""
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import torch
from common.data import load, batches
from common.registry import load_model, ckpt_path, ALL_MODELS, TITLES
from common.utils import RESULTS, rel, count_params, count_macs, file_size_mb, latency_ms, peak_memory_mb

THREADS = torch.get_num_threads()
names = ["teacher"] + ALL_MODELS
data, rows, preds = {}, [], {}
for name in names:
    if not os.path.exists(ckpt_path(name)):
        print(f"{name}: checkpoint missing - skipped"); continue
    model, info = load_model(name)
    ds, res = info["dataset"], info["resolution"]
    if ds not in data:
        data[ds] = load(ds)
    _, _, xte, yte, classes = data[ds]
    print(f"Benchmarking {TITLES[name]} ...", flush=True)
    torch.set_num_threads(THREADS)
    t, logits = time.perf_counter(), []
    with torch.no_grad():
        for xb, _ in batches(xte, yte, ds, 500, size=res):
            logits.append(model(xb))
    total = time.perf_counter() - t
    logits = torch.cat(logits)
    top5 = logits.topk(5, 1).indices
    preds[name] = top5[:, 0]
    _, mem_est = peak_memory_mb(model, res)
    try:                                   # real peak, measured in a fresh process (Linux)
        mem_meas = float(subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "common", "measure_memory.py"), name], capture_output=True, text=True, check=True).stdout)
    except Exception:
        mem_meas = None
    rows.append({"model": name, "title": TITLES[name], "dataset": ds, "input_res": res,
                 "mode": info.get("mode", "teacher" if name == "teacher" else "nas"),
                 "top1_acc_%": round(100 * (top5[:, 0] == yte).float().mean().item(), 2),
                 "top5_acc_%": round(100 * (top5 == yte[:, None]).any(1).float().mean().item(), 2),
                 "params": count_params(model), "MACs_M": round(count_macs(model, res) / 1e6, 1),
                 "size_MB": round(file_size_mb(ckpt_path(name)), 2),
                 "latency_1img_ms": round(statistics.median(latency_ms(model, res, warmup=30, runs=100) for _ in range(5)), 3),
                 "throughput_img_per_s": round(len(yte) / total, 1),
                 "peak_mem_measured_MB": None if mem_meas is None else round(mem_meas, 2),
                 "peak_mem_estimated_MB": round(mem_est, 2)})
    print("   ", rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "all_models_benchmark.csv"), index=False)

# ---- Q1: baseline vs distilled, KD gain
q1 = df[df.dataset == "cifar100"].copy()
gain = {}
for s in ("mobilenetv2", "squeezenet", "efficientnet_lite0"):
    b, k = q1[q1.model == f"{s}_baseline"], q1[q1.model == f"{s}_kd"]
    if len(b) and len(k):
        gain[f"{s}_kd"] = round(k["top1_acc_%"].iloc[0] - b["top1_acc_%"].iloc[0], 2)
q1["kd_gain_top1"] = q1.model.map(gain)
t = q1[q1.model == "teacher"]
if len(t):
    q1["compression_vs_teacher_x"] = (t["params"].iloc[0] / q1["params"]).round(1)
    q1["speedup_vs_teacher_x"] = (t["latency_1img_ms"].iloc[0] / q1["latency_1img_ms"]).round(2)
q1.to_csv(os.path.join(RESULTS, "q1_kd_benchmark.csv"), index=False)
df[df.dataset == "cifar10"].to_csv(os.path.join(RESULTS, "q2_nas_benchmark.csv"), index=False)


def pred_table(ds, models):
    _, _, _, yte, classes = data[ds]
    tab = pd.DataFrame({"image_id": range(len(yte)), "true_label": [classes[i] for i in yte]})
    for n in models:
        tab[f"{n}_pred"] = [classes[i] for i in preds[n]]
        tab[f"{n}_correct"] = (preds[n] == yte).int().numpy()
    return tab


q1_models = [n for n in names if n in preds and n != "nas_best"]
if q1_models:
    pred_table("cifar100", q1_models).to_csv(os.path.join(RESULTS, "q1_test_predictions.csv"), index=False)
if "nas_best" in preds:
    pred_table("cifar10", ["nas_best"]).to_csv(os.path.join(RESULTS, "q2_test_predictions.csv"), index=False)

# ---- chart: baseline vs distilled students
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
labels = ["MobileNetV2", "SqueezeNet", "EfficientNet-Lite0"]
keys = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
base = [q1.loc[q1.model == f"{k}_baseline", "top1_acc_%"].squeeze() for k in keys]
kd = [q1.loc[q1.model == f"{k}_kd", "top1_acc_%"].squeeze() for k in keys]
if all(isinstance(v, float) for v in base + kd):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    xs = range(len(labels))
    b1 = ax.bar([i - 0.2 for i in xs], base, 0.4, color="#8A949C", label="baseline (no distillation)")
    b2 = ax.bar([i + 0.2 for i in xs], kd, 0.4, color="#2F7FC1", label="distilled from ResNet-50")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2, f"{b.get_height():.2f}", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    if len(t):
        ax.axhline(t["top1_acc_%"].iloc[0], color="#1B4F8A", ls="--", lw=1, label=f"teacher {t['top1_acc_%'].iloc[0]:.2f}%")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels); ax.set_ylabel("CIFAR-100 test top-1 (%)")
    ax.set_ylim(0, max(base + kd + [t["top1_acc_%"].iloc[0] if len(t) else 0]) * 1.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=3); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "q1_kd_comparison.png"), dpi=150); plt.close()

print("\n" + df.drop(columns=["title"]).to_string(index=False))
if gain:
    print("\nKnowledge-distillation gain (top-1, distilled - baseline): "
          + ", ".join(f"{k.replace('_kd', '')} {v:+.2f}" for k, v in gain.items()))
print(f"\nSaved CSVs and charts to {rel(RESULTS)}/")
