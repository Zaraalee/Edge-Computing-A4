"""
make_charts.py - charts for the report, drawn from the CSV/JSON results.
  results/nas_search_scatter.png     every searched architecture: latency vs size, accepted/rejected
  results/nas_search_progress.png    validation accuracy of each trained candidate + best so far
  results/nas_best_training.png      the best architecture trained from scratch
  results/q1_teacher_training.png    teacher accuracy per epoch
  results/q1_student_training.png    baseline vs distilled students, test accuracy per epoch
Run:  python make_charts.py
"""
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
GREY, BLUE, DARK, ORANGE, RED = "#8A949C", "#2F7FC1", "#1B4F8A", "#D98A2B", "#C0392B"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.25)


# ---------------------------------------------------------------- Q2: search
hist_path = os.path.join(RES, "nas_search_history.csv")
if os.path.exists(hist_path):
    h = pd.read_csv(hist_path)
    best = json.load(open(os.path.join(RES, "nas_best_config.json")))
    lim = best["constraints"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    acc, rej = h[h.status == "accepted"], h[h.status == "rejected"]
    ax.scatter(rej.latency_ms, rej.size_MB, marker="x", color=RED, label=f"rejected ({len(rej)})")
    sc = ax.scatter(acc.latency_ms, acc.size_MB, c=acc.val_acc, cmap="Blues", edgecolor=DARK, s=40,
                    label=f"accepted and trained ({len(acc)})")
    b = h[h.id == best["id"]]
    ax.scatter(b.latency_ms, b.size_MB, marker="*", s=260, color=ORANGE, edgecolor="black", zorder=5, label="best")
    ax.axvline(lim["max_latency_ms"], color=GREY, ls="--", lw=1)
    ax.axhline(lim["max_size_mb"], color=GREY, ls="--", lw=1)
    ax.text(lim["max_latency_ms"], ax.get_ylim()[1] * 0.55, f" latency limit {lim['max_latency_ms']} ms", fontsize=8, va="top", rotation=90)
    ax.text(ax.get_xlim()[1] * 0.99, lim["max_size_mb"], f"size limit {lim['max_size_mb']} MB ", fontsize=8, ha="right", va="bottom")
    plt.colorbar(sc, ax=ax, label="validation accuracy (%)")
    ax.set_xlabel("CPU latency, 1 image, 1 thread (ms)"); ax.set_ylabel("model file size (MB)")
    ax.set_title("Hardware-aware search: every candidate"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax); plt.tight_layout(); plt.savefig(os.path.join(RES, "nas_search_scatter.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(7, 3.6))
    acc = acc.reset_index(drop=True)
    colors = [GREY if g == 0 else BLUE for g in acc.generation]
    ax.scatter(range(1, len(acc) + 1), acc.val_acc, c=colors, s=28)
    ax.plot(range(1, len(acc) + 1), acc.val_acc.cummax(), color=ORANGE, lw=2, label="best so far")
    ax.scatter([], [], color=GREY, label="random initial population"); ax.scatter([], [], color=BLUE, label="mutated children")
    ax.set_xlabel("trained candidate (in search order)"); ax.set_ylabel("validation accuracy (%)")
    ax.set_title("Evolutionary search progress"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax); plt.tight_layout(); plt.savefig(os.path.join(RES, "nas_search_progress.png"), dpi=150); plt.close()

nb = os.path.join(RES, "nas_best_training_history.csv")
if os.path.exists(nb):
    d = pd.read_csv(nb)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(d.epoch, d.train_acc, "-o", ms=3, color=GREY, label="train")
    ax.plot(d.epoch, d.test_acc, "-o", ms=3, color=BLUE, label="test")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.set_title("Best NAS architecture trained from scratch (CIFAR-10)")
    ax.legend(frameon=False); style(ax); plt.tight_layout()
    plt.savefig(os.path.join(RES, "nas_best_training.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- Q1: training curves
q1 = os.path.join(RES, "q1_training_history.csv")
if os.path.exists(q1):
    d = pd.read_csv(q1)
    t = d[d.run == "teacher_resnet50"]
    if len(t):
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(t.epoch, t.train_acc, "-o", ms=3, color=GREY, label="train top-1")
        ax.plot(t.epoch, t.test_top1, "-o", ms=3, color=BLUE, label="test top-1")
        ax.plot(t.epoch, t.test_top5, "-o", ms=3, color=DARK, label="test top-5")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.set_title("ResNet-50 teacher on CIFAR-100")
        ax.legend(frameon=False); style(ax); plt.tight_layout()
        plt.savefig(os.path.join(RES, "q1_teacher_training.png"), dpi=150); plt.close()
    names = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet", "efficientnet_lite0": "EfficientNet-Lite0"}
    if any(d.run.str.endswith("_kd")):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
        for ax, (k, title) in zip(axes, names.items()):
            for mode, col in (("baseline", GREY), ("kd", BLUE)):
                r = d[d.run == f"{k}_{mode}"]
                ax.plot(r.epoch, r.test_top1, "-o", ms=3, color=col, label="distilled" if mode == "kd" else "baseline")
            ax.set_title(title); ax.set_xlabel("epoch"); style(ax)
        axes[0].set_ylabel("CIFAR-100 test top-1 (%)"); axes[0].legend(frameon=False)
        plt.tight_layout(); plt.savefig(os.path.join(RES, "q1_student_training.png"), dpi=150); plt.close()
print("charts saved to results/")
