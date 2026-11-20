"""
q2_nas/train_best.py   (Q2 part 7: train the best discovered architecture from scratch)
---------------------------------------------------------------------------------------
WHAT : reads results/nas_best_config.json (written by the search), builds that architecture with
       NEW random weights and trains it properly on the full CIFAR-10 training set.
WHY  : during the search every candidate is trained only briefly on a subset, which is enough
       to RANK architectures but not to reach their real accuracy. The winner is retrained
       from scratch with a full schedule; nothing is reused from the search.
HOW  : 50,000 training images, SGD + Nesterov, one-cycle LR (peak 0.1), weight decay 5e-4,
       crop + flip. Final accuracy on the 10,000 test images (used here for the first time).
       Saves checkpoints/nas_best_model.pth with the config inside, so it can be rebuilt.

Run:  python q2_nas/train_best.py --epochs 30
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
from common.data import load
from common.train import train_epoch, evaluate
from common.utils import set_seed, CKPT, RESULTS, rel, file_size_mb
from q2_nas.search_space import build_model, describe, key
from q2_nas import hardware

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=0.1)
args = ap.parse_args()

best = json.load(open(os.path.join(RESULTS, "nas_best_config.json")))
cfg, res = best["config"], best["config"]["resolution"]
print(f"Best architecture from the search (#{best['id']}, val acc {best['val_acc']}% after the short search training):")
print("  " + describe(cfg))

set_seed(0)
xtr, ytr, xte, yte, _ = load("cifar10")
model = build_model(cfg).to(memory_format=torch.channels_last)        # fresh random weights
steps = (len(xtr) + 127) // 128
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps,
                                            pct_start=0.25, div_factor=10, final_div_factor=100)
rows = []
for ep in range(1, args.epochs + 1):
    loss, acc, sec = train_epoch(model, xtr, ytr, "cifar10", opt, sched, size=res)
    te = evaluate(model, xte, yte, "cifar10", res)
    rows.append({"epoch": ep, "loss": round(loss, 4), "train_acc": round(acc, 2), "test_acc": round(te, 2),
                 "epoch_s": round(sec, 1)})
    print(f"Epoch {ep:2d}/{args.epochs}  loss {loss:.3f}  train {acc:5.2f}%  test {te:5.2f}%  ({sec:.0f}s)", flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "nas_best_training_history.csv"), index=False)

model = model.to(memory_format=torch.contiguous_format)
hw = hardware.measure(model, res)
path = os.path.join(CKPT, "nas_best_model.pth")
torch.save({"config": cfg, "arch": key(cfg), "num_classes": 10, "epochs": args.epochs, "test_acc": round(te, 2),
            "hardware": hw, "state_dict": model.state_dict()}, path)
print(f"\nFinal test accuracy (CIFAR-10, 10,000 images): {te:.2f}%")
print(f"Hardware: latency {hw['latency_ms']:.2f} ms (1 image, 1 thread), size {hw['size_MB']:.3f} MB, "
      f"{hw['params']:,} params, {hw['MACs_M']} M MACs, peak memory (est.) {hw['peak_mem_estimated_MB']:.2f} MB")
ok, why = hardware.check(hw, best["constraints"])
print(f"Constraints {best['constraints']}: {'satisfied' if ok else 'VIOLATED: ' + why}")
print(f"Saved -> {rel(path)} ({file_size_mb(path):.3f} MB)")
