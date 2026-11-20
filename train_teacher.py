"""
q1_distillation/train_teacher.py   (Q1 part 1: the teacher)
-----------------------------------------------------------
WHAT : trains a ResNet-50 on CIFAR-100 from scratch and saves checkpoints/teacher_resnet50.pth
WHY  : the teacher is the large, accurate network whose knowledge the small students learn from.
HOW  : SGD (Nesterov momentum 0.9, weight decay 5e-4), one-cycle learning rate (peak 0.1),
       batch 128, random-crop + flip augmentation, label smoothing 0.1. The state is saved
       after every epoch, so the run resumes automatically if interrupted. The model from the
       LAST epoch is kept (the test set is not used to choose an epoch).

Run:  python q1_distillation/train_teacher.py --epochs 20
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.data import load
from common.train import train_epoch, evaluate_top5
from common.utils import set_seed, CKPT, rel, count_params, count_macs, file_size_mb
from q1_distillation.models import ResNet50
from q1_distillation.runner import make_optimizer, save_resume, load_resume, log_history

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args()

set_seed(0)
xtr, ytr, xte, yte, classes = load("cifar100")
print(f"CIFAR-100: {len(xtr)} train / {len(xte)} test images, {len(classes)} classes")
model = ResNet50(100).to(memory_format=torch.channels_last)
print(f"Teacher ResNet-50: {count_params(model):,} parameters, {count_macs(model) / 1e6:.1f} M MACs per image")

steps = (len(xtr) + args.batch - 1) // args.batch
opt, sched = make_optimizer(model.parameters(), args.epochs, steps, lr=args.lr)
start = load_resume("teacher", {"model": model}, {"opt": opt}, {"sched": sched})
for ep in range(start + 1, args.epochs + 1):
    loss, tr_acc, sec = train_epoch(model, xtr, ytr, "cifar100", opt, sched, args.batch, label_smoothing=0.1)
    top1, top5 = evaluate_top5(model, xte, yte, "cifar100")
    print(f"Epoch {ep:2d}/{args.epochs}  loss {loss:.3f}  train {tr_acc:5.2f}%  test top-1 {top1:5.2f}%  "
          f"top-5 {top5:5.2f}%  ({sec / 60:.1f} min)", flush=True)
    log_history("teacher_resnet50", [{"run": "teacher_resnet50", "model": "resnet50", "mode": "teacher", "epoch": ep,
                                      "loss": round(loss, 4), "train_acc": round(tr_acc, 2),
                                      "test_top1": round(top1, 2), "test_top5": round(top5, 2),
                                      "epoch_min": round(sec / 60, 2)}])
    save_resume("teacher", ep, {"model": model}, {"opt": opt}, {"sched": sched})

top1, top5 = evaluate_top5(model, xte, yte, "cifar100")      # final model
path = os.path.join(CKPT, "teacher_resnet50.pth")
torch.save({"arch": "resnet50", "num_classes": 100, "epochs": args.epochs,
            "test_top1": round(top1, 2), "test_top5": round(top5, 2), "state_dict": model.state_dict()}, path)
print(f"\nFinal teacher accuracy: top-1 {top1:.2f}%  top-5 {top5:.2f}%")
print(f"Saved -> {rel(path)}  ({file_size_mb(path):.1f} MB)")
