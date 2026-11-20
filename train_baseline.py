"""
q1_distillation/train_baseline.py   (Q1 part 5: students WITHOUT distillation)
------------------------------------------------------------------------------
WHAT : trains MobileNetV2, SqueezeNet and EfficientNet-Lite0 on CIFAR-100 with ordinary
       cross-entropy on the true labels only. No teacher is involved anywhere in this script.
WHY  : these are the reference results. Distillation is only useful if a distilled student
       beats the same student trained this way, with the same data, epochs and optimiser.
HOW  : identical recipe to train_kd.py (SGD + Nesterov, one-cycle LR peak 0.1, weight decay
       5e-4, batch 128, crop + flip). The three students are independent models; they are
       trained in the same loop only to share the data loading. Resumes automatically.

Run:  python q1_distillation/train_baseline.py --epochs 15
      python q1_distillation/train_baseline.py --students squeezenet --epochs 15   (just one)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from common.data import load, batches
from common.train import to_cl, evaluate_top5
from common.utils import set_seed, CKPT, rel, count_params, file_size_mb
from q1_distillation.models import build, STUDENTS, STUDENT_NAMES
from q1_distillation.runner import make_optimizer, save_resume, load_resume, log_history

ap = argparse.ArgumentParser()
ap.add_argument("--students", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args()

set_seed(0)
xtr, ytr, xte, yte, _ = load("cifar100")
models = {n: build(n).to(memory_format=torch.channels_last) for n in args.students}
steps = (len(xtr) + args.batch - 1) // args.batch
opts, scheds = {}, {}
for n, m in models.items():
    opts[n], scheds[n] = make_optimizer(m.parameters(), args.epochs, steps, lr=args.lr)
    print(f"{STUDENT_NAMES[n]:20s} {count_params(m):>10,} parameters")
run = "baseline_" + "_".join(args.students)
start = load_resume(run, models, opts, scheds)

for ep in range(start + 1, args.epochs + 1):
    for m in models.values():
        m.train()
    stats = {n: [0.0, 0, 0] for n in models}                 # loss sum, correct, count
    t0 = time.time()
    for xb, yb in batches(xtr, ytr, "cifar100", args.batch, shuffle=True, train_aug=True):
        xb = to_cl(xb)
        for n, m in models.items():
            out = m(xb)
            loss = F.cross_entropy(out, yb)                   # hard labels only
            opts[n].zero_grad(set_to_none=True); loss.backward(); opts[n].step(); scheds[n].step()
            stats[n][0] += loss.item() * len(yb); stats[n][1] += (out.argmax(1) == yb).sum().item(); stats[n][2] += len(yb)
    rows = []
    for n, m in models.items():
        top1, top5 = evaluate_top5(m, xte, yte, "cifar100")
        loss, acc = stats[n][0] / stats[n][2], 100 * stats[n][1] / stats[n][2]
        print(f"Epoch {ep:2d}/{args.epochs}  {STUDENT_NAMES[n]:20s} loss {loss:.3f}  train {acc:5.2f}%  "
              f"test top-1 {top1:5.2f}%  top-5 {top5:5.2f}%", flush=True)
        rows.append({"run": f"{n}_baseline", "model": n, "mode": "baseline", "epoch": ep, "loss": round(loss, 4),
                     "train_acc": round(acc, 2), "test_top1": round(top1, 2), "test_top5": round(top5, 2),
                     "epoch_min": round((time.time() - t0) / 60, 2)})
    print(f"  epoch time {(time.time() - t0) / 60:.1f} min", flush=True)
    log_history(run, rows)
    save_resume(run, ep, models, opts, scheds)

print()
for n, m in models.items():
    top1, top5 = evaluate_top5(m, xte, yte, "cifar100")
    path = os.path.join(CKPT, f"student_{n}_baseline.pth")
    torch.save({"arch": n, "mode": "baseline", "num_classes": 100, "epochs": args.epochs,
                "test_top1": round(top1, 2), "test_top5": round(top5, 2), "state_dict": m.state_dict()}, path)
    print(f"{STUDENT_NAMES[n]:20s} baseline: top-1 {top1:.2f}%  top-5 {top5:.2f}%   saved -> {rel(path)} "
          f"({file_size_mb(path):.2f} MB)")
