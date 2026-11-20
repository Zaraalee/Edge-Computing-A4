"""
q1_distillation/train_kd.py   (Q1 part 4: students trained WITH knowledge distillation)
---------------------------------------------------------------------------------------
WHAT : trains MobileNetV2, SqueezeNet and EfficientNet-Lite0 on CIFAR-100 using the trained
       ResNet-50 teacher, with the combined loss of kd.py:
           (1 - alpha) * CE(true label) + alpha * T^2 * KL(soft targets) + beta * feature loss
WHY  : the teacher's soft probabilities and intermediate features carry more information than
       the one-hot label, so a small student can learn better than from the labels alone.
HOW  : the teacher is frozen (eval mode, no gradients). For every batch it runs ONCE and gives
       its logits and its 4x4 hint feature map; each student then computes the combined loss
       and is updated. Each student has its own 1x1-conv adapter for feature distillation
       (trained together with the student, thrown away afterwards). Same optimiser, epochs and
       augmentation as train_baseline.py, so the comparison is fair. Resumes automatically.

Run:  python q1_distillation/train_kd.py --epochs 15 --T 4 --alpha 0.7 --beta 1.0
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.data import load, batches
from common.train import to_cl, evaluate_top5
from common.utils import set_seed, CKPT, rel, count_params, file_size_mb
from q1_distillation.models import build, ResNet50, STUDENTS, STUDENT_NAMES
from q1_distillation.kd import DistillationLoss
from q1_distillation.runner import make_optimizer, save_resume, load_resume, log_history

ap = argparse.ArgumentParser()
ap.add_argument("--students", nargs="+", default=list(STUDENTS), choices=list(STUDENTS))
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--T", type=float, default=4.0, help="softmax temperature")
ap.add_argument("--alpha", type=float, default=0.7, help="weight of the soft-target loss")
ap.add_argument("--beta", type=float, default=1.0, help="weight of the feature loss (0 = logits only)")
args = ap.parse_args()

set_seed(0)
xtr, ytr, xte, yte, _ = load("cifar100")

# ---- frozen teacher
ck = torch.load(os.path.join(CKPT, "teacher_resnet50.pth"), map_location="cpu", weights_only=False)
teacher = ResNet50(100)
teacher.load_state_dict(ck["state_dict"])
teacher = teacher.to(memory_format=torch.channels_last).eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"Teacher ResNet-50 loaded (test top-1 {ck['test_top1']:.2f}%), frozen")
print(f"KD settings: T = {args.T}, alpha = {args.alpha} (soft) / {1 - args.alpha:.1f} (hard), beta = {args.beta} (features)")

# ---- students + their distillation losses (each loss owns the student's feature adapter)
models = {n: build(n).to(memory_format=torch.channels_last) for n in args.students}
crits = {n: DistillationLoss(m.hint_channels, teacher.hint_channels, args.T, args.alpha, args.beta)
         .to(memory_format=torch.channels_last) for n, m in models.items()}
steps = (len(xtr) + args.batch - 1) // args.batch
opts, scheds = {}, {}
for n, m in models.items():
    params = list(m.parameters()) + list(crits[n].parameters())   # student + adapter
    opts[n], scheds[n] = make_optimizer(params, args.epochs, steps, lr=args.lr)
    print(f"{STUDENT_NAMES[n]:20s} {count_params(m):>10,} parameters, hint {m.hint_channels} ch -> adapter -> "
          f"{teacher.hint_channels} ch")
run = "kd_" + "_".join(args.students)
mods = {**models, **{f"{n}_adapter": c for n, c in crits.items()}}
start = load_resume(run, mods, opts, scheds)

for ep in range(start + 1, args.epochs + 1):
    for n in models:
        models[n].train(); crits[n].train()
    stats = {n: {"loss": 0.0, "hard": 0.0, "soft": 0.0, "feat": 0.0, "correct": 0, "count": 0} for n in models}
    t0 = time.time()
    for xb, yb in batches(xtr, ytr, "cifar100", args.batch, shuffle=True, train_aug=True):
        xb = to_cl(xb)
        with torch.no_grad():
            t_logits, t_hint = teacher(xb, with_hint=True)     # teacher runs once per batch
        for n, m in models.items():
            s_logits, s_hint = m(xb, with_hint=True)
            loss, parts = crits[n](s_logits, s_hint, t_logits, t_hint, yb)
            opts[n].zero_grad(set_to_none=True); loss.backward(); opts[n].step(); scheds[n].step()
            st = stats[n]; k = len(yb)
            st["loss"] += loss.item() * k; st["count"] += k
            for p in ("hard", "soft", "feat"):
                st[p] += parts[p] * k
            st["correct"] += (s_logits.argmax(1) == yb).sum().item()
    rows = []
    for n, m in models.items():
        top1, top5 = evaluate_top5(m, xte, yte, "cifar100")
        st = stats[n]; c = st["count"]
        print(f"Epoch {ep:2d}/{args.epochs}  {STUDENT_NAMES[n]:20s} loss {st['loss'] / c:.3f} (hard {st['hard'] / c:.3f}, "
              f"soft {st['soft'] / c:.3f}, feat {st['feat'] / c:.3f})  train {100 * st['correct'] / c:5.2f}%  "
              f"test top-1 {top1:5.2f}%  top-5 {top5:5.2f}%", flush=True)
        rows.append({"run": f"{n}_kd", "model": n, "mode": "kd", "epoch": ep, "loss": round(st["loss"] / c, 4),
                     "loss_hard": round(st["hard"] / c, 4), "loss_soft": round(st["soft"] / c, 4),
                     "loss_feat": round(st["feat"] / c, 4), "train_acc": round(100 * st["correct"] / c, 2),
                     "test_top1": round(top1, 2), "test_top5": round(top5, 2),
                     "epoch_min": round((time.time() - t0) / 60, 2)})
    print(f"  epoch time {(time.time() - t0) / 60:.1f} min", flush=True)
    log_history(run, rows)
    save_resume(run, ep, mods, opts, scheds)

print()
for n, m in models.items():
    top1, top5 = evaluate_top5(m, xte, yte, "cifar100")
    path = os.path.join(CKPT, f"student_{n}_kd.pth")
    torch.save({"arch": n, "mode": "kd", "num_classes": 100, "epochs": args.epochs,
                "kd": {"T": args.T, "alpha": args.alpha, "beta": args.beta},
                "test_top1": round(top1, 2), "test_top5": round(top5, 2), "state_dict": m.state_dict()}, path)
    print(f"{STUDENT_NAMES[n]:20s} distilled: top-1 {top1:.2f}%  top-5 {top5:.2f}%   saved -> {rel(path)} "
          f"({file_size_mb(path):.2f} MB)")
