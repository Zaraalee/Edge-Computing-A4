"""
q2_nas/evolution_search.py   (Q2 parts 3-6: NAS algorithm + hardware-aware search pipeline)
------------------------------------------------------------------------------------------
WHAT : regularized (aging) evolutionary search over the space in search_space.py, on CIFAR-10.
WHY  : trying all 2.5 million architectures is impossible; evolution keeps a population of good
       architectures and improves it by small mutations, so it finds a strong one with only a
       few dozen trained candidates.
HOW  : for every candidate architecture
        1. generate the network from its config                       (search_space.build_model)
        2. measure CPU latency (1 image, 1 thread), file size and peak memory      (hardware.py)
        3. REJECT it immediately if it breaks the deployment constraints (no training wasted)
        4. otherwise train it briefly and measure validation accuracy (fitness)
        5. log everything to results/nas_search_history.csv and .json
       Evolution loop (Real et al., 2019):
        - start with a random population of P valid architectures
        - repeat: pick S random members (tournament), mutate the best one into a child,
          evaluate the child, add it to the population and remove the OLDEST member
       The test set is never used: fitness is accuracy on a 5,000-image validation split
       taken from the CIFAR-10 training set.

Run:  python q2_nas/evolution_search.py            (resumes automatically if interrupted)
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
from common.data import load
from common.train import train_epoch, evaluate, to_cl
from common.utils import set_seed, RESULTS, rel
from q2_nas.search_space import sample_config, mutate, build_model, key, describe, space_size
from q2_nas import hardware

ap = argparse.ArgumentParser()
ap.add_argument("--population", type=int, default=10)
ap.add_argument("--tournament", type=int, default=4)
ap.add_argument("--evaluations", type=int, default=40, help="number of architectures to TRAIN")
ap.add_argument("--train-images", type=int, default=10000, help="training subset per candidate")
ap.add_argument("--epochs", type=int, default=3, help="epochs per candidate")
ap.add_argument("--max-latency-ms", type=float, default=hardware.CONSTRAINTS["max_latency_ms"])
ap.add_argument("--max-size-mb", type=float, default=hardware.CONSTRAINTS["max_size_mb"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
constraints = {"max_latency_ms": args.max_latency_ms, "max_size_mb": args.max_size_mb}

HIST_CSV = os.path.join(RESULTS, "nas_search_history.csv")
HIST_JSON = os.path.join(RESULTS, "nas_search_history.json")
BEST_JSON = os.path.join(RESULTS, "nas_best_config.json")

# ---------------------------------------------------------------- data split ---
xtr, ytr, _, _, _ = load("cifar10")
g = torch.Generator().manual_seed(123)
perm = torch.randperm(len(xtr), generator=g)
val_idx = perm[:5000]                                     # validation (fitness) split
train_pool = perm[5000:]                                  # candidates train on a subset of this

# ---------------------------------------------------------------- history (resume) ---
history = json.load(open(HIST_JSON)) if os.path.exists(HIST_JSON) else []
rng = random.Random(args.seed + len(history))             # (different stream after a resume)
print(f"Search space: {space_size():,} architectures | constraints: latency <= {constraints['max_latency_ms']} ms, "
      f"size <= {constraints['max_size_mb']} MB | population {args.population}, tournament {args.tournament}, "
      f"{args.evaluations} trained candidates, {args.epochs} epochs on {args.train_images} images each")
if history:
    print(f"Resuming: {len(history)} candidates already in the history")


def save():
    json.dump(history, open(HIST_JSON, "w"), indent=1)
    rows = []
    for h in history:
        r = {k: v for k, v in h.items() if k != "config"}
        r["config"] = json.dumps(h["config"])
        rows.append(r)
    pd.DataFrame(rows).to_csv(HIST_CSV, index=False)


def evaluate_candidate(config, generation, parent):
    """Steps 1-5 above for one architecture. Returns the history record."""
    set_seed(args.seed)
    model = build_model(config).to(memory_format=torch.channels_last)
    hw = hardware.measure(model, config["resolution"])
    ok, reason = hardware.check(hw, constraints)
    rec = {"id": len(history), "generation": generation, "parent": parent, "arch": key(config),
           **hw, "status": "accepted" if ok else "rejected", "reject_reason": reason,
           "val_acc": None, "train_time_s": None, "config": config}
    if ok:
        torch.manual_seed(args.seed)
        sub = train_pool[torch.randperm(len(train_pool))[:args.train_images]]
        steps = args.epochs * ((len(sub) + 127) // 128)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps, pct_start=0.3)
        t0 = time.time()
        for _ in range(args.epochs):
            train_epoch(model, xtr, ytr, "cifar10", opt, sched, size=config["resolution"], index=sub)
        rec["train_time_s"] = round(time.time() - t0, 1)
        rec["val_acc"] = round(evaluate(model, xtr, ytr, "cifar10", config["resolution"], index=val_idx), 2)
    history.append(rec)
    save()
    status = f"val acc {rec['val_acc']:.2f}%  ({rec['train_time_s']}s)" if ok else f"REJECTED: {reason}"
    print(f"[{rec['id']:3d}] gen {generation:2d}  {key(config):46s} lat {hw['latency_ms']:.2f} ms  "
          f"size {hw['size_MB']:.3f} MB  mem {hw['peak_mem_estimated_MB']:.2f} MB  -> {status}", flush=True)
    return rec


def trained():
    return [h for h in history if h["status"] == "accepted"]


seen = {h["arch"] for h in history}
# ---------------------------------------------------------------- 1) random initial population ---
while len(trained()) < min(args.population, args.evaluations):
    c = sample_config(rng)
    if key(c) in seen:
        continue
    seen.add(key(c))
    evaluate_candidate(c, generation=0, parent=None)

# ---------------------------------------------------------------- 2) evolution ---
while len(trained()) < args.evaluations:
    population = trained()[-args.population:]            # aging: only the newest P stay alive
    contenders = rng.sample(population, min(args.tournament, len(population)))
    parent = max(contenders, key=lambda h: h["val_acc"])
    child = mutate(parent["config"], rng)
    if key(child) in seen:
        continue
    seen.add(key(child))
    evaluate_candidate(child, generation=parent["generation"] + 1, parent=parent["id"])

# ---------------------------------------------------------------- 3) best architecture ---
best = max(trained(), key=lambda h: h["val_acc"])
json.dump({"id": best["id"], "arch": best["arch"], "val_acc": best["val_acc"],
           "latency_ms": best["latency_ms"], "size_MB": best["size_MB"], "params": best["params"],
           "MACs_M": best["MACs_M"], "constraints": constraints, "config": best["config"]},
          open(BEST_JSON, "w"), indent=1)
n_rej = sum(h["status"] == "rejected" for h in history)
print(f"\nSearched {len(history)} architectures: {len(trained())} trained, {n_rej} rejected by the hardware constraints")
print(f"Best: #{best['id']} {best['arch']}  val acc {best['val_acc']:.2f}%  "
      f"latency {best['latency_ms']:.2f} ms  size {best['size_MB']:.3f} MB")
print("      " + describe(best["config"]))
print(f"Saved {rel(BEST_JSON)}, {rel(HIST_CSV)}, {rel(HIST_JSON)}")
