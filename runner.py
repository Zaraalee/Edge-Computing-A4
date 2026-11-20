"""
q1_distillation/runner.py
-------------------------
Small helpers shared by the Q1 training scripts:
  * the optimiser recipe (SGD + Nesterov momentum + weight decay, one-cycle learning rate)
  * resumable training: the full state (models, optimisers, schedulers, RNG, epoch) is saved
    after every epoch, so a long CPU run that gets interrupted continues where it stopped
  * the per-epoch history CSV
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from common.utils import CKPT, RESULTS

RESUME_DIR = os.path.join(CKPT, "_resume")


def make_optimizer(model_params, epochs, steps_per_epoch, lr=0.1, wd=5e-4):
    opt = torch.optim.SGD(model_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                                pct_start=0.25, div_factor=10, final_div_factor=100)
    return opt, sched


def save_resume(run, epoch, modules, optimizers, schedulers):
    os.makedirs(RESUME_DIR, exist_ok=True)
    torch.save({"epoch": epoch,
                "modules": {k: m.state_dict() for k, m in modules.items()},
                "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                "schedulers": {k: s.state_dict() for k, s in schedulers.items()},
                "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state())},
               os.path.join(RESUME_DIR, f"{run}.pt"))


def load_resume(run, modules, optimizers, schedulers):
    """Returns the number of finished epochs (0 if there is nothing to resume)."""
    path = os.path.join(RESUME_DIR, f"{run}.pt")
    if not os.path.exists(path):
        return 0
    st = torch.load(path, map_location="cpu", weights_only=False)
    for k, m in modules.items():
        m.load_state_dict(st["modules"][k])
    for k, o in optimizers.items():
        o.load_state_dict(st["optimizers"][k])
    for k, s in schedulers.items():
        s.load_state_dict(st["schedulers"][k])
    random.setstate(st["rng"][0]); np.random.set_state(st["rng"][1]); torch.set_rng_state(st["rng"][2])
    print(f"Resuming {run} after epoch {st['epoch']}", flush=True)
    return st["epoch"]


def log_history(run, rows):
    """Append per-epoch rows to results/q1_training_history.csv (one file for all Q1 runs)."""
    path = os.path.join(RESULTS, "q1_training_history.csv")
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new_keys = set(zip(df.run, df.epoch))                # replace rows of a re-run epoch
        old = old[[(r, e) not in new_keys for r, e in zip(old.run, old.epoch)]]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
