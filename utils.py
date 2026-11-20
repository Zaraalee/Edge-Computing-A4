"""common/utils.py - small helpers shared by every script (seeding, evaluation, measurement)."""
import os
import random
import time
import tracemalloc
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
for _d in (CKPT, RESULTS, ONNX_DIR):
    os.makedirs(_d, exist_ok=True)


def rel(path):
    """Path relative to the project folder, for printing (e.g. checkpoints/teacher.pth)."""
    return os.path.relpath(path, ROOT)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def count_macs(model, size=32):
    """Multiply-accumulate operations for one image (Conv2d + Linear layers, via forward hooks)."""
    macs = []

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            macs.append(o.numel() // o.shape[0] * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1])
        elif isinstance(m, torch.nn.Linear):
            macs.append(m.in_features * m.out_features)
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, size, size))
    model.train(was_training)
    for h in hs:
        h.remove()
    return sum(macs)


@torch.no_grad()
def latency_ms(model, size=32, warmup=10, runs=50, threads=1):
    """Mean CPU time (ms) to classify ONE image, on `threads` threads (edge setting: 1 thread)."""
    old = torch.get_num_threads()
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(1, 3, size, size)
    for _ in range(warmup):
        model(x)
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    ms = (time.perf_counter() - t) / runs * 1000
    torch.set_num_threads(old)
    return ms


def _status_mb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) / 1024
    return None


@torch.no_grad()
def peak_memory_mb(model, size=32, batch=1):
    """Peak inference memory, two ways:
      measured : increase of the process's peak resident memory (Linux VmHWM, reset through
                 /proc/self/clear_refs) while running the model. Only supported on Linux;
                 returns None elsewhere.
      estimated: weights + the largest (input + output) activation of any single layer, in MB.
    """
    model.eval()
    x = torch.randn(batch, 3, size, size)
    measured = None
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")                                   # reset the peak (high-water mark)
        before = _status_mb("VmRSS")
        model(x)
        measured = max(0.0, _status_mb("VmHWM") - before)
    except OSError:
        pass
    peak = [0]

    def hook(m, i, o):
        t_in = sum(t.numel() for t in i if torch.is_tensor(t))
        t_out = o.numel() if torch.is_tensor(o) else 0
        peak[0] = max(peak[0], (t_in + t_out) * 4)
    hs = [m.register_forward_hook(hook) for m in model.modules() if len(list(m.children())) == 0]
    model(x)
    for h in hs:
        h.remove()
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    return measured, (peak[0] + weights) / 1e6
