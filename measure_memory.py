"""
common/measure_memory.py
------------------------
Measures the real peak inference memory of ONE model in a FRESH Python process (so memory
freed by other models cannot be reused and hide the cost). Linux only (/proc/self/status).

  peak inference memory = VmHWM (peak resident memory) while running the FIRST image
                          - VmRSS (resident memory) just before, with the model already loaded
This is what a freshly started edge application needs on top of the loaded model. It includes
PyTorch's one-time initialisation of its CPU kernels (about 10 MB here), which is why even tiny
models show ~10 MB; benchmark.py also reports an estimate from the layer sizes (weights + the
largest single-layer activation) that isolates the model's own share.
Usage: python common/measure_memory.py <model_name>      -> prints the value in MB
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from common.registry import load_model


def status_mb(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) / 1024


if __name__ == "__main__":
    torch.set_num_threads(1)
    model, info = load_model(sys.argv[1])
    x = torch.randn(1, 3, info["resolution"], info["resolution"])
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")                                   # reset the peak to the current usage
    before = status_mb("VmRSS")
    with torch.no_grad():
        model(x)
    print(f"{status_mb('VmHWM') - before:.3f}")
