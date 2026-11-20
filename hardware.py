"""
q2_nas/hardware.py
------------------
Hardware-aware measurements used by the search, all done on the real CPU:
  * latency   : mean time to classify ONE image on ONE CPU thread (edge setting)
  * storage   : real size of the saved model file (state_dict written with torch.save)
  * memory    : peak inference memory - measured (Linux) and estimated from activation sizes
and the deployment constraints a candidate must satisfy.
"""
import io
import statistics
import torch
from common.utils import latency_ms, peak_memory_mb, count_params, count_macs

# Default deployment budget of the target edge device (can be changed on the command line).
CONSTRAINTS = {"max_latency_ms": 1.5, "max_size_mb": 1.0}


def storage_mb(model):
    """Bytes of the model file as it would be saved to disk (torch.save of the state_dict)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1e6


def measure(model, resolution, repeats=3):
    """All hardware metrics of an (untrained or trained) model. Latency does not depend on the
    weights, so candidates can be checked BEFORE spending time on training them."""
    lat = statistics.median(latency_ms(model, resolution, warmup=5, runs=30) for _ in range(repeats))
    mem_measured, mem_est = peak_memory_mb(model, resolution)
    return {"params": count_params(model), "MACs_M": round(count_macs(model, resolution) / 1e6, 2),
            "size_MB": round(storage_mb(model), 4), "latency_ms": round(lat, 3),
            "peak_mem_measured_MB": None if mem_measured is None else round(mem_measured, 3),
            "peak_mem_estimated_MB": round(mem_est, 3)}


def check(metrics, constraints=CONSTRAINTS):
    """Returns (ok, reason). An architecture is rejected if it is too slow or too large."""
    reasons = []
    if metrics["latency_ms"] > constraints["max_latency_ms"]:
        reasons.append(f"latency {metrics['latency_ms']:.2f} ms > {constraints['max_latency_ms']} ms")
    if metrics["size_MB"] > constraints["max_size_mb"]:
        reasons.append(f"size {metrics['size_MB']:.2f} MB > {constraints['max_size_mb']} MB")
    return (not reasons), "; ".join(reasons)
