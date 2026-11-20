"""common/train.py - the plain supervised training loop and evaluation used by Q1 baselines,
the teacher, and Q2 (NAS candidates and the final architecture)."""
import time
import torch
import torch.nn.functional as F
from common.data import batches


def to_cl(x):
    """channels_last memory layout: 1.2-1.6x faster convolutions on this CPU."""
    return x.contiguous(memory_format=torch.channels_last)


def train_epoch(model, x, y, name, optimizer, scheduler=None, batch_size=128, size=32,
                index=None, label_smoothing=0.0):
    """One pass over the (sub)set. Returns (mean loss, train accuracy %, seconds)."""
    model.train()
    t0, tot, correct, loss_sum = time.time(), 0, 0, 0.0
    for xb, yb in batches(x, y, name, batch_size, shuffle=True, train_aug=True, size=size, index=index):
        out = model(to_cl(xb))
        loss = F.cross_entropy(out, yb, label_smoothing=label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item() * len(yb); correct += (out.argmax(1) == yb).sum().item(); tot += len(yb)
    return loss_sum / tot, 100.0 * correct / tot, time.time() - t0


@torch.no_grad()
def predict(model, x, name, size=32, index=None, batch_size=500, return_logits=False):
    model.eval()
    outs = [model(to_cl(xb)) for xb, _ in batches(x, torch.zeros(len(x), dtype=torch.long), name,
                                                    batch_size, size=size, index=index)]
    logits = torch.cat(outs)
    return logits if return_logits else logits.argmax(1)


def evaluate(model, x, y, name, size=32, index=None):
    """Top-1 accuracy (%) on the given images."""
    p = predict(model, x, name, size, index)
    target = y if index is None else y[index]
    return 100.0 * (p == target).float().mean().item()


def evaluate_top5(model, x, y, name, size=32):
    logits = predict(model, x, name, size, return_logits=True)
    top5 = logits.topk(5, 1).indices
    return (100.0 * (top5[:, 0] == y).float().mean().item(),
            100.0 * (top5 == y[:, None]).any(1).float().mean().item())
