"""
q2_nas/search_space.py
----------------------
The searchable CNN architecture space and the automated architecture generator.

An architecture is a small dictionary (a "config"), for example
    {"resolution": 32, "activation": "relu6",
     "stages": [{"depth": 2, "channels": 32, "kernel": 3, "block": "dsconv"},
                {"depth": 1, "channels": 64, "kernel": 5, "block": "conv"},
                {"depth": 3, "channels": 128, "kernel": 3, "block": "dsconv"}]}

What can vary (the six required dimensions)
  * convolutional depth       : 1-3 blocks in each of the 3 stages            (3 .. 9 blocks)
  * number of filters         : a channel choice per stage
  * kernel size               : 3, 5 or 7 per stage
  * depthwise separable blocks: each stage is either a standard conv block or a
                                depthwise-separable block (depthwise kxk + pointwise 1x1)
  * activation function       : ReLU, ReLU6, Hardswish or SiLU (whole network)
  * input resolution          : 16, 24, 32 or 40 pixels

Size of the space: 4 resolutions x 4 activations x (3 depths x 3 widths x 3 kernels x 2 blocks)^3
                 = 16 x 54^3 = 2,519,424 different architectures.

build_model(config) is the "generator": it turns any config into a PyTorch network.
"""
import copy
import json
import random
import torch
import torch.nn as nn

RESOLUTIONS = [16, 24, 32, 40]
ACTIVATIONS = {"relu": nn.ReLU, "relu6": nn.ReLU6, "hardswish": nn.Hardswish, "silu": nn.SiLU}
DEPTHS = [1, 2, 3]
CHANNELS = [[16, 24, 32], [32, 48, 64], [64, 96, 128]]      # choices for stage 1, 2, 3
KERNELS = [3, 5, 7]
BLOCKS = ["conv", "dsconv"]
STEM_CHANNELS = 16


def space_size():
    per_stage = len(DEPTHS) * len(CHANNELS[0]) * len(KERNELS) * len(BLOCKS)
    return len(RESOLUTIONS) * len(ACTIVATIONS) * per_stage ** 3


# ---------------------------------------------------------------- sampling ---
def sample_config(rng=random):
    """A random architecture from the space."""
    return {"resolution": rng.choice(RESOLUTIONS),
            "activation": rng.choice(list(ACTIVATIONS)),
            "stages": [{"depth": rng.choice(DEPTHS), "channels": rng.choice(CHANNELS[i]),
                        "kernel": rng.choice(KERNELS), "block": rng.choice(BLOCKS)} for i in range(3)]}


def mutate(config, rng=random):
    """Copy of `config` with ONE randomly chosen decision changed (used by the evolutionary search)."""
    c = copy.deepcopy(config)
    what = rng.choice(["resolution", "activation", "depth", "channels", "kernel", "block"])
    if what == "resolution":
        c["resolution"] = rng.choice([r for r in RESOLUTIONS if r != c["resolution"]])
    elif what == "activation":
        c["activation"] = rng.choice([a for a in ACTIVATIONS if a != c["activation"]])
    else:
        i = rng.randrange(3)
        options = {"depth": DEPTHS, "channels": CHANNELS[i], "kernel": KERNELS, "block": BLOCKS}[what]
        c["stages"][i][what] = rng.choice([o for o in options if o != c["stages"][i][what]])
    return c


def key(config):
    """A unique, readable string for a config (used to skip duplicates)."""
    s = "-".join(f"{st['block'][0]}{st['depth']}x{st['channels']}k{st['kernel']}" for st in config["stages"])
    return f"r{config['resolution']}-{config['activation']}-{s}"


# ---------------------------------------------------------------- generator ---
class ConvBlock(nn.Module):
    """Standard block: kxk convolution -> BatchNorm -> activation."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, k, stride, k // 2, bias=False), nn.BatchNorm2d(cout), act())

    def forward(self, x):
        return self.body(x)


class DSConvBlock(nn.Module):
    """Depthwise separable block: kxk depthwise conv (one filter per channel) -> BN -> act ->
    1x1 pointwise conv (mixes channels) -> BN -> act. Uses about k*k times fewer MACs than a
    standard conv with the same input and output channels."""

    def __init__(self, cin, cout, k, stride, act):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cin, k, stride, k // 2, groups=cin, bias=False), nn.BatchNorm2d(cin), act(),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), act())

    def forward(self, x):
        return self.body(x)


class GeneratedNet(nn.Module):
    """Network built from a config: stem -> 3 stages (the first block of stages 2 and 3 halves
    the resolution) -> global average pooling -> linear classifier."""

    def __init__(self, config, num_classes=10):
        super().__init__()
        self.config = config
        act = ACTIVATIONS[config["activation"]]
        layers = [ConvBlock(3, STEM_CHANNELS, 3, 1, act)]
        cin = STEM_CHANNELS
        for i, st in enumerate(config["stages"]):
            Block = DSConvBlock if st["block"] == "dsconv" else ConvBlock
            for d in range(st["depth"]):
                stride = 2 if (i > 0 and d == 0) else 1
                layers.append(Block(cin, st["channels"], st["kernel"], stride, act))
                cin = st["channels"]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def build_model(config, num_classes=10):
    """The automated architecture generator: config dict -> PyTorch model."""
    return GeneratedNet(config, num_classes)


def describe(config):
    lines = [f"input {config['resolution']}x{config['resolution']}, activation {config['activation']}"]
    for i, st in enumerate(config["stages"]):
        lines.append(f"stage {i + 1}: {st['depth']} x {st['block']} {st['kernel']}x{st['kernel']}, {st['channels']} ch")
    return "; ".join(lines)


if __name__ == "__main__":                    # quick demo: sample, build and run 3 random architectures
    random.seed(0)
    print(f"Search space size: {space_size():,} architectures")
    for _ in range(3):
        c = sample_config()
        m = build_model(c)
        out = m(torch.randn(2, 3, c["resolution"], c["resolution"]))
        print(key(c), "->", tuple(out.shape), f"{sum(p.numel() for p in m.parameters()):,} params")
    print(json.dumps(c, indent=1))
