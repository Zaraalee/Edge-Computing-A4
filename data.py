"""
common/data.py
--------------
Loads CIFAR-10 (Q2) and CIFAR-100 (Q1) into memory and gives fast batch iterators.

Where the data comes from
  CIFAR-10 : torchvision download, or data/CIFAR-10-images/ (image-folder copy)
  CIFAR-100: torchvision download, or the official "python version" from
             https://www.cs.toronto.edu/~kriz/cifar.html, either as data/cifar-100-python.tar.gz
             or extracted as data/cifar-100-python/{train,test,meta}
Both are cached as .npz files in data/ so later runs start instantly.
"""
import os
import pickle
import tarfile
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

STATS = {  # per-channel mean / std of the training set
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
CIFAR100_CLASSES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _cifar10_raw():
    try:
        import torchvision
        tr = torchvision.datasets.CIFAR10(DATA, train=True, download=True)
        te = torchvision.datasets.CIFAR10(DATA, train=False, download=True)
        return tr.data, np.array(tr.targets), te.data, np.array(te.targets)
    except Exception as e:
        print(f"[data] torchvision download failed ({type(e).__name__}); using image folder")
    from PIL import Image
    out = []
    for split in ("train", "test"):
        xs, ys = [], []
        for ci, c in enumerate(CIFAR10_CLASSES):
            folder = os.path.join(DATA, "CIFAR-10-images", split, c)
            for f in sorted(os.listdir(folder)):
                xs.append(np.array(Image.open(os.path.join(folder, f)).convert("RGB")))
                ys.append(ci)
        out += [np.stack(xs), np.array(ys)]
    return out


def _cifar100_raw():
    folder = os.path.join(DATA, "cifar-100-python")          # extracted archive (train, test, meta)
    if os.path.exists(os.path.join(folder, "train")):
        out = []
        for split in ("train", "test"):
            with open(os.path.join(folder, split), "rb") as f:
                d = pickle.load(f, encoding="latin1")
            out += [d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d["fine_labels"])]
        try:
            with open(os.path.join(folder, "meta"), "rb") as f:
                names = pickle.load(f, encoding="latin1")["fine_label_names"]
        except Exception:                       # meta only holds the 100 names (alphabetical)
            names = CIFAR100_CLASSES
        return (*out, names)
    tar_path = os.path.join(DATA, "cifar-100-python.tar.gz")
    if not os.path.exists(tar_path):
        try:
            import torchvision
            tr = torchvision.datasets.CIFAR100(DATA, train=True, download=True)
            te = torchvision.datasets.CIFAR100(DATA, train=False, download=True)
            return tr.data, np.array(tr.targets), te.data, np.array(te.targets), tr.classes
        except Exception as e:
            raise FileNotFoundError(
                "CIFAR-100 not found. Download cifar-100-python.tar.gz from "
                "https://www.cs.toronto.edu/~kriz/cifar.html into data/") from e
    out = []
    with tarfile.open(tar_path) as tf:
        for split in ("train", "test"):
            d = pickle.load(tf.extractfile(f"cifar-100-python/{split}"), encoding="latin1")
            x = d["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)   # -> NHWC uint8
            out += [x, np.array(d["fine_labels"])]
        meta = pickle.load(tf.extractfile("cifar-100-python/meta"), encoding="latin1")
    return (*out, meta["fine_label_names"])


def load(name):
    """Returns x_train, y_train, x_test, y_test (uint8 NHWC / int64 tensors) and class names."""
    cache = os.path.join(DATA, f"{name}_cache.npz")
    if not os.path.exists(cache):
        os.makedirs(DATA, exist_ok=True)
        if name == "cifar10":
            xtr, ytr, xte, yte = _cifar10_raw(); classes = CIFAR10_CLASSES
        else:
            xtr, ytr, xte, yte, classes = _cifar100_raw()
        np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte, classes=np.array(classes))
    d = np.load(cache)
    return (torch.from_numpy(d["xtr"]), torch.from_numpy(d["ytr"]).long(),
            torch.from_numpy(d["xte"]), torch.from_numpy(d["yte"]).long(),
            [str(c) for c in d["classes"]])


def preprocess(x_uint8, name, size=32):
    """uint8 NHWC -> normalised float NCHW (optionally resized for Q2's input-resolution search)."""
    mean, std = STATS[name]
    x = x_uint8.permute(0, 3, 1, 2).float() / 255.0
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    if size != 32:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x


def augment(x):
    """Random crop (4-pixel reflect padding, a different crop per image) + random horizontal flip."""
    n, _, h, w = x.shape
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,)); j = torch.randint(0, 9, (n,))
    rows = (i[:, None] + torch.arange(h))[:, :, None]          # n x h x 1
    cols = (j[:, None] + torch.arange(w))[:, None, :]          # n x 1 x w
    x = xp.permute(0, 2, 3, 1)[torch.arange(n)[:, None, None], rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(n) < 0.5
    x[flip] = x[flip].flip(3)
    return x.contiguous()


def batches(x, y, name, batch_size=128, shuffle=False, train_aug=False, size=32, index=None):
    """Yields (inputs, labels). `index` restricts the iteration to a subset of image indices."""
    idx = index if index is not None else torch.arange(x.shape[0])
    if shuffle:
        idx = idx[torch.randperm(len(idx))]
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        xb = preprocess(x[b], name)
        if train_aug:
            xb = augment(xb)
        if size != 32:
            xb = F.interpolate(xb, size=(size, size), mode="bilinear", align_corners=False)
        yield xb, y[b]
