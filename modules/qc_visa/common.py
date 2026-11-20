"""smartbai shared utilities: VisA split loading, cached 256x256 image arrays, metrics."""
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(os.environ.get("SMARTBAI_ROOT", Path(__file__).resolve().parents[2]))  # holds data/VisA, spot-diff/, cache/
DATA = ROOT / "data" / "VisA"
SPLITS = ROOT / "spot-diff" / "split_csv"
CACHE = ROOT / "cache"
SIZE = 256
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def split(name):
    d = pd.read_csv(SPLITS / f"{name}.csv")
    d["y"] = (d.label == "anomaly").astype(int)
    return d


def _load(p, mask=False):
    if mask:
        if not isinstance(p, str) or not p:
            return np.zeros((SIZE, SIZE), np.uint8)
        m = Image.open(DATA / p).convert("L").resize((SIZE, SIZE), Image.NEAREST)
        return (np.asarray(m) > 0).astype(np.uint8)
    return np.asarray(Image.open(DATA / p).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))


def images(paths, mask=False):
    """uint8 array (N,256,256,3) or (N,256,256) for masks, cached on disk by the list of paths."""
    paths = [p if isinstance(p, str) else "" for p in paths]
    key = hashlib.md5(("m" if mask else "i").join(paths).encode()).hexdigest()[:16]
    f = CACHE / f"{'mask' if mask else 'img'}_{SIZE}_{key}.npy"
    if f.exists():
        return np.load(f)
    with ThreadPoolExecutor(8) as ex:
        a = np.stack(list(ex.map(lambda p: _load(p, mask), paths)))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, a)
    return a


def to_tensor(a, device):
    x = torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255)
    return ((x - MEAN) / STD).to(device)


def img_metrics(y, s):
    return dict(img_auroc=float(roc_auc_score(y, s)), img_ap=float(average_precision_score(y, s)))


def done_keys(csv, cols):
    if not Path(csv).exists():
        return set()
    d = pd.read_csv(csv)
    return set(map(tuple, d[cols].astype(str).values))


def append(csv, row):
    pd.DataFrame([row]).to_csv(csv, mode="a", header=not Path(csv).exists(), index=False)
