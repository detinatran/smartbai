"""smartbai step 1 (STEP1_PROTOCOL.md): PatchCore on VisA with a budget of real normal images.
usage: patchcore.py --split 1cls|2cls_highshot --budgets 1.0 0.1 0.02 --seeds 0 1 2 --out results/step1_patchcore.csv"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

import common as C


class Features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()

    @torch.no_grad()
    def forward(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l2 = m.layer2(m.layer1(x))
        l3 = m.layer3(l2)
        l2 = F.avg_pool2d(l2, 3, 1, 1)
        l3 = F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([l2, l3], 1)  # B, 1536, 32, 32
        b, c, h, w = f.shape
        f = F.adaptive_avg_pool1d(f.permute(0, 2, 3, 1).reshape(-1, 1, c), 1024)
        return f.reshape(b, h, w, 1024)


def extract(net, a, device, bs=32):
    out = []
    for i in range(0, len(a), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(net(C.to_tensor(a[i:i + bs], device)).float())
    return torch.cat(out)  # N, h, w, 1024


def coreset(x, frac, gen, proj_dim=128):
    """greedy k-center coreset on a random projection (PatchCore)."""
    n = len(x)
    k = max(1, int(n * frac))
    p = torch.randn(x.shape[1], proj_dim, device=x.device, generator=gen) / proj_dim ** 0.5
    z = x @ p
    idx = [int(torch.randint(n, (1,), device=x.device, generator=gen))]
    dmin = ((z - z[idx[0]]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(dmin.argmax())
        idx.append(i)
        dmin = torch.minimum(dmin, ((z - z[i]) ** 2).sum(1))
    return x[torch.tensor(idx, device=x.device)]


def nn_dist(q, bank, chunk=4096):
    out = []
    bn = (bank ** 2).sum(1)
    for i in range(0, len(q), chunk):
        a = q[i:i + chunk]
        d = (a ** 2).sum(1, keepdim=True) - 2 * a @ bank.T + bn
        out.append(d.min(1).values.clamp_min(0).sqrt())
    return torch.cat(out)


def gauss(x, sigma=4):
    k = int(4 * sigma + 1) | 1
    t = torch.arange(k, device=x.device) - k // 2
    g = torch.exp(-t.float() ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).view(1, 1, 1, -1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.transpose(2, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="1cls")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 0.1, 0.02])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--objects", nargs="*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    net = Features().to(dev)
    d = C.split(a.split)
    done = C.done_keys(a.out, ["split", "object", "budget", "seed"])
    for obj in a.objects or sorted(d.object.unique()):
        tr = d[(d.object == obj) & (d.split == "train") & (d.y == 0)].reset_index(drop=True)
        te = d[(d.object == obj) & (d.split == "test")].reset_index(drop=True)
        xtr_all, xte, mte = C.images(tr.image.tolist()), C.images(te.image.tolist()), C.images(te["mask"].tolist(), mask=True)
        fte = extract(net, xte, dev)
        for b in a.budgets:
            for s in a.seeds:
                if (a.split, obj, str(b), str(s)) in done:
                    continue
                t0 = time.time()
                rng = np.random.default_rng(1000 * s + int(b * 1000))
                n = max(5, int(round(b * len(tr))))
                sel = np.sort(rng.choice(len(tr), min(n, len(tr)), replace=False))
                ftr = extract(net, xtr_all[sel], dev).reshape(-1, 1024)
                gen = torch.Generator(device=dev).manual_seed(s)
                bank = coreset(ftr, 0.1, gen)
                h = fte.shape[1]
                pd_ = nn_dist(fte.reshape(-1, 1024), bank).reshape(len(te), 1, h, h)
                img_score = pd_.amax((1, 2, 3)).cpu().numpy()
                amap = gauss(F.interpolate(pd_, size=C.SIZE, mode="bilinear", align_corners=False)).squeeze(1).cpu().numpy()
                row = dict(method="patchcore", split=a.split, object=obj, budget=b, seed=s, n_normal=len(sel), n_anom=0,
                           **C.img_metrics(te.y.values, img_score),
                           pix_auroc=float(roc_auc_score(mte.reshape(-1), amap.reshape(-1))) if mte.any() else np.nan,
                           bank=len(bank), sec=round(time.time() - t0, 1))
                C.append(a.out, row)
                print(row, flush=True)
                del ftr, bank
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
