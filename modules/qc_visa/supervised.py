"""smartbai step 1 (STEP1_PROTOCOL.md): supervised ResNet-50 normal/defect classifier on VisA 2cls_highshot with a
budget of k real defect images per object.
usage: supervised.py --ks 60 20 5 --seeds 0 1 2 --out results/step1_supervised.csv"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

import common as C


def augment(x, gen):
    b = len(x)
    flip_h = torch.rand(b, device=x.device, generator=gen) < 0.5
    flip_v = torch.rand(b, device=x.device, generator=gen) < 0.5
    x = torch.where(flip_h.view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where(flip_v.view(-1, 1, 1, 1), x.flip(2), x)
    gain = 1 + 0.1 * (torch.rand(b, 1, 1, 1, device=x.device, generator=gen) * 2 - 1)
    bias = 0.1 * (torch.rand(b, 1, 1, 1, device=x.device, generator=gen) * 2 - 1)
    return x * gain + bias


def train_eval(xtr, ytr, xte, yte, seed, dev, steps=600, bs=32):
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Linear(net.fc.in_features, 1)
    net = net.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    scaler = torch.amp.GradScaler()
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    rng = np.random.default_rng(seed)
    net.train()
    for _ in range(steps):
        idx = np.concatenate([rng.choice(pos, bs // 2), rng.choice(neg, bs // 2)])
        x = augment(C.to_tensor(xtr[idx], dev), gen).contiguous(memory_format=torch.channels_last)
        y = torch.tensor(ytr[idx], dtype=torch.float32, device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(net(x).squeeze(1), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sch.step()
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(xte), 64):
            out.append(net(C.to_tensor(xte[i:i + 64], dev).contiguous(memory_format=torch.channels_last)).float().squeeze(1).cpu())
    return C.img_metrics(yte, torch.cat(out).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[60, 20, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--objects", nargs="*")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    d = C.split("2cls_highshot")
    done = C.done_keys(a.out, ["object", "budget", "seed"])
    for obj in a.objects or sorted(d.object.unique()):
        tr = d[(d.object == obj) & (d.split == "train")].reset_index(drop=True)
        te = d[(d.object == obj) & (d.split == "test")].reset_index(drop=True)
        xtr_all, xte = C.images(tr.image.tolist()), C.images(te.image.tolist())
        nrm, anom = np.where(tr.y == 0)[0], np.where(tr.y == 1)[0]
        for k in a.ks:
            for s in a.seeds:
                if (obj, str(k), str(s)) in done:
                    continue
                t0 = time.time()
                rng = np.random.default_rng(1000 * s + k)
                sel = np.concatenate([nrm, np.sort(rng.choice(anom, min(k, len(anom)), replace=False))])
                m = train_eval(xtr_all[sel], tr.y.values[sel], xte, te.y.values, s, dev, steps=a.steps)
                row = dict(method="supervised_r50", split="2cls_highshot", object=obj, budget=k, seed=s, n_normal=len(nrm),
                           n_anom=min(k, len(anom)), **m, pix_auroc=np.nan, bank=np.nan, sec=round(time.time() - t0, 1))
                C.append(a.out, row)
                print(row, flush=True)


if __name__ == "__main__":
    main()
