"""训练五个方案：仿真预训练 → 实测自监督微调。"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import config as C
from . import models as M
from .realdata import real_inputs
from . import data as D

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))


def load_sim(name: str, scatter: bool = False) -> dict:
    f = os.path.join(C.DATA_DIR, f"sim{'_scatter' if scatter else ''}_{name}.npz")
    z = np.load(f)
    return {k: z[k] for k in z.files}


def masked_mse(pred, target, room, density=None, weight_obs: float = 0.0):
    pred = pred.squeeze(1) if pred.dim() == 4 else pred
    w = room
    if density is not None and weight_obs > 0:
        w = room * (1.0 + weight_obs * density)
        w = w / (w.sum() / max(float(room.sum()), 1.0))
    diff = (pred - target) ** 2
    return (diff * w).sum() / w.sum().clamp_min(1e-6)


def train_one(name: str, tr: dict, va: dict, mean: float, std: float,
              epochs: int = C.EPOCHS, verbose: bool = True) -> tuple[nn.Module, dict]:
    torch.manual_seed(C.SEED_TRAIN)
    np.random.seed(C.SEED_TRAIN)

    model = M.build(name)
    opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    xtr = torch.from_numpy(tr["x"]).float()
    ytr = torch.from_numpy(tr["y"]).float()
    rtr = torch.from_numpy(tr["room"]).float()
    dtr = torch.from_numpy(tr["x"][:, 5]).float()
    xva = torch.from_numpy(va["x"]).float()
    yva = torch.from_numpy(va["y"]).float()
    rva = torch.from_numpy(va["room"]).float()
    dva = torch.from_numpy(va["x"][:, 5]).float()

    n = xtr.shape[0]
    best = {"val": 1e9, "epoch": -1, "state": None, "train": 1e9}
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot, cnt = 0.0, 0
        for i in range(0, n, C.BATCH):
            idx = perm[i:i + C.BATCH]
            pred = model(M.inputs_for(name, xtr[idx]))
            loss = masked_mse(pred, ytr[idx], rtr[idx],
                              dtr[idx] if name == "pgmpr" else None,
                              2.0 if name == "pgmpr" else 0.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
            cnt += len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(M.inputs_for(name, xva))
            vloss = float(masked_mse(pv, yva, rva, dva if name == "pgmpr" else None,
                                     2.0 if name == "pgmpr" else 0.0))
        tr_loss = tot / max(cnt, 1)
        hist.append({"epoch": ep, "train": tr_loss, "val": vloss,
                     "lr": float(sched.get_last_lr()[0])})
        if vloss < best["val"] - 1e-6:
            best = {"val": vloss, "train": tr_loss, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [{name}] ep {ep:3d} train {tr_loss:.4f} val {vloss:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} ({time.time()-t0:.0f}s)")
        if ep - best["epoch"] >= C.PATIENCE:
            break

    model.load_state_dict(best["state"])
    model.eval()
    meta = {"name": name, "best_epoch": best["epoch"] + 1, "best_val": best["val"],
            "best_train": best["train"], "epochs_run": len(hist),
            "params": int(sum(p.numel() for p in model.parameters())),
            "seconds": round(time.time() - t0, 1), "mean": mean, "std": std,
            "history": hist}
    return model, meta


def finetune(model, name: str, mean: float, std: float, steps: int = 240,
             monitor_seed: int = 0) -> dict:
    """实测自监督微调：随机把测点分成"输入"与"监督"两组，两组不相交。"""
    real = D.load_measured()
    tr_idx = np.load(os.path.join(C.DATA_DIR, "measured.npz"))["train_idx"]
    n = len(tr_idx)
    rng = np.random.default_rng(monitor_seed)
    mon = rng.choice(n, size=min(6, n // 4), replace=False)
    mon_set = set(mon.tolist())

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    best = {"err": 1e9, "state": None, "step": -1}
    ys = real["rssi"][tr_idx]
    hist = []

    for step in range(steps):
        k_in = int(rng.integers(max(6, n // 3), max(7, int(n * 0.75)) + 1))
        perm = rng.permutation(n)
        # 监控点永远不进输入
        cand = [i for i in perm if i not in mon_set]
        inp = np.array(sorted(cand[:k_in]))
        sup = np.array(sorted(set(range(n)) - set(inp.tolist())))
        if len(sup) == 0:
            continue

        pack = real_inputs(tr_idx[inp], mean, std)
        x = torch.from_numpy(pack["x"][None]).float()
        pred = model(M.inputs_for(name, x))

        supxy = real["xy"][tr_idx[sup]]
        level_vals = (ys[sup] - mean) / std
        fx = np.clip(supxy[:, 0] / C.CELL - 0.5, 0, C.GRID_N - 1.001)
        fy = np.clip(supxy[:, 1] / C.CELL - 0.5, 0, C.GRID_N - 1.001)
        ix = np.floor(fx).astype(int)
        iy = np.floor(fy).astype(int)
        tx = torch.from_numpy((fx - ix).astype(np.float32))
        ty = torch.from_numpy((fy - iy).astype(np.float32))
        ixt = torch.from_numpy(ix.astype(np.int64))
        iyt = torch.from_numpy(iy.astype(np.int64))

        p00 = pred[0, 0, iyt, ixt]
        p10 = pred[0, 0, iyt, torch.clamp(ixt + 1, max=C.GRID_N - 1)]
        p01 = pred[0, 0, torch.clamp(iyt + 1, max=C.GRID_N - 1), ixt]
        p11 = pred[0, 0, torch.clamp(iyt + 1, max=C.GRID_N - 1), torch.clamp(ixt + 1, max=C.GRID_N - 1)]
        pv = (p00 * (1 - tx) * (1 - ty) + p10 * tx * (1 - ty)
              + p01 * (1 - tx) * ty + p11 * tx * ty)
        loss = ((pv - torch.from_numpy(level_vals.astype(np.float32))) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                perr = []
                for _ in range(3):
                    inp2 = np.array(sorted(rng.choice([i for i in range(n) if i not in mon_set],
                                                      size=min(12, n - len(mon_set)), replace=False)))
                    pk = real_inputs(tr_idx[inp2], mean, std)
                    t2 = torch.from_numpy(pk["x"][None]).float()
                    out = model(M.inputs_for(name, t2)).numpy()[0, 0] * std + mean
                    from .realdata import sample_bilinear
                    p = sample_bilinear(out, real["xy"][tr_idx[mon]])
                    perr.append(np.sqrt(np.mean((p - ys[mon]) ** 2)))
                merr = float(np.mean(perr))
            if merr < best["err"]:
                best = {"err": merr, "step": step,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            hist.append({"step": step, "sup": int(len(sup)), "loss": float(loss), "mon_rmse": merr})
            model.train()

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    return {"monitor_rmse": best["err"], "best_step": best["step"], "history": hist}


def save(model, meta: dict, name: str) -> None:
    d = os.path.join(C.MODEL_DIR, name)
    os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(d, "weights.pt"))
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main(only: list[str] | None = None, finetune_steps: int = 240) -> None:
    with open(os.path.join(C.DATA_DIR, "sim_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    mean, std = meta["mean"], meta["std"]

    tr = load_sim("train")
    va = load_sim("val")
    tr_s = load_sim("train", scatter=True)
    va_s = load_sim("val", scatter=True)

    names = only or ["ae", "unet", "resunet", "unet_scatter", "pgmpr"]
    for name in names:
        print(f"=== 训练 {name}")
        t0 = time.time()
        if name == "unet_scatter":
            model, m = train_one(name, tr_s, va_s, mean, std, epochs=40)
        elif name == "pgmpr":
            # 方案五用两份训练集（常规 + 散落）
            merged = {"x": np.concatenate([tr["x"], tr_s["x"]]),
                      "y": np.concatenate([tr["y"], tr_s["y"]]),
                      "room": np.concatenate([tr["room"], tr_s["room"]])}
            model, m = train_one(name, merged, va, mean, std, epochs=40)
        else:
            model, m = train_one(name, tr, va, mean, std, epochs=40)

        ft = finetune(model, name, mean, std, steps=finetune_steps)
        m["finetune"] = {k: v for k, v in ft.items() if k != "history"}
        m["finetune_history"] = ft["history"]
        m["total_seconds"] = round(time.time() - t0, 1)
        save(model, m, name)
        print(f"=== {name} 完成：best_val={m['best_val']:.4f} params={m['params']} "
              f"微调监控 RMSE={ft['monitor_rmse']:.3f} dB，耗时 {m['total_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if a.bench:
        with open(os.path.join(C.DATA_DIR, "sim_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        tr = load_sim("train")
        va = load_sim("val")
        t0 = time.time()
        _, m = train_one("unet", tr, va, meta["mean"], meta["std"], epochs=2)
        print("2 epochs:", round(time.time() - t0, 1), "s", m["params"])
    else:
        main([s for s in a.only.split(",") if s] or None, a.steps)
