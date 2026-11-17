"""部署层：物理趋势 + 实测残差 的融合模型，以及留一法（LOO）选参。

离线五方案是"研究口径"；交付软件需要一个既能吃户型/AP/家具、
又能在有实测点时把精度拉满的部署模型。这里定义候选公式，
**只用 29 个训练点的留一法误差选参**，最后在 12 个测试点上做一次性评估。

候选：
  RBF(L)   物理趋势 + RBF 残差（L = 1.5/2.5/3.5/5.0 m）
  IDWr(p)  物理趋势 + 反距离加权残差（p = 1/2/3）
  NET      方案五网络（TTA）
  MIX(a)   a·NET + (1−a)·IDW_RAW，a 由 LOO 选
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import config as C
from . import models as M
from .data import level_of
from .evaluate import metrics, baseline_idw
from .realdata import load_real, real_inputs, sample_bilinear, predict_field
from .physics import path_loss_field

PL0, NEXP = C.PL0, C.NEXP


def _trend(pack, xy):
    f = path_loss_field(pack["ap"], [], PL0, NEXP, C.GRID_N, C.CELL)
    return sample_bilinear(f, xy)


def residual_rbf(pack, xy_eval, length=3.0, lam=0.5):
    base_obs = _trend(pack, pack["obs_xy"])
    r = pack["obs_val"] - base_obs
    d2 = ((pack["obs_xy"][:, None] - pack["obs_xy"][None]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * length ** 2)) + lam * np.eye(len(r))
    a = np.linalg.solve(K, r)
    d2p = ((xy_eval[:, None] - pack["obs_xy"][None]) ** 2).sum(-1)
    return _trend(pack, xy_eval) + np.exp(-d2p / (2 * length ** 2)) @ a


def residual_idw(pack, xy_eval, power=2.0):
    base_obs = _trend(pack, pack["obs_xy"])
    r = pack["obs_val"] - base_obs
    d = np.sqrt(((xy_eval[:, None] - pack["obs_xy"][None]) ** 2).sum(-1))
    w = 1.0 / np.maximum(d, 0.05) ** power
    return _trend(pack, xy_eval) + (w * r).sum(1) / w.sum(1)


def loo(pack, xy, rssi, fn, **kw):
    """留一法：逐个拿掉一个测点，用其余点预测它。"""
    preds = np.zeros(len(rssi))
    for i in range(len(rssi)):
        keep = np.array([j for j in range(len(rssi)) if j != i])
        sub = {**pack}
        sub["obs_xy"] = pack["obs_xy"][keep]
        sub["obs_val"] = pack["obs_val"][keep]
        preds[i] = fn(sub, xy[i:i + 1], **kw)[0]
    return preds


def main():
    real = load_real()
    tr, te = real["train_idx"], real["test_idx"]
    thr = [float(t) for t in real["thresholds"]]
    xy_all, rssi_all = real["xy"], real["rssi"]
    xy_tr, y_tr = xy_all[tr], rssi_all[tr]
    xy_te, y_te = xy_all[te], rssi_all[te]
    lv_te = level_of(y_te, thr)

    pack = real_inputs(tr, -56.98366, 9.98995)
    base_idw = baseline_idw(pack, xy_te)

    # ---- 候选残差插值
    rows = []
    for L in (1.5, 2.5, 3.5, 5.0):
        p = residual_rbf(pack, xy_te, length=L)
        loo_p = loo(pack, xy_tr, y_tr, residual_rbf, length=L)
        rows.append({"name": f"RBF L={L}", "pred": p,
                     "loo_rmse": float(np.sqrt(np.mean((loo_p - y_tr) ** 2))),
                     "loo_level": float((level_of(loo_p, thr) == level_of(y_tr, thr)).mean())})
    for pw in (1.0, 2.0, 3.0):
        p = residual_idw(pack, xy_te, power=pw)
        loo_p = loo(pack, xy_tr, y_tr, residual_idw, power=pw)
        rows.append({"name": f"IDWr p={pw}", "pred": p,
                     "loo_rmse": float(np.sqrt(np.mean((loo_p - y_tr) ** 2))),
                     "loo_level": float((level_of(loo_p, thr) == level_of(y_tr, thr)).mean())})

    # ---- 方案五（TTA）+ 与 IDW 的混合
    model = M.build("pgmpr")
    model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, "pgmpr", "weights.pt"),
                                     map_location="cpu"))
    model.eval()
    net_te = sample_bilinear(predict_field(model, pack["x"], -56.98366, 9.98995, "pgmpr", tta=True), xy_te)

    net_loo = np.zeros(len(y_tr))
    for i in range(len(y_tr)):
        keep = np.array([j for j in range(len(y_tr)) if j != i])
        pk = real_inputs(tr[keep], -56.98366, 9.98995)
        f = predict_field(model, pk["x"], -56.98366, 9.98995, "pgmpr", tta=False)
        net_loo[i] = sample_bilinear(f, xy_tr[i:i + 1])[0]
    idw_loo = np.array([baseline_idw({**pack, "obs_xy": pack["obs_xy"][[j for j in range(len(y_tr)) if j != i]],
                                     "obs_val": pack["obs_val"][[j for j in range(len(y_tr)) if j != i]]},
                                    xy_tr[i:i + 1])[0] for i in range(len(y_tr))])

    best_a, best_loo = 0.0, 1e9
    for a in np.linspace(0, 1, 21):
        loo_p = a * net_loo + (1 - a) * idw_loo
        e = float(np.sqrt(np.mean((loo_p - y_tr) ** 2)))
        if e < best_loo:
            best_loo, best_a = e, float(a)
    mix_te = best_a * net_te + (1 - best_a) * base_idw

    rows.append({"name": "NET(pgmpr+TTA)", "pred": net_te,
                 "loo_rmse": float(np.sqrt(np.mean((net_loo - y_tr) ** 2))),
                 "loo_level": float((level_of(net_loo, thr) == level_of(y_tr, thr)).mean())})
    rows.append({"name": f"MIX a={best_a:.2f}", "pred": mix_te,
                 "loo_rmse": best_loo,
                 "loo_level": float((level_of(best_a * net_loo + (1 - best_a) * idw_loo, thr) == level_of(y_tr, thr)).mean()), "a": best_a})
    rows.append({"name": "IDW_RAW", "pred": base_idw,
                 "loo_rmse": float(np.sqrt(np.mean((idw_loo - y_tr) ** 2))),
                 "loo_level": float((level_of(idw_loo, thr) == level_of(y_tr, thr)).mean())})

    for r in rows:
        m = metrics(y_te, r["pred"], lv_te, level_of(r["pred"], thr))
        r.update({k: m[k] for k in ("rmse", "mae", "level_acc", "level_acc_pm1", "binary_acc")})

    rows_sorted = sorted(rows, key=lambda r: r["loo_rmse"])
    best = rows_sorted[0]
    out = {"candidates": [{k: v for k, v in r.items() if k != "pred"} for r in rows],
           "best_by_loo": best["name"],
           "best": {k: v for k, v in best.items() if k != "pred"},
           "mix_alpha": best_a,
           "params": {"pl0": PL0, "n": NEXP, "mean": -56.98366, "std": 9.98995}}

    with open(os.path.join(C.OUT_DIR, "fusion.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"{'候选':18s} {'LOO RMSE':>9s} {'测试 RMSE':>9s} {'分级':>7s} {'±1':>7s} {'二分':>7s}")
    for r in rows_sorted:
        print(f"{r['name']:18s} {r['loo_rmse']:9.3f} {r['rmse']:9.3f} "
              f"{r['level_acc']*100:6.1f}% {r['level_acc_pm1']*100:6.1f}% {r['binary_acc']*100:6.1f}%")
    print("按 LOO 选中:", best["name"])
    return out


if __name__ == "__main__":
    main()
