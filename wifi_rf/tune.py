"""部署公式的离线调参（只用 29 个训练点的留一法，测试集不参与选择）。"""
from __future__ import annotations

import itertools
import json
import os

import numpy as np

from . import config as C
from .data import level_of
from .realdata import load_real, real_inputs
from .physics import path_loss_field
from .realdata import sample_bilinear

PL0, NEXP = C.PL0, C.NEXP


def solve_predict(xy_obs, r_obs, xy_q, length, lam, drift):
    d2 = ((xy_obs[:, None] - xy_obs[None]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * length ** 2)) + lam * np.eye(len(xy_obs))
    cols = [np.ones(len(xy_obs))]
    if drift:
        cols += [xy_obs[:, 0], xy_obs[:, 1]]
    G = np.stack(cols, 1)
    # 同时求 RBF 权重与漂移系数（带小正则的最小二乘，联合求解）
    n, m = len(xy_obs), G.shape[1]
    A = np.zeros((n + m, n + m))
    A[:n, :n] = K
    A[:n, n:] = G
    A[n:, :n] = G.T
    A[n:, n:] = np.eye(m) * 1e-6
    b = np.concatenate([r_obs, np.zeros(m)])
    sol = np.linalg.solve(A, b)
    alpha, beta = sol[:n], sol[n:]
    d2q = ((xy_q[:, None] - xy_obs[None]) ** 2).sum(-1)
    kq = np.exp(-d2q / (2 * length ** 2))
    out = kq @ alpha + beta[0]
    if drift:
        out = out + beta[1] * xy_q[:, 0] + beta[2] * xy_q[:, 1]
    return out


def main():
    real = load_real()
    tr, te = real["train_idx"], real["test_idx"]
    thr = [float(t) for t in real["thresholds"]]
    xy = real["xy"]
    y = real["rssi"]
    xy_tr, y_tr = xy[tr], y[tr]
    xy_te, y_te = xy[te], y[te]
    lv_te = level_of(y_te, thr)

    pack = real_inputs(tr, -56.98366, 9.98995)
    ap = pack["ap"]

    def trend(q):
        f = path_loss_field(ap, [], PL0, NEXP, C.GRID_N, C.CELL)
        return sample_bilinear(f, q)

    r_tr = y_tr - trend(xy_tr)

    best = None
    grid = list(itertools.product((1.5, 2.0, 2.5, 3.0, 4.0, 6.0), (0.02, 0.1, 0.3, 1.0), (False, True)))
    rows = []
    for length, lam, drift in grid:
        loo = np.zeros(len(y_tr))
        for i in range(len(y_tr)):
            keep = np.array([j for j in range(len(y_tr)) if j != i])
            loo[i] = solve_predict(xy_tr[keep], r_tr[keep], xy_tr[i:i + 1], length, lam, drift)[0] + trend(xy_tr[i:i + 1])[0]
        e = float(np.sqrt(np.mean((loo - y_tr) ** 2)))
        rows.append({"L": length, "lam": lam, "drift": drift, "loo": e})
        if best is None or e < best["loo"]:
            best = {"L": length, "lam": lam, "drift": drift, "loo": e}

    test = solve_predict(xy_tr, r_tr, xy_te, best["L"], best["lam"], best["drift"]) + trend(xy_te)
    m = {
        "rmse": float(np.sqrt(np.mean((test - y_te) ** 2))),
        "mae": float(np.mean(np.abs(test - y_te))),
        "level_acc": float((level_of(test, thr) == lv_te).mean()),
        "level_acc_pm1": float((np.abs(level_of(test, thr) - lv_te) <= 1).mean()),
    }
    rows.sort(key=lambda r: r["loo"])
    print("LOO 前 6:")
    for r in rows[:6]:
        print(f"  L={r['L']:.1f} lam={r['lam']:.2f} drift={r['drift']} LOO={r['loo']:.3f}")
    print("选中:", best, "测试:", {k: round(v, 3) for k, v in m.items()})
    out = {"grid": rows, "best": best, "test": m, "pl0": PL0, "n": NEXP}
    with open(os.path.join(C.OUT_DIR, "deploy_tune.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
