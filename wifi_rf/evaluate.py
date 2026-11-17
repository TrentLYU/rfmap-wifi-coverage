"""评估：30% 隐藏实测点上的误差与五级分级准确率。

口径：所有方法都只用 29 个训练测点作为输入，12 个测试点全程不参与，
只在最后用于评分；预测值在测试点坐标上做双线性采样。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import config as C
from . import models as M
from .data import level_of
from .physics import idw_prior, room_mask, path_loss_field
from .realdata import load_real, real_inputs, sample_bilinear, predict_field

METHODS = [
    ("ae", "方案一 · 卷积补全自编码器"),
    ("unet", "方案二 · 小型 U-Net"),
    ("resunet", "方案三 · 插值引导残差 U-Net"),
    ("unet_scatter", "方案四 · 散落布点训练的 U-Net"),
    ("pgmpr", "方案五 · 物理引导多先验残差 U-Net"),
]


# ---------------------------------------------------------------- 基线
def baseline_nearest(pack, xy):
    obs_xy, obs_val = pack["obs_xy"], pack["obs_val"]
    d = np.sqrt(((xy[:, None, :] - obs_xy[None]) ** 2).sum(-1))
    return obs_val[d.argmin(1)]


def baseline_idw(pack, xy, power=2.0):
    obs_xy, obs_val = pack["obs_xy"], pack["obs_val"]
    d = np.sqrt(((xy[:, None, :] - obs_xy[None]) ** 2).sum(-1))
    w = 1.0 / np.maximum(d, 0.05) ** power
    return (w * obs_val).sum(1) / w.sum(1)


def _gp_residual(pack, xy, length=3.0, lam=0.5):
    obs_xy = pack["obs_xy"]
    phys_obs = pack["phys"][
        np.clip((obs_xy[:, 1] / C.CELL).astype(int), 0, C.GRID_N - 1),
        np.clip((obs_xy[:, 0] / C.CELL).astype(int), 0, C.GRID_N - 1)]
    r = pack["obs_val"] - phys_obs
    d2 = ((obs_xy[:, None] - obs_xy[None]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * length ** 2)) + lam * np.eye(len(obs_xy))
    a = np.linalg.solve(K, r)
    d2p = ((xy[:, None] - obs_xy[None]) ** 2).sum(-1)
    kp = np.exp(-d2p / (2 * length ** 2))
    return kp @ a


def baseline_gp(pack, xy):
    """对数距离趋势（已拟合的 PL0 / n）+ RBF 残差。"""
    phys = path_loss_field(pack["ap"], [], C.PL0, C.NEXP, C.GRID_N, C.CELL)
    X, Y = np.meshgrid((np.arange(C.GRID_N) + 0.5) * C.CELL,
                       (np.arange(C.GRID_N) + 0.5) * C.CELL, indexing="xy")
    pts = np.stack([X, Y], -1).reshape(-1, 2)
    full = phys.reshape(-1)
    d2 = ((xy[:, None] - pack["obs_xy"][None]) ** 2).sum(-1)
    base_obs = pack["phys"][
        np.clip((pack["obs_xy"][:, 1] / C.CELL).astype(int), 0, C.GRID_N - 1),
        np.clip((pack["obs_xy"][:, 0] / C.CELL).astype(int), 0, C.GRID_N - 1)]
    r = pack["obs_val"] - base_obs
    length, lam = 3.0, 0.5
    do2 = ((pack["obs_xy"][:, None] - pack["obs_xy"][None]) ** 2).sum(-1)
    K = np.exp(-do2 / (2 * length ** 2)) + lam * np.eye(len(r))
    a = np.linalg.solve(K, r)
    kp = np.exp(-d2 / (2 * length ** 2))
    trend = path_loss_field(pack["ap"], [], C.PL0, C.NEXP, C.GRID_N, C.CELL)
    from .realdata import sample_bilinear
    return sample_bilinear(trend, xy) + kp @ a


# ---------------------------------------------------------------- 指标
def metrics(true, pred, levels_true, levels_pred):
    e = pred - true
    exact = float((levels_pred == levels_true).mean())
    within1 = float((np.abs(levels_pred - levels_true) <= 1).mean())
    binary_true = levels_true >= 3
    binary_pred = levels_pred >= 3
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "max_abs": float(np.max(np.abs(e))),
        "bias": float(np.mean(e)),
        "level_acc": exact,
        "level_acc_pm1": within1,
        "binary_acc": float((binary_true == binary_pred).mean()),
        "n": int(len(true)),
    }


def run(save_name: str = "metrics.json") -> dict:
    with open(os.path.join(C.DATA_DIR, "sim_meta.json"), encoding="utf-8") as f:
        smeta = json.load(f)
    mean, std = smeta["mean"], smeta["std"]
    real = load_real()
    tr = real["train_idx"]
    te = real["test_idx"]
    thr = [float(t) for t in real["thresholds"]]
    tx, ty = real["xy"][te], real["rssi"][te]
    lv_true = level_of(ty, thr)

    pack = real_inputs(tr, mean, std)
    x = pack["x"]

    results = []
    # ---- 基线
    for key, label in (("nearest", "基线 · 最近邻"),
                       ("idw", "基线 · 反距离加权 (IDW, p=2)"),
                       ("gp", "基线 · 对数距离趋势 + RBF 残差")):
        if key == "nearest":
            pred = baseline_nearest(pack, tx)
        elif key == "idw":
            pred = baseline_idw(pack, tx)
        else:
            pred = baseline_gp(pack, tx)
        results.append({"key": key, "label": label, "kind": "baseline",
                        **metrics(ty, pred, lv_true, level_of(pred, thr)),
                        "pred": pred.tolist()})

    # ---- 五个网络
    fields = {}
    for key, label in METHODS:
        model = M.build(key)
        ckpt = os.path.join(C.MODEL_DIR, key, "weights.pt")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        for tta in (False, True):
            f = predict_field(model, x, mean, std, key, tta=tta)
            pred = sample_bilinear(f, tx)
            name = key + ("_tta" if tta else "")
            results.append({"key": name, "label": label + ("（TTA）" if tta else ""),
                            "kind": "net",
                            **metrics(ty, pred, lv_true, level_of(pred, thr)),
                            "pred": pred.tolist()})
            fields[name] = f

    # 部署模型：物理趋势 + 实测残差（IDW p=2），只用 29 个标定点
    from .deploy import deploy_field
    dep = deploy_field(pack["obs_xy"], pack["obs_val"], pack["ap"])
    pred_dep = sample_bilinear(dep, tx)
    results.append({"key": "deploy", "label": "部署 · 物理+实测残差", "kind": "deploy",
                    **metrics(ty, pred_dep, lv_true, level_of(pred_dep, thr)),
                    "pred": pred_dep.tolist()})
    fields["deploy"] = dep

    # 记录每个测试点的明细（用无 TTA 版本）
    detail = []
    for j, i in enumerate(te):
        row = {"idx": int(i), "x": float(tx[j, 0]), "y": float(tx[j, 1]),
               "true": float(ty[j]), "true_level": int(lv_true[j])}
        for key, _ in METHODS:
            row[key] = float(fields[key][
                np.clip(int(tx[j, 1] / C.CELL), 0, C.GRID_N - 1),
                np.clip(int(tx[j, 0] / C.CELL), 0, C.GRID_N - 1)])
        row["deploy"] = float(pred_dep[j])
        for key in ("nearest", "idw", "gp"):
            r = next(r for r in results if r["key"] == key)
            row[key] = float(r["pred"][j])
        detail.append(row)

    # ---- 输入点数扫描（K = 5/10/15/20/25）
    scan = []
    rng = np.random.default_rng(123)
    n = len(tr)
    for k in (5, 10, 15, 20, 25):
        reps = 6
        acc = {key: [] for key, _ in METHODS}
        acc["idw"] = []
        acc["gp"] = []
        for _ in range(reps):
            sub = np.sort(rng.choice(n, size=min(k, n), replace=False))
            pk = real_inputs(tr[sub], mean, std)
            for key, _ in METHODS:
                model = M.build(key)
                model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, key, "weights.pt"),
                                                 map_location="cpu"))
                model.eval()
                f = predict_field(model, pk["x"], mean, std, key)
                p = sample_bilinear(f, tx)
                acc[key].append(float(np.sqrt(np.mean((p - ty) ** 2))))
            acc["idw"].append(float(np.sqrt(np.mean((baseline_idw(pk, tx) - ty) ** 2))))
            acc["gp"].append(float(np.sqrt(np.mean((baseline_gp(pk, tx) - ty) ** 2))))
        for key, vals in acc.items():
            scan.append({"k": k, "key": key, "rmse_mean": float(np.mean(vals)),
                         "rmse_std": float(np.std(vals))})

    # ---- 零观测（纯户型 + AP）能力测试
    empty = real_inputs(np.array([], dtype=int), mean, std)
    zero_obs = []
    valid = real["valid"]
    for key, label in METHODS:
        model = M.build(key)
        model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, key, "weights.pt"),
                                         map_location="cpu"))
        model.eval()
        f = predict_field(model, empty["x"], mean, std, key)
        p = sample_bilinear(f, real["xy"][valid])
        zero_obs.append({"key": key, "label": label,
                         "rmse": float(np.sqrt(np.mean((p - real["rssi"][valid]) ** 2))),
                         "mae": float(np.mean(np.abs(p - real["rssi"][valid])))})

    out = {"thresholds": thr, "n_train": int(len(tr)), "n_test": int(len(te)),
           "results": results, "detail": detail, "scan": scan, "zero_obs": zero_obs,
           "test_idx": te.tolist(), "train_idx": tr.tolist(),
           "mean": mean, "std": std}
    with open(os.path.join(C.OUT_DIR, save_name), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    r = run()
    for row in r["results"]:
        print(f"{row['label']:38s} RMSE {row['rmse']:6.3f}  MAE {row['mae']:6.3f}  "
              f"分级 {row['level_acc']*100:5.1f}%  ±1 {row['level_acc_pm1']*100:5.1f}%  "
              f"二分类 {row['binary_acc']*100:5.1f}%")
    print("零观测:", [(z["key"], round(z["rmse"], 2)) for z in r["zero_obs"]])
