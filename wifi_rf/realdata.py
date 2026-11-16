"""真实房间的输入构建与采样工具。"""
from __future__ import annotations

import os

import numpy as np
import torch

from . import config as C
from .physics import build_inputs, cell_centers, room_mask


def load_real() -> dict:
    z = np.load(os.path.join(C.DATA_DIR, "measured.npz"))
    return {"xy": z["xy"], "rssi": z["rssi"], "ap": z["ap"], "valid": z["valid"],
            "train_idx": z["train_idx"], "test_idx": z["test_idx"],
            "thresholds": z["thresholds"]}


def real_inputs(obs_idx: np.ndarray, mean: float, std: float, *,
                ap=None, obstacles=None, pl0: float | None = None,
                n_exp: float | None = None,
                grid_fields: dict | None = None) -> dict:
    """构造真实房间的 7 通道输入。

    obs_idx: 用作观测的实测点索引；grid_fields 可传入预计算结果以加速。
    pl0 / n_exp 默认取 config 中拟合好的路径损耗参数。
    """
    from .physics import path_loss_field

    pl0 = C.PL0 if pl0 is None else pl0
    n_exp = C.NEXP if n_exp is None else n_exp
    d = load_real()
    ap = np.array(d["ap"] if ap is None else ap, dtype=np.float64)
    obstacles = obstacles or []
    mask = room_mask()
    rm = (mask > 0)

    obs_xy = d["xy"][obs_idx]
    obs_val = d["rssi"][obs_idx]

    om = np.zeros((C.GRID_N, C.GRID_N), dtype=np.float32)
    ix = np.clip((obs_xy[:, 0] / C.CELL).astype(int), 0, C.GRID_N - 1)
    iy = np.clip((obs_xy[:, 1] / C.CELL).astype(int), 0, C.GRID_N - 1)
    om[iy, ix] = 1.0

    phys = path_loss_field(ap, obstacles, pl0, n_exp, C.GRID_N, C.CELL)
    x = build_inputs(np.zeros_like(mask), om, mask, obs_xy, obs_val, phys, mean, std)
    return {"x": x, "mask": mask, "obs_mask": om, "obs_xy": obs_xy, "obs_val": obs_val,
            "phys": phys, "ap": ap}


def sample_bilinear(field: np.ndarray, xy: np.ndarray,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """在网格上做双线性采样（xy 为米，格心在 (i+0.5)*CELL）。

    房间外常被填成 -120 dBm 的「无效」哨兵值。若直接在房外格点上插值，
    边界附近（尤其 AP 贴墙时）会被 -120 拖低，出现「读数很低但颜色很深」
    的矛盾。这里对含哨兵值的场默认只在有效角点上按权重归一化，
    与 web/index.html 的 sampleBilinear 完全一致。
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    n = field.shape[0]
    fx = np.clip(xy[:, 0] / C.CELL - 0.5, 0, n - 1.001)
    fy = np.clip(xy[:, 1] / C.CELL - 0.5, 0, n - 1.001)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, n - 1), np.minimum(y0 + 1, n - 1)
    tx, ty = fx - x0, fy - y0
    w = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)
    idx = ((y0, x0), (y0, x1), (y1, x0), (y1, x1))
    if mask is None and np.min(field) <= -119.0:
        mask = field > -119.0
    if mask is None:
        return (field[y0, x0] * w[0] + field[y0, x1] * w[1]
                + field[y1, x0] * w[2] + field[y1, x1] * w[3])
    num, den = np.zeros(len(xy)), np.zeros(len(xy))
    for wi, (iy, ix) in zip(w, idx):
        ok = mask[iy, ix] > 0
        num += np.where(ok, wi * field[iy, ix], 0.0)
        den += np.where(ok, wi, 0.0)
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), -120.0)


def predict_field(model, x: np.ndarray, mean: float, std: float, name: str,
                  tta: bool = False) -> np.ndarray:
    """把模型输出的标准化场还原成 dBm（房间外填 -120）。"""
    from .models import inputs_for

    t = torch.from_numpy(x[None]).float()
    with torch.no_grad():
        if tta:
            acc = 0.0
            for k in range(4):
                tt = t
                if k & 1:
                    tt = torch.flip(tt, dims=[3])
                if k & 2:
                    tt = torch.flip(tt, dims=[2])
                o = model(inputs_for(name, tt))
                if k & 1:
                    o = torch.flip(o, dims=[3])
                if k & 2:
                    o = torch.flip(o, dims=[2])
                acc = acc + o
            out = (acc / 4.0).numpy()[0, 0]
        else:
            out = model(inputs_for(name, t)).numpy()[0, 0]
    field = out * std + mean
    return np.where(x[2] > 0, field, -120.0).astype(np.float32)
