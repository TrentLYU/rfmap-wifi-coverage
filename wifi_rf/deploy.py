"""部署模型：物理趋势 + 实测残差（反距离加权）。

pred(x) = PL_ap(x; 当前家具) + Σ_i w_i(x)·[ rssi_i − PL_cal(x_i) ]

物理趋势负责随"AP 位置 / 墙体 / 家具"变化，实测残差负责这台路由器、
这套户型本身的偏移与多径特征。同一套公式在 web/nn.js 里逐元素实现，
浏览器与 Python 结果一致（见 scripts 校验）。
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .physics import path_loss_field, room_mask
from .realdata import sample_bilinear

PL0, NEXP = C.PL0, C.NEXP
AP_CAL = C.AP_POS
POWER = 2.0
EPS = 0.05


def deploy_field(xy_obs, y_obs, ap=None, obstacles=None, n=C.GRID_N, cell=C.CELL,
                 pl0=PL0, n_exp=NEXP, power=POWER) -> np.ndarray:
    ap = np.array(AP_CAL if ap is None else ap, dtype=np.float64)
    obstacles = obstacles or []
    phys = path_loss_field(ap, obstacles, pl0, n_exp, n, cell)
    xy_obs = np.asarray(xy_obs, dtype=np.float64)
    y_obs = np.asarray(y_obs, dtype=np.float64)
    if len(xy_obs) == 0:
        return np.where(room_mask(n, cell) > 0, phys, -120.0).astype(np.float32)

    cal = path_loss_field(np.array(AP_CAL), [], pl0, n_exp, n, cell)
    r = y_obs - sample_bilinear(cal, xy_obs)

    from .physics import cell_centers
    X, Y = cell_centers(n, cell)
    d = np.sqrt((X[..., None] - xy_obs[:, 0]) ** 2 + (Y[..., None] - xy_obs[:, 1]) ** 2)
    w = 1.0 / np.maximum(d, EPS) ** power
    resid = (w * r).sum(-1) / w.sum(-1)
    out = phys + resid
    return np.where(room_mask(n, cell) > 0, out, -120.0).astype(np.float32)
