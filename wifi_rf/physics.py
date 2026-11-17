"""几何 / 物理层：网格、射线-矩形衰减、路径损耗场、先验图。

**JS 前端使用同一套公式**（web/nn.js + web/app.js），因此这里的实现
必须保持纯 numpy、无隐式状态，方便逐元素对照。
"""
from __future__ import annotations

import numpy as np

from . import config as C


# ---------------------------------------------------------------- 网格
def cell_centers(n: int = C.GRID_N, cell: float = C.CELL) -> tuple[np.ndarray, np.ndarray]:
    """返回 (n, n) 的格心坐标 X, Y（米）。原点在房间左下角。"""
    idx = (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(idx, idx, indexing="xy")
    return X, Y


def room_mask(n: int = C.GRID_N, cell: float = C.CELL,
              w: float = C.ROOM_W, h: float = C.ROOM_H) -> np.ndarray:
    """房间有效区掩码（1=室内）。"""
    X, Y = cell_centers(n, cell)
    return ((X <= w) & (Y <= h)).astype(np.float32)


# ---------------------------------------------------------------- 射线-矩形
def seg_rect_hit(p0, p1, rect) -> np.ndarray:
    """向量化的线段-矩形相交判定（Liang–Barsky）。

    p0: (2,) 起点；p1: (...,2) 终点数组；rect: (x0, x1, y0, y1)
    返回与 p1 形状一致的 0/1 数组。
    """
    p1 = np.asarray(p1, dtype=np.float64)
    x0, x1, y0, y1 = rect
    ax, ay = float(p0[0]), float(p0[1])
    dx = p1[..., 0] - ax
    dy = p1[..., 1] - ay
    tmin = np.zeros_like(dx)
    tmax = np.ones_like(dx)
    ok = np.ones_like(dx, dtype=bool)

    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        # p*t <= q ；p == 0 时要求 q >= 0
        flat_p = p
        parallel = np.abs(flat_p) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(parallel, 0.0, q / np.where(parallel, 1.0, flat_p))
        ok &= ~(parallel & (q < 0))
        pos = (~parallel) & (flat_p > 0)
        neg = (~parallel) & (flat_p < 0)
        tmax = np.where(pos, np.minimum(tmax, t), tmax)
        tmin = np.where(neg, np.maximum(tmin, t), tmin)

    return (ok & (tmax >= tmin) & (tmax >= 0) & (tmin <= 1)).astype(np.float32)


def obstacle_field(ap, obstacles, n: int = C.GRID_N, cell: float = C.CELL) -> np.ndarray:
    """AP 到每个格心的连线穿过障碍物造成的总衰减（dB，正数）。"""
    X, Y = cell_centers(n, cell)
    pts = np.stack([X, Y], axis=-1)
    att = np.zeros((n, n), dtype=np.float64)
    for ob in obstacles:
        rect = (ob["x"] - ob["w"] / 2, ob["x"] + ob["w"] / 2,
                ob["y"] - ob["h"] / 2, ob["y"] + ob["h"] / 2)
        att += float(ob.get("db", 6.0)) * seg_rect_hit(ap, pts, rect)
    return att.astype(np.float32)


def path_loss_field(ap, obstacles, pl0: float, n_exp: float,
                    n: int = C.GRID_N, cell: float = C.CELL,
                    include_obstacles: bool = True) -> np.ndarray:
    """PL(d) = PL0 + 10·n·log10(max(d,1)) + 障碍衰减；返回 RSSI(dBm) = −PL。"""
    X, Y = cell_centers(n, cell)
    d = np.maximum(np.hypot(X - ap[0], Y - ap[1]), 1.0)
    pl = pl0 + 10.0 * n_exp * np.log10(d)
    if include_obstacles and obstacles:
        pl = pl + obstacle_field(ap, obstacles, n, cell)
    return (-pl).astype(np.float32)


# ---------------------------------------------------------------- 先验图
def idw_prior(obs_xy: np.ndarray, obs_val: np.ndarray, mask: np.ndarray,
              n: int = C.GRID_N, cell: float = C.CELL, power: float = 2.0,
              fill: float = 0.0) -> np.ndarray:
    """反距离加权插值先验；房间外/无观测处填 fill。"""
    X, Y = cell_centers(n, cell)
    if len(obs_xy) == 0:
        return np.full((n, n), fill, dtype=np.float32)
    d = np.sqrt((X[..., None] - obs_xy[:, 0]) ** 2 + (Y[..., None] - obs_xy[:, 1]) ** 2)
    d = np.maximum(d, 0.05)
    w = 1.0 / d ** power
    out = (w * obs_val).sum(-1) / w.sum(-1)
    out = np.where(mask > 0, out, fill)
    return out.astype(np.float32)


def dist_to_obs(obs_xy: np.ndarray, n: int = C.GRID_N, cell: float = C.CELL,
                cap: float = 6.0) -> np.ndarray:
    X, Y = cell_centers(n, cell)
    if len(obs_xy) == 0:
        return np.ones((n, n), dtype=np.float32)
    d = np.sqrt((X[..., None] - obs_xy[:, 0]) ** 2 + (Y[..., None] - obs_xy[:, 1]) ** 2)
    return np.clip(d.min(-1) / cap, 0.0, 1.0).astype(np.float32)


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """可分离高斯模糊（边界按反射填充）。"""
    r = int(max(1, round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="reflect"), k, "valid"), 0, a)
    out = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="reflect"), k, "valid"), 1, out)
    return out.astype(np.float32)


def obs_density(mask_obs: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    b = gaussian_blur(mask_obs.astype(np.float32), sigma)
    mx = b.max()
    return (b / mx if mx > 1e-6 else b).astype(np.float32)


def build_inputs(rssi_field: np.ndarray, obs_mask: np.ndarray, room_m: np.ndarray,
                 obs_xy: np.ndarray, obs_val: np.ndarray, prior_phys: np.ndarray,
                 mean: float, std: float) -> np.ndarray:
    """方案五的 7 通道输入（训练/推理共用）。"""
    obs_vals = np.zeros_like(rssi_field, dtype=np.float32)
    if len(obs_xy):
        ix = np.clip((obs_xy[:, 0] / C.CELL).astype(int), 0, C.GRID_N - 1)
        iy = np.clip((obs_xy[:, 1] / C.CELL).astype(int), 0, C.GRID_N - 1)
        obs_vals[iy, ix] = obs_val
    ch0 = np.where(obs_mask > 0, (obs_vals - mean) / std, 0.0)
    ch3 = np.where(room_m > 0, (prior_phys - mean) / std, 0.0)
    ch6 = (idw_prior(obs_xy, obs_val, room_m) - mean) / std
    ch6 = np.where(room_m > 0, ch6, 0.0)
    return np.stack([
        ch0.astype(np.float32),
        obs_mask.astype(np.float32),
        room_m.astype(np.float32),
        ch3.astype(np.float32),
        dist_to_obs(obs_xy),
        obs_density(obs_mask),
        ch6.astype(np.float32),
    ]).astype(np.float32)
