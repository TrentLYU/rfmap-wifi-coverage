"""物理仿真数据生成（多户型 RSSI 覆盖真值）。

为什么需要仿真
--------------
实测只有 1 张地图（49 点），直接训练必然过拟合。这里按室内对数距离路损 +
障碍穿透损耗 + 空间相关阴影场，合成大量随机户型的整图真值，供网络学习
"从稀疏观测补全整图"的通用先验。

模型口径（与任务书 / docs/03 一致）
-----------------------------------
    PL(d) = PL0 + 10*n*log10(max(d,0.5)/1.0)
    rssi  = -PL(d) - sum(障碍穿透损耗) + 阴影场 + 逐点噪声
其中近场用 0.5 m 截断，避免 log 发散并让 AP 附近自然成为最强区。
只依赖 numpy / scipy，网格与几何类直接复用 rfmap 既有模块，避免口径漂移。
"""

from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import gaussian_filter

from .grid import Grid

__all__ = ['random_room', 'simulate_rssi', 'make_sample', 'build_dataset',
           'reobserve_dataset', 'RSSI_MIN', 'RSSI_MAX']

RSSI_MIN = -95.0
RSSI_MAX = -25.0


# --------------------------------------------------------------------------
# 几何辅助
# --------------------------------------------------------------------------
def _overlap(a: dict, b: dict) -> bool:
    """两矩形是否重叠（边相切不算）。"""
    return (a['x0'] < b['x1'] and b['x0'] < a['x1']
            and a['y0'] < b['y1'] and b['y0'] < a['y1'])


def _in_obstacle(x: float, y: float, obstacles: list[dict]) -> bool:
    return any(o['x0'] <= x <= o['x1'] and o['y0'] <= y <= o['y1']
               for o in obstacles)


def random_room(rng, grid) -> dict:
    """生成一个随机户型（矩形房间 + 若干内部障碍/结构柱）。

    为什么用矩形 + 障碍而非任意多边形：网络输入是规则栅格，矩形外墙加随机障碍
    已能提供"尺寸 / 长宽比 / 障碍布局 / AP 位置"四个维度的多样性，且掩码可用
    纯向量化布尔运算精确求出，避免多边形填充的边界歧义。
    面积占比被约束在 30%~80%，既保证房间信息量足够，又避免贴满整块网格。
    """
    gx0, gy0 = grid.x0, grid.y0
    gw = grid.x_max - grid.x0
    gh = grid.y_max - grid.y0
    full_area = gw * gh
    X, Y = grid.centers()

    for _ in range(64):
        # 先用面积占比锁定房间大小，再让长宽比随机，覆盖大小/狭长两类户型
        frac = rng.uniform(0.40, 0.72)
        w = rng.uniform(7.5, 11.3)
        h = frac * full_area / w
        if not (7.5 <= h <= 11.3):
            continue
        rx0 = gx0 + rng.uniform(0.0, max(gw - w, 0.0))
        ry0 = gy0 + rng.uniform(0.0, max(gh - h, 0.0))
        rx1, ry1 = rx0 + w, ry0 + h

        # 内部障碍：0~3 个，尺寸适中，互不重叠（贴墙柱/家具的抽象）
        obstacles: list[dict] = []
        for _n in range(int(rng.integers(0, 4))):
            for _try in range(20):
                ow = rng.uniform(0.4, 1.8)
                oh = rng.uniform(0.4, 1.8)
                if rx1 - ow <= rx0 or ry1 - oh <= ry0:
                    break
                ox0 = rng.uniform(rx0, rx1 - ow)
                oy0 = rng.uniform(ry0, ry1 - oh)
                rect = {'x0': ox0, 'x1': ox0 + ow,
                        'y0': oy0, 'y1': oy0 + oh}
                if not any(_overlap(rect, o) for o in obstacles):
                    obstacles.append(rect)
                    break

        mask = (X >= rx0) & (X <= rx1) & (Y >= ry0) & (Y <= ry1)
        for o in obstacles:
            mask &= ~((X >= o['x0']) & (X <= o['x1'])
                      & (Y >= o['y0']) & (Y <= o['y1']))

        frac_real = float(mask.mean())
        if not (0.32 <= frac_real <= 0.78):
            continue

        # AP 放在房间内、避开门内障碍
        for _try in range(100):
            ax = rng.uniform(rx0 + 0.3, rx1 - 0.3)
            ay = rng.uniform(ry0 + 0.3, ry1 - 0.3)
            if not _in_obstacle(ax, ay, obstacles):
                break

        polygon = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]
        return {'mask': mask,
                'polygon': polygon,
                'ap': (float(ax), float(ay)),
                'obstacles': obstacles,
                'room_wh': (float(w), float(h))}

    # 兜底：极端 rng 下也要返回一个合法房间
    rx0, ry0, rx1, ry1 = gx0 + 1.0, gy0 + 1.0, gx0 + 10.0, gy0 + 10.0
    mask = (X >= rx0) & (X <= rx1) & (Y >= ry0) & (Y <= ry1)
    return {'mask': mask,
            'polygon': [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)],
            'ap': (rx0 + 4.5, ry0 + 4.5),
            'obstacles': [],
            'room_wh': (9.0, 9.0)}


# --------------------------------------------------------------------------
# 线段与矩形相交（向量化 Liang-Barsky）
# --------------------------------------------------------------------------
def _slab(P, D, lo, hi):
    """线段在某坐标轴 [lo,hi] 板内的参数区间 (tmin,tmax)。"""
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - P) / D
        t2 = (hi - P) / D
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    parallel = np.abs(D) < 1e-12
    inside = (P >= lo) & (P <= hi)
    tmin = np.where(parallel, np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(parallel, np.where(inside, np.inf, -np.inf), tmax)
    return tmin, tmax


def _crossing_mask(X, Y, ax, ay, rect) -> np.ndarray:
    """格点中心到 AP 的线段是否穿过矩形内部（两端点都在矩形外时为"穿过"）。"""
    dx = ax - X
    dy = ay - Y
    txmin, txmax = _slab(X, dx, rect['x0'], rect['x1'])
    tymin, tymax = _slab(Y, dy, rect['y0'], rect['y1'])
    t_enter = np.maximum(txmin, tymin)
    t_exit = np.minimum(txmax, tymax)
    # 与 (0,1) 有交叠才算穿过，避免把端点贴在矩形上误判
    return (t_enter <= t_exit) & (t_exit > 0.0) & (t_enter < 1.0)


def simulate_rssi(room, rng, grid) -> np.ndarray:
    """按物理模型生成整图真值 RSSI（dBm），房间外为 np.nan。"""
    X, Y = grid.centers()
    mask = np.asarray(room['mask'], dtype=bool)
    ax, ay = room['ap']
    d = np.hypot(X - ax, Y - ay)

    # 对数距离路损：室内参考点 1m、指数 1.8~3.2 覆盖开阔~隔断场景
    pl0 = rng.uniform(32.0, 46.0)
    n = rng.uniform(1.8, 3.2)
    path_loss = pl0 + 10.0 * n * np.log10(np.maximum(d, 0.5) / 1.0)

    # 障碍穿透损耗：每种障碍按材质给一个固定系数，再按是否穿过累加，
    # 逐格重抽会退化成噪声并破坏空间连续性
    wall_att = np.zeros(X.shape, dtype=float)
    for o in room['obstacles']:
        coef = rng.uniform(4.0, 14.0)
        wall_att += coef * _crossing_mask(X, Y, ax, ay, o)

    # 空间相关阴影场：白噪声做大核高斯滤波，再缩放到目标标准差
    sigma_cells = rng.uniform(1.0, 4.0)
    target_std = rng.uniform(1.5, 4.5)
    smooth = gaussian_filter(rng.standard_normal(X.shape),
                             sigma=sigma_cells, mode='nearest')
    s = float(smooth.std())
    shadow = smooth / s * target_std if s > 1e-9 else np.zeros_like(smooth)

    # 逐点独立噪声：幅度 U(0.3,1.2)，随机正负以保持无偏
    mag = rng.uniform(0.3, 1.2, size=X.shape)
    noise = mag * np.where(rng.random(X.shape) < 0.5, -1.0, 1.0)

    rssi = -path_loss - wall_att + shadow + noise
    rssi = np.clip(rssi, RSSI_MIN, RSSI_MAX).astype(np.float64)
    rssi[~mask] = np.nan
    return rssi


# --------------------------------------------------------------------------
# 观测点抽样
# --------------------------------------------------------------------------
def _grid_obs(rng, grid, mask, js, iss, n) -> list[tuple[int, int]]:
    """近似等距网格 + 抖动布点，返回去重后的 (j,i) 列表。"""
    xc = grid.x_coords()[iss]
    yc = grid.y_coords()[js]
    xmin, xmax = float(xc.min()), float(xc.max())
    ymin, ymax = float(yc.min()), float(yc.max())
    w = max(xmax - xmin, grid.cell)
    h = max(ymax - ymin, grid.cell)
    step = float(np.sqrt(w * h / max(n, 1)))
    ncol = max(1, int(round(w / step)))
    nrow = max(1, int(round(h / step)))
    gxs = np.linspace(xmin, xmax, ncol)
    gys = np.linspace(ymin, ymax, nrow)
    ny, nx = mask.shape
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for yy in gys:
        for xx in gxs:
            jx = xx + rng.uniform(-0.4, 0.4) * step
            jy = yy + rng.uniform(-0.4, 0.4) * step
            jj, ii = grid.xy_to_idx(jx, jy)
            if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]:
                key = (int(jj), int(ii))
                if key not in seen:
                    seen.add(key)
                    out.append(key)
    return out


def _scatter_obs(rng, grid, mask, n) -> np.ndarray:
    """散落式布点：模拟现场"随手多挑几个点"，刻意**不保证等距覆盖**。

    与 :func:`_grid_obs` 的差别正是**方案四**要考察的变量：

    * 等距网格布点隐含一个强先验 —— 观测点均匀铺满房间，任何位置附近都有已知点；
    * 真实现场布点常见两种形态，本函数都覆盖：

      1. **纯随机撒点**（最散）：完全不管间距，可能出现大片空白区；
      2. **簇状聚集**：人站在某处连测几点（如沿墙、家具旁），形成 2~4 个局部点团，
         点团之间可能隔得很远。

    这两种形态都会让某些区域"最近观测点距离"显著变大，从而考验模型的
    **外推**能力，而不是只做局部插值。
    """
    js, iss = np.nonzero(mask)
    m = int(js.size)
    n = int(min(n, m))
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    # 50% 纯随机、50% 簇状（点数太少时退化为纯随机，否则簇心撑不开）
    if rng.random() < 0.5 or n < 4:
        sel = rng.choice(m, size=n, replace=False)
        return np.stack([js[sel], iss[sel]], axis=1).astype(np.int64)

    n_cluster = int(rng.integers(2, 5))
    centers = rng.choice(m, size=min(n_cluster, m), replace=False)
    # 簇内散布尺度：0.3~1.6 m（cell=0.2 m，故 1.5~8 格）
    sigma_cells = rng.uniform(1.5, 8.0)

    ny, nx = mask.shape
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    guard = 0
    while len(out) < n and guard < n * 400:
        guard += 1
        c = int(centers[int(rng.integers(len(centers)))])
        dj, di = rng.normal(0.0, sigma_cells, 2)
        jj, ii = int(round(js[c] + dj)), int(round(iss[c] + di))
        if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]:
            key = (jj, ii)
            if key not in seen:
                seen.add(key)
                out.append(key)

    # 簇状采样可能因落在房间外/柱体内而凑不满，用全局随机补齐
    if len(out) < n:
        pool = rng.permutation(m)
        for k in pool:
            if len(out) >= n:
                break
            key = (int(js[k]), int(iss[k]))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return np.array(out, dtype=np.int64)


def _pick_obs(rng, grid, mask, n_obs, mode: str = 'mixed') -> np.ndarray:
    """抽取观测点位置。

    Parameters
    ----------
    mode : str
        ``'mixed'``（默认，历史数据集用）：约 50% 等距网格+抖动、50% 纯随机；
        ``'scatter'``：100% 散落式（方案四），见 :func:`_scatter_obs`；
        ``'grid'``：100% 等距网格+抖动（仅作对照诊断用）。
    """
    js, iss = np.nonzero(mask)
    m = int(js.size)
    n = int(min(n_obs, m))

    if mode == 'scatter':
        return _scatter_obs(rng, grid, mask, n)
    if mode == 'grid':
        cand = _grid_obs(rng, grid, mask, js, iss, n)
        if len(cand) > n:
            sel = rng.choice(len(cand), size=n, replace=False)
            cand = [cand[int(k)] for k in sel]
        return np.array(cand, dtype=np.int64)

    if rng.random() < 0.5:
        cand = _grid_obs(rng, grid, mask, js, iss, n)
        if len(cand) > n:
            sel = rng.choice(len(cand), size=n, replace=False)
            cand = [cand[int(k)] for k in sel]
        elif len(cand) < n:
            have = set(cand)
            pool = np.arange(m)
            rng.shuffle(pool)
            for k in pool:
                if len(cand) >= n:
                    break
                key = (int(js[k]), int(iss[k]))
                if key not in have:
                    have.add(key)
                    cand.append(key)
        idx = np.array(cand, dtype=np.int64)
    else:
        sel = rng.choice(m, size=n, replace=False)
        idx = np.stack([js[sel], iss[sel]], axis=1).astype(np.int64)
    return idx


def make_sample(room, rng, grid, n_obs, obs_mode: str = 'mixed') -> dict:
    """在一张户型图上抽观测点，打包成网络训练样本。"""
    ny, nx = grid.shape
    mask = np.asarray(room['mask'], dtype=bool)
    rssi = simulate_rssi(room, rng, grid)
    target = np.where(mask, rssi, 0.0).astype(np.float32)

    obs_idx = _pick_obs(rng, grid, mask, int(n_obs), mode=obs_mode)
    obs_value = np.zeros((ny, nx), dtype=np.float32)
    obs_mask = np.zeros((ny, nx), dtype=np.float32)
    jj, ii = obs_idx[:, 0], obs_idx[:, 1]
    obs_value[jj, ii] = target[jj, ii]
    obs_mask[jj, ii] = 1.0

    return {'obs_value': obs_value,
            'obs_mask': obs_mask,
            'room': mask.astype(np.float32),
            'target': target,
            'obs_idx': obs_idx.astype(np.int32)}


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------
def build_dataset(n_samples, seed, out_path, grid, obs_range=(5, 30),
                  verbose=True, obs_mode: str = 'mixed') -> str:
    """生成数据集并保存为 .npz（固定 seed 完全可复现）。

    管线共用同一个 rng 顺序消费随机数，因此相同 seed 的两次调用逐元素一致。
    ``obs_mode`` 只影响观测点位置，不改动机理；默认 ``'mixed'`` 保证历史数据集
    可原样重建。
    """
    rng = np.random.default_rng(seed)
    ny, nx = grid.shape
    obs_value = np.zeros((n_samples, ny, nx), dtype=np.float32)
    obs_mask = np.zeros((n_samples, ny, nx), dtype=np.float32)
    room_arr = np.zeros((n_samples, ny, nx), dtype=np.float32)
    target = np.zeros((n_samples, ny, nx), dtype=np.float32)
    room_wh = np.zeros((n_samples, 2), dtype=np.float32)

    lo, hi = int(obs_range[0]), int(obs_range[1])
    for k in range(n_samples):
        room = random_room(rng, grid)
        n_obs = int(rng.integers(lo, hi + 1))
        s = make_sample(room, rng, grid, n_obs, obs_mode=obs_mode)
        obs_value[k] = s['obs_value']
        obs_mask[k] = s['obs_mask']
        room_arr[k] = s['room']
        target[k] = s['target']
        room_wh[k] = room['room_wh']
        if verbose and (k + 1) % 200 == 0:
            print('  %d/%d' % (k + 1, n_samples))

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(out_path, obs_value=obs_value, obs_mask=obs_mask,
                        room=room_arr, target=target, room_wh=room_wh)
    if verbose:
        print('saved ->', out_path)
    return out_path


def reobserve_dataset(src_path, dst_path, seed, grid, obs_mode='scatter',
                      count_range=None, verbose=True) -> str:
    """复用已有数据集的**户型和信号真值**，只重新抽取观测点位置。

    为什么必须这样做（方案四的控制变量）
    ------------------------------------
    方案四要回答的是一个单因素问题："把训练时的观测点布得更散落，模型会不会更准？"

    如果连户型和真值一起重新生成，两者差异里就同时混进了"户型分布"和"布点方式"
    两个变量，实验结论无法归因。本函数直接逐元素复用 ``src`` 的 ``room`` / ``target``
    数组，只用新的 rng 重抽观测点，使**唯一变量是布点方式**。

    Parameters
    ----------
    count_range : tuple | None
        ``None``（默认）表示**沿用每个样本原本的观测点个数** —— 位置变、数量不变，
        这是最严格的消融；给定 ``(lo, hi)`` 则同时改变点数分布（"更散且更少"）。
    """
    d = np.load(src_path)
    room = d['room'].astype(np.float32)
    target = d['target'].astype(np.float32)
    room_wh = d['room_wh']
    n = room.shape[0]

    rng = np.random.default_rng(seed)
    obs_value = np.zeros((n,) + grid.shape, dtype=np.float32)
    obs_mask = np.zeros((n,) + grid.shape, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)

    for k in range(n):
        mask = room[k] > 0.5
        if count_range is None:
            n_obs = int(round(float(d['obs_mask'][k].sum())))
        else:
            n_obs = int(rng.integers(int(count_range[0]), int(count_range[1]) + 1))
        idx = _pick_obs(rng, grid, mask, n_obs, mode=obs_mode)
        if idx.size:
            jj, ii = idx[:, 0], idx[:, 1]
            obs_value[k][jj, ii] = target[k][jj, ii]
            obs_mask[k][jj, ii] = 1.0
        counts[k] = int(idx.shape[0])
        if verbose and (k + 1) % 400 == 0:
            print('  重抽观测点 %d/%d' % (k + 1, n), flush=True)

    out_dir = os.path.dirname(dst_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(dst_path, obs_value=obs_value, obs_mask=obs_mask,
                        room=room, target=target, room_wh=room_wh)
    if verbose:
        print('saved -> %s（观测点/样本：均值 %.1f，范围 %d~%d）'
              % (dst_path, counts.mean(), counts.min(), counts.max()))
    return dst_path


def main(argv=None) -> int:
    """命令行入口：按固定参数生成 train/val/test 三份数据集。"""
    import argparse
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description='物理仿真 RSSI 数据集生成')
    ap.add_argument('--out-dir', default=os.path.join(root, 'data/sim'))
    ap.add_argument('--grid', default=os.path.join(root, 'data/rfmap/grid.json'))
    args = ap.parse_args(argv)

    grid = Grid.load(args.grid) if os.path.exists(args.grid) else Grid()
    for name, n, seed in (('train', 1200, 20261021),
                          ('val', 200, 20261022),
                          ('test', 200, 20261023)):
        build_dataset(n, seed, os.path.join(args.out_dir, name + '.npz'), grid)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
