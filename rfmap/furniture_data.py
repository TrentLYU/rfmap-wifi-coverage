"""放家具场景的实测数据、家具几何与物理仿真。

坐标约定
--------
原点为室内西南角，x 向东、y 向北，单位 m。该约定与
``实验室户型图（加入家具版本）.png`` 的家具定位一致。实测表的 y=12
按 11.731/12 对齐到北内墙，x 保持现场标称值。

RSSI 符号
---------
工作表前 10 列把 RSSI 写成了正数，而后 2 列为负数。Wi-Fi RSSI 在本实验中
应为负值，因此统一使用 ``-abs(raw)``，同时保留原始读数，绝不静默覆盖源文件。

家具不是房间外区域：桌椅占据掩码是独立输入通道，RSSI 真值与损失仍在家具
覆盖的有效室内格点上定义。仿真中的家具衰减只是可复现的简化射线遮挡模型，
不代表对材质、多径或真实家具因果效应的精确测量。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from . import geometry as G
from .grid import Grid, DEFAULT as DEFAULT_GRID
from .simulate import _crossing_mask, _pick_obs

Y_SCALE = G.INNER_H / 12.0
FURNITURE_RADIUS = 0.4
RSSI_MIN = -95.0
RSSI_MAX = -15.0
SIM_SEEDS = {'train': 20261121, 'val': 20261122, 'test': 20261123}
SCATTER_SEEDS = {'train': 20261131, 'val': 20261132, 'test': 20261133}


@dataclass(frozen=True)
class MeasuredPoint:
    idx: int
    x_data: float
    y_data: float
    raw_rssi: float
    rssi: float
    x: float
    y: float
    inside_room: bool

    def to_row(self) -> dict:
        return {
            'idx': self.idx,
            'x_data': self.x_data,
            'y_data': self.y_data,
            'raw_rssi': self.raw_rssi,
            'rssi_dbm': self.rssi,
            'x_room': round(self.x, 4),
            'y_room': round(self.y, 4),
            'inside_room': self.inside_room,
            'br04_status': '正常' if -70.0 <= self.rssi <= -40.0 else
                           ('偏强' if self.rssi > -40.0 else '偏弱'),
            'sign_corrected': self.raw_rssi > 0,
        }


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_workbook(path: str) -> tuple[list[MeasuredPoint], list[dict], dict]:
    """读取 RSSI 矩阵和两组家具中心，返回测点、家具与质检信息。"""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    xs = [float(ws.cell(1, c).value) for c in range(2, 14)]
    pts: list[MeasuredPoint] = []
    for r in range(2, 14):
        y = float(ws.cell(r, 1).value)
        for c, x in zip(range(2, 14), xs):
            raw = ws.cell(r, c).value
            if raw is None:
                continue
            raw = float(raw)
            xr = x
            yr = y * Y_SCALE
            pts.append(MeasuredPoint(
                idx=len(pts) + 1, x_data=x, y_data=y, raw_rssi=raw,
                rssi=-abs(raw), x=xr, y=yr,
                inside_room=G.inside_room(xr, yr, tol=1e-3)))

    furniture: list[dict] = []
    # 西南组：距 3 号南墙为 y，距 4 号西墙为 x。
    for c in range(2, 6):
        name = str(ws.cell(15, c).value)
        y = float(ws.cell(16, c).value)
        x = float(ws.cell(17, c).value)
        furniture.append({
            'name': name, 'kind': 'table' if '桌' in name else 'chair',
            'x': x, 'y': y, 'radius': FURNITURE_RADIUS,
            'source': {'distance_south_wall': y, 'distance_west_wall': x},
            'position_rule': '实测中心点；未使用示意图防重叠外移位置',
        })
    # 东北组：x=净宽-距 2 号东墙，y=净高-距 1 号北墙。
    for c in range(2, 6):
        name = str(ws.cell(19, c).value)
        dn = float(ws.cell(20, c).value)
        de = float(ws.cell(21, c).value)
        x = G.INNER_W - de
        y = G.INNER_H - dn
        furniture.append({
            'name': name, 'kind': 'table' if '桌' in name else 'chair',
            'x': x, 'y': y, 'radius': FURNITURE_RADIUS,
            'source': {'distance_north_wall': dn, 'distance_east_wall': de},
            'position_rule': '实测中心点；未使用示意图防重叠外移位置',
        })

    quality = quality_report(pts, furniture)
    quality.update({
        'source_file': os.path.basename(path),
        'source_sha256': _sha256(path),
        'sheet': ws.title,
        'coordinate_convention': '原点=室内西南角；x 向东；y 向北',
        'x_scale': 1.0,
        'y_scale': Y_SCALE,
        'rssi_rule': 'rssi_dbm = -abs(原始数值)',
        'orientation_assumption': '按家具图约定将表格 x 增大方向解释为向东；若现场方向相反，需镜像后重跑',
    })
    return pts, furniture, quality


def quality_report(pts: list[MeasuredPoint], furniture: list[dict]) -> dict:
    vals = np.array([p.rssi for p in pts], dtype=float)
    raw = np.array([p.raw_rssi for p in pts], dtype=float)
    xs = sorted({p.x_data for p in pts})
    ys = sorted({p.y_data for p in pts})
    have = {(p.x_data, p.y_data) for p in pts}
    missing = [[x, y] for y in ys for x in xs if (x, y) not in have]
    xy = np.array([[p.x, p.y] for p in pts])
    if len(xy) > 1:
        d = np.sqrt(((xy[:, None] - xy[None, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        min_spacing = float(d.min())
    else:
        min_spacing = 0.0
    near = []
    for p in pts:
        df = min(math.hypot(p.x - f['x'], p.y - f['y']) for f in furniture)
        if df <= 1.0:
            near.append(p.idx)
    return {
        'n_points': len(pts),
        'nominal_cells': len(xs) * len(ys),
        'missing_cells': missing,
        'n_missing': len(missing),
        'x_values_data': xs,
        'y_values_data': ys,
        'rssi_min': float(vals.min()),
        'rssi_max': float(vals.max()),
        'rssi_mean': round(float(vals.mean()), 3),
        'rssi_median': round(float(np.median(vals)), 3),
        'raw_positive_count': int((raw > 0).sum()),
        'raw_negative_count': int((raw < 0).sum()),
        'br04_too_strong': [p.idx for p in pts if p.rssi > -40.0],
        'br04_too_weak': [p.idx for p in pts if p.rssi < -70.0],
        'n_br04_normal': int(sum(-70.0 <= p.rssi <= -40.0 for p in pts)),
        'outside_room_idx': [p.idx for p in pts if not p.inside_room],
        'min_neighbor_spacing_m': round(min_spacing, 3),
        'near_furniture_1m_idx': near,
        'furniture_count': len(furniture),
        'table_count': sum(f['kind'] == 'table' for f in furniture),
        'chair_count': sum(f['kind'] == 'chair' for f in furniture),
    }


def write_normalized(pts: list[MeasuredPoint], furniture: list[dict], quality: dict,
                     out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fields = ['idx', 'x_data', 'y_data', 'raw_rssi', 'rssi_dbm', 'x_room', 'y_room',
              'inside_room', 'br04_status', 'sign_corrected']
    with open(os.path.join(out_dir, 'measured.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in pts:
            w.writerow(p.to_row())
    with open(os.path.join(out_dir, 'furniture.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        fields_f = ['name', 'kind', 'x', 'y', 'radius', 'source', 'position_rule']
        w = csv.DictWriter(f, fieldnames=fields_f)
        w.writeheader()
        for item in furniture:
            row = dict(item)
            row['source'] = json.dumps(row['source'], ensure_ascii=False)
            w.writerow(row)
    with open(os.path.join(out_dir, 'furniture.json'), 'w', encoding='utf-8') as f:
        json.dump(furniture, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, 'data_quality.json'), 'w', encoding='utf-8') as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)


def furniture_mask(furniture: list[dict], grid: Grid = DEFAULT_GRID,
                   room: np.ndarray | None = None) -> np.ndarray:
    X, Y = grid.centers()
    mask = np.zeros(grid.shape, dtype=bool)
    for f in furniture:
        mask |= (X - f['x']) ** 2 + (Y - f['y']) ** 2 <= f['radius'] ** 2
    if room is not None:
        mask &= room > 0.5
    return mask.astype(np.float32)


def furniture_distance(furniture: list[dict], grid: Grid = DEFAULT_GRID) -> np.ndarray:
    X, Y = grid.centers()
    out = np.full(grid.shape, np.inf, dtype=float)
    for f in furniture:
        d = np.maximum(np.hypot(X - f['x'], Y - f['y']) - f['radius'], 0.0)
        out = np.minimum(out, d)
    return out


def point_furniture_distance(points: list[MeasuredPoint], furniture: list[dict]) -> np.ndarray:
    return np.array([
        min(max(math.hypot(p.x - f['x'], p.y - f['y']) - f['radius'], 0.0)
            for f in furniture)
        for p in points
    ], dtype=float)


def _valid_center(room: dict, grid: Grid, x: float, y: float) -> bool:
    j, i = grid.xy_to_idx(x, y)
    return 0 <= j < grid.ny and 0 <= i < grid.nx and bool(room['mask'][j, i])


def random_furniture(rng, room: dict, grid: Grid) -> list[dict]:
    """生成两组“1 桌 + 3 椅”，布局形式与真实场景一致、位置随机。"""
    X, Y = grid.centers()
    js, iss = np.nonzero(room['mask'])
    candidates = np.stack([X[js, iss], Y[js, iss]], axis=1)
    ax, ay = room['ap']
    furniture: list[dict] = []
    for group in range(2):
        table = None
        for _ in range(100):
            x, y = candidates[int(rng.integers(len(candidates)))]
            if math.hypot(x - ax, y - ay) < 0.8:
                continue
            if all(math.hypot(x - f['x'], y - f['y']) > 2.0 for f in furniture):
                table = (float(x), float(y))
                break
        if table is None:
            x, y = candidates[int(rng.integers(len(candidates)))]
            table = (float(x), float(y))
        furniture.append({'name': f'sim_table_{group}', 'kind': 'table',
                          'x': table[0], 'y': table[1], 'radius': 0.4})
        base_angle = rng.uniform(0, 2 * np.pi)
        for k in range(3):
            placed = False
            for _ in range(50):
                ang = base_angle + 2 * np.pi * k / 3 + rng.normal(0, 0.35)
                rad = rng.uniform(0.7, 1.15)
                x = table[0] + rad * math.cos(ang)
                y = table[1] + rad * math.sin(ang)
                if not _valid_center(room, grid, x, y):
                    continue
                if math.hypot(x - ax, y - ay) < 0.55:
                    continue
                furniture.append({'name': f'sim_chair_{group}_{k}', 'kind': 'chair',
                                  'x': float(x), 'y': float(y), 'radius': 0.4})
                placed = True
                break
            if not placed:
                x, y = candidates[int(rng.integers(len(candidates)))]
                furniture.append({'name': f'sim_chair_{group}_{k}', 'kind': 'chair',
                                  'x': float(x), 'y': float(y), 'radius': 0.4})
    return furniture


def _segment_circle_hit(X, Y, ax: float, ay: float, f: dict) -> np.ndarray:
    """AP 到格点的线段是否与家具圆相交。"""
    dx, dy = X - ax, Y - ay
    den = np.maximum(dx * dx + dy * dy, 1e-12)
    t = ((f['x'] - ax) * dx + (f['y'] - ay) * dy) / den
    tc = np.clip(t, 0.0, 1.0)
    px = ax + tc * dx
    py = ay + tc * dy
    dist2 = (px - f['x']) ** 2 + (py - f['y']) ** 2
    return (dist2 <= f['radius'] ** 2) & (t >= 0.0) & (t <= 1.0)


def simulate_rssi(room: dict, furniture: list[dict], rng, grid: Grid) -> np.ndarray:
    """对数路损 + 结构障碍 + 家具遮挡 + 相关阴影 + 独立噪声。"""
    X, Y = grid.centers()
    mask = np.asarray(room['mask'], dtype=bool)
    ax, ay = room['ap']
    d = np.hypot(X - ax, Y - ay)
    pl0 = rng.uniform(22.0, 46.0)
    n = rng.uniform(1.8, 3.2)
    path_loss = pl0 + 10.0 * n * np.log10(np.maximum(d, 0.5))

    attenuation = np.zeros(X.shape, dtype=float)
    for o in room['obstacles']:
        attenuation += rng.uniform(4.0, 14.0) * _crossing_mask(X, Y, ax, ay, o)
    for f in furniture:
        coef = rng.uniform(1.0, 4.0) if f['kind'] == 'table' else rng.uniform(0.4, 2.0)
        attenuation += coef * _segment_circle_hit(X, Y, ax, ay, f)

    sigma_cells = rng.uniform(1.0, 4.0)
    target_std = rng.uniform(1.5, 4.5)
    smooth = gaussian_filter(rng.standard_normal(X.shape), sigma=sigma_cells,
                             mode='nearest')
    s = float(smooth.std())
    shadow = smooth / s * target_std if s > 1e-9 else np.zeros_like(smooth)
    mag = rng.uniform(0.3, 1.2, size=X.shape)
    noise = mag * np.where(rng.random(X.shape) < 0.5, -1.0, 1.0)
    rssi = np.clip(-path_loss - attenuation + shadow + noise, RSSI_MIN, RSSI_MAX)
    rssi[~mask] = np.nan
    return rssi


def build_dataset(n_samples: int, seed: int, out_path: str, grid: Grid = DEFAULT_GRID,
                  obs_mode: str = 'mixed', obs_range=(5, 30), verbose=True) -> str:
    from .simulate import random_room

    rng = np.random.default_rng(seed)
    shape = (n_samples,) + grid.shape
    obs_value = np.zeros(shape, dtype=np.float32)
    obs_mask = np.zeros(shape, dtype=np.float32)
    room_arr = np.zeros(shape, dtype=np.float32)
    target = np.zeros(shape, dtype=np.float32)
    furn_arr = np.zeros(shape, dtype=np.float32)
    room_wh = np.zeros((n_samples, 2), dtype=np.float32)
    for k in range(n_samples):
        room = random_room(rng, grid)
        furn = random_furniture(rng, room, grid)
        n_obs = int(rng.integers(obs_range[0], obs_range[1] + 1))
        rssi = simulate_rssi(room, furn, rng, grid)
        mask = room['mask']
        tgt = np.where(mask, rssi, 0.0).astype(np.float32)
        idx = _pick_obs(rng, grid, mask, n_obs, mode=obs_mode)
        jj, ii = idx[:, 0], idx[:, 1]
        obs_value[k][jj, ii] = tgt[jj, ii]
        obs_mask[k][jj, ii] = 1.0
        room_arr[k] = mask.astype(np.float32)
        target[k] = tgt
        furn_arr[k] = furniture_mask(furn, grid, room_arr[k])
        room_wh[k] = room['room_wh']
        if verbose and (k + 1) % 200 == 0:
            print(f'  家具仿真 {obs_mode}: {k + 1}/{n_samples}', flush=True)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path, obs_value=obs_value, obs_mask=obs_mask,
                        room=room_arr, target=target, furniture=furn_arr,
                        room_wh=room_wh)
    if verbose:
        print(f'  保存 {out_path}', flush=True)
    return out_path


def reobserve_dataset(src_path: str, dst_path: str, seed: int,
                      grid: Grid = DEFAULT_GRID, verbose=True) -> str:
    """逐元素保留户型、家具和真值，只把观测改为散落布点。"""
    d = np.load(src_path)
    n = d['room'].shape[0]
    rng = np.random.default_rng(seed)
    obs_value = np.zeros_like(d['obs_value'])
    obs_mask = np.zeros_like(d['obs_mask'])
    for k in range(n):
        mask = d['room'][k] > 0.5
        n_obs = int(d['obs_mask'][k].sum())
        idx = _pick_obs(rng, grid, mask, n_obs, mode='scatter')
        jj, ii = idx[:, 0], idx[:, 1]
        obs_value[k][jj, ii] = d['target'][k][jj, ii]
        obs_mask[k][jj, ii] = 1.0
        if verbose and (k + 1) % 400 == 0:
            print(f'  散落重抽: {k + 1}/{n}', flush=True)
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    np.savez_compressed(dst_path, obs_value=obs_value, obs_mask=obs_mask,
                        room=d['room'], target=d['target'], furniture=d['furniture'],
                        room_wh=d['room_wh'])
    if verbose:
        print(f'  保存 {dst_path}', flush=True)
    return dst_path


def generate_datasets(work_dir: str, n_train=1200, n_val=200, n_test=200,
                      grid: Grid = DEFAULT_GRID) -> dict:
    mixed_dir = os.path.join(work_dir, 'sim')
    scatter_dir = os.path.join(work_dir, 'sim_scatter')
    paths = {'mixed': {}, 'scatter': {}}
    for split, n in (('train', n_train), ('val', n_val), ('test', n_test)):
        p = build_dataset(n, SIM_SEEDS[split], os.path.join(mixed_dir, split + '.npz'),
                          grid=grid, obs_mode='mixed')
        paths['mixed'][split] = p
        ps = reobserve_dataset(p, os.path.join(scatter_dir, split + '.npz'),
                               SCATTER_SEEDS[split], grid=grid)
        paths['scatter'][split] = ps
    return paths
