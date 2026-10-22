"""`rfmap.simulate` 自检。

覆盖任务书要求的 6 条断言；用较小样本量以便快速反复运行。
运行：``.venv/bin/python -m rfmap.simulate_selfcheck``
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from .grid import Grid
from . import simulate as S


def _load(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def main() -> int:
    grid = Grid()
    tmp = tempfile.mkdtemp(prefix='simcheck_')
    ds_path = os.path.join(tmp, 'check.npz')

    S.build_dataset(60, seed=12345, out_path=ds_path, grid=grid, verbose=False)
    d = _load(ds_path)
    obs_value, obs_mask = d['obs_value'], d['obs_mask']
    room, target = d['room'], d['target']

    ok = True

    # 1) 观测点上的 obs_value 与 target 逐元素完全一致
    same = np.array_equal(obs_value[obs_mask == 1], target[obs_mask == 1])
    print('[1] 观测点 obs_value == target :', same)
    ok &= same

    # 2) 所有观测点都落在房间内
    inside = bool(np.all(room[obs_mask == 1] == 1))
    print('[2] 所有观测点 room==1        :', inside)
    ok &= inside

    # 3) 房间掩码占比落在 30%~80%
    frac = room.reshape(room.shape[0], -1).mean(axis=1)
    frac_ok = bool(frac.min() >= 0.30 and frac.max() <= 0.80)
    print('[3] 房间占比 min/max          : %.4f / %.4f  -> %s'
          % (frac.min(), frac.max(), frac_ok))
    ok &= frac_ok

    # 4) 房间内 target 值域在 [-95,-25]
    rng_ok = bool(target[room == 1].min() >= -95.0
                  and target[room == 1].max() <= -25.0)
    print('[4] target 值域 min/max       : %.2f / %.2f  -> %s'
          % (target[room == 1].min(), target[room == 1].max(), rng_ok))
    ok &= rng_ok

    # 5) 同 seed 两次构建逐元素完全一致
    p1 = os.path.join(tmp, 'rep1.npz')
    p2 = os.path.join(tmp, 'rep2.npz')
    S.build_dataset(5, seed=777, out_path=p1, grid=grid, verbose=False)
    S.build_dataset(5, seed=777, out_path=p2, grid=grid, verbose=False)
    a, b = _load(p1), _load(p2)
    rep = all(np.array_equal(a[k], b[k]) for k in a)
    print('[5] 同 seed 两次结果一致       :', rep)
    ok &= rep

    # 6) target 最大值落在"距 AP 最近 15% 区域"的比例 > 50%
    rng = np.random.default_rng(999)
    X, Y = grid.centers()
    hit = total = 0
    for _ in range(100):
        room_g = S.random_room(rng, grid)
        rssi = S.simulate_rssi(room_g, rng, grid)
        mask = room_g['mask']
        ax, ay = room_g['ap']
        dist = np.hypot(X - ax, Y - ay)
        thr = np.quantile(dist[mask], 0.15)
        region = mask & (dist <= thr)
        vals = np.where(mask, rssi, -np.inf)
        jj, ii = np.unravel_index(int(np.argmax(vals)), vals.shape)
        hit += bool(region[jj, ii])
        total += 1
    ratio = hit / total
    ratio_ok = ratio > 0.5
    print('[6] 峰值落在近 AP 15%% 区域     : %.2f  (%d/%d)  -> %s'
          % (ratio, hit, total, ratio_ok))
    ok &= ratio_ok

    print('\nRESULT:', 'ALL PASS' if ok else 'FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
