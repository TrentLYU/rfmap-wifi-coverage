"""评估：MAE / RMSE(dB) / 推理时间 / 与简单插值的对比 / 跨场景泛化。

公平性约定（任务书要求）
------------------------
1. 三个网络与三条插值基线在**每一次评估中使用完全相同的输入点集合**；
2. 隐藏实测点从不进入输入通道，也不参与任何训练或早停；
3. 误差只在隐藏实测点上统计，单位为 dB；
4. 同一张参考地图的重复记录不会被拆到不同集合（本项目只有一张，故不适用，
   但划分函数仍然按 map_id 组织，便于以后扩展）。

两类结果分开报告
----------------
* ``sim``：仿真测试集（随机户型、随机 AP）—— 衡量**跨场景 / 跨户型**能力；
* ``real``：真实 49 点留出 —— 衡量**同一房间内的稀疏重建**能力。

两者结论强度不同，报告中不得混为一谈。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import data as D
from . import geometry as G
from .grid import DEFAULT as DEFAULT_GRID
from .models import (BASELINE_ZOO, BASELINE_LABELS, MODEL_LABELS,
                     Normalizer, build_input_tensor, idw_predict, prepare_prior,
                     log_distance_trend, nearest_obs_distance, obs_density,
                     build_input_tensor_v5)
from .train import load_trained, masked_mse


# ==========================================================================
# 固定划分协议（写死，可复现）
# ==========================================================================
PROTOCOLS: dict[str, list[tuple[float, float]]] = {
    # 角块外推：取 x 小 / y 大的一角，全部隐藏点都远离其余观测点
    'corner': [(0.0, 8.0), (0.0, 10.0), (0.0, 12.0),
               (2.0, 8.0), (2.0, 10.0), (2.0, 12.0)],
    # 中心块插值：房间中央 3x3，四周仍有观测点
    'center': [(float(x), float(y)) for x in (4, 6, 8) for y in (4, 6, 8)],
}

PROTOCOL_LABELS = {
    'corner': '角块留出（外推）',
    'center': '中心块留出（插值）',
}


def split_points(pts, protocol: str):
    """按协议把 49 点拆成 (输入点, 隐藏点)。"""
    hidden_keys = {(round(x, 3), round(y, 3)) for x, y in PROTOCOLS[protocol]}
    inp, hid = [], []
    for p in pts:
        key = (round(p.x_data, 3), round(p.y_data, 3))
        (hid if key in hidden_keys else inp).append(p)
    return inp, hid


def build_obs(points, grid):
    """把测点铺到网格上，返回 (obs_value, obs_mask)。"""
    obs_v = np.zeros(grid.shape, dtype=np.float32)
    obs_m = np.zeros(grid.shape, dtype=np.float32)
    for p in points:
        j, i = grid.xy_to_idx(p.x, p.y)
        # 越界或落在柱体内的测点直接跳过（defensive）
        if not (0 <= j < grid.ny and 0 <= i < grid.nx):
            continue
        obs_v[j, i] = p.rssi
        obs_m[j, i] = 1.0
    return obs_v, obs_m


def bilinear_sample(field: np.ndarray, x: float, y: float, grid) -> float:
    """在预测网格上做双线性采样（评估时比取最近格点更公平）。"""
    fx = (x - grid.x0) / grid.cell - 0.5
    fy = (y - grid.y0) / grid.cell - 0.5
    i0 = int(np.floor(fx))
    j0 = int(np.floor(fy))
    tx = fx - i0
    ty = fy - j0

    def val(j, i):
        if 0 <= j < grid.ny and 0 <= i < grid.nx:
            v = field[j, i]
            return v if np.isfinite(v) else np.nan
        return np.nan

    vs = []
    ws = []
    for dj, wy in ((0, 1 - ty), (1, ty)):
        for di, wx in ((0, 1 - tx), (1, tx)):
            v = val(j0 + dj, i0 + di)
            if np.isfinite(v):
                vs.append(v * wx * wy)
                ws.append(wx * wy)
    if not ws or sum(ws) <= 0:
        v = val(j0, i0)
        return float(v) if np.isfinite(v) else float('nan')
    return float(sum(vs) / sum(ws))


# ==========================================================================
# 指标
# ==========================================================================
def metrics(preds: list[float], truths: list[float]) -> dict:
    a = np.asarray(preds, dtype=float)
    b = np.asarray(truths, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) == 0:
        return {'mae': None, 'rmse': None, 'max_abs': None, 'n': 0}
    err = a - b
    return {
        'mae': round(float(np.abs(err).mean()), 3),
        'rmse': round(float(np.sqrt((err ** 2).mean())), 3),
        'max_abs': round(float(np.abs(err).max()), 3),
        'bias': round(float(err.mean()), 3),
        'n': int(len(a)),
    }


# ==========================================================================
# 神经网络推理
# ==========================================================================
def net_input_mode(meta) -> str:
    """从 meta 判定模型的输入模式（'base' / 'prior' / 'multiprior'）。"""
    return meta.get('input_mode') or ('prior' if meta.get('has_prior') else 'base')


def _net_inputs(mode, obs_v, obs_m, room, grid, norm, prior=None):
    """按输入模式构造网络输入，返回 ``(x, prior_dbm_or_None)``。

    这个函数是**唯一**的输入构造入口：推理（:func:`_net_predict_once`）与
    实测微调（:func:`finetune_real`）都必须走它。早期版本两处各写了一份，
    结果方案五（7 通道）在微调时仍按 3 通道构造，直接报
    "expected input to have 7 channels, but got 4" —— 集中到一处可以根除这类问题。
    """
    X, Y = grid.centers()
    if mode == 'multiprior':
        js, iss = np.nonzero(obs_m > 0.5)
        pts = np.stack([X[js, iss], Y[js, iss]], axis=1)
        vals = obs_v[js, iss]
        tr = log_distance_trend(pts, vals, X, Y, room > 0.5)
        di = nearest_obs_distance(obs_m, grid.cell)
        de = obs_density(obs_m)
        if prior is None:
            prior = idw_predict(pts, vals, X, Y, room > 0.5)
        return build_input_tensor_v5(obs_v, obs_m, room, norm, tr, di, de), prior
    if mode == 'prior':
        if prior is None:
            js, iss = np.nonzero(obs_m > 0.5)
            pts = np.stack([X[js, iss], Y[js, iss]], axis=1)
            prior = idw_predict(pts, obs_v[js, iss], X, Y, room > 0.5)
        return build_input_tensor(obs_v, obs_m, room, norm), prior
    return build_input_tensor(obs_v, obs_m, room, norm), None


def _forward_net(model, mode, x, prior, norm, room, grid, device):
    """喂给模型并解码回 dBm 全图。"""
    xt = torch.from_numpy(x[None]).to(device)
    if mode in ('prior', 'multiprior'):
        pt = torch.from_numpy(prepare_prior(prior, room, norm)[None, None]).to(device)
        with torch.no_grad():
            out = model(xt, pt)
    else:
        with torch.no_grad():
            out = model(xt)
    pred = norm.decode(out[0, 0].detach().cpu().numpy()).astype(np.float64)
    return np.where(room > 0.5, pred, np.nan)


def _net_predict_once(model, mode, norm, obs_v, obs_m, room, grid, device,
                      prior):
    """单次前向（不含 TTA）。返回 dBm 全图，房间外 NaN。"""
    x, pr = _net_inputs(mode, obs_v, obs_m, room, grid, norm, prior)
    return _forward_net(model, mode, x, pr, norm, room, grid, device)


# TTA 用的四种几何变体。翻转是自逆运算，所以正向变换和逆变换用同一个函数。
_TTA_VARIANTS = (
    ('id', lambda a: a),
    ('ud', np.flipud),
    ('lr', np.fliplr),
    ('rot180', lambda a: np.flipud(np.fliplr(a))),
)


def nn_predict_grid(model, meta, norm, obs_v, obs_m, room, grid, device='cpu',
                    prior=None, tta: bool = False):
    """把观测铺成输入张量并推理，返回 dBm 全图（房间外 NaN）。

    Parameters
    ----------
    tta : bool
        启用测试时增强（Test-Time Augmentation）：对输入做恒等 / 上下翻转 /
        左右翻转 / 旋转 180° 四种几何变体各推理一次，把结果逆变换后取平均。

        **为什么在补全任务上可以这么做**：模型看到的是"户型掩码 + 稀疏观测"，
        把整幅图翻转等价于换了一间镜像户型 —— 只要训练分布对镜像是对称的
        （随机户型生成确实是对称的），翻转后的预测就应该同样有效，平均能
        压低单次前向的随机波动。但它并非无条件成立，本项目在报告里做了
        开/关消融，以实测数据说话。
    """
    mode = net_input_mode(meta)

    if not tta:
        return _net_predict_once(model, mode, norm, obs_v, obs_m, room, grid,
                                 device, prior)

    acc = np.zeros(grid.shape, dtype=np.float64)
    n = 0
    for _, f in _TTA_VARIANTS:
        ov = f(obs_v)
        om = f(obs_m)
        rm = f(room)
        pr = f(prior) if prior is not None else None
        p = _net_predict_once(model, mode, norm, ov, om, rm, grid, device, pr)
        p = np.nan_to_num(p, nan=0.0)
        acc += f(p)
        n += 1
    out = acc / max(n, 1)
    return np.where(room > 0.5, out, np.nan)


def time_inference(fn, repeats: int = 30) -> float:
    fn()  # 预热
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return round((time.perf_counter() - t0) / repeats * 1000, 2)  # ms


# ==========================================================================
# 实测微调（任务书要求的第二步）
# ==========================================================================
def finetune_real(model, meta, norm, pts, grid, room, epochs: int = 300,
                  lr: float = 3e-4, seed: int = 0, device: str = 'cpu',
                  min_obs_ratio: float = 0.5):
    """用真实测点做**自监督微调**，并返回一份新的模型副本。

    每轮从输入点里随机抽一部分当"给定观测"，剩下的当监督目标 —— 也就是
    让模型反复练习"用这个房间的一小撮点去补全同一房间的其它点"。
    抽到的点既当输入又当答案是被禁止的，这里的输入集合与监督集合始终不相交。

    Parameters
    ----------
    pts : list[Point]
        只允许传**协议内的输入点**；隐藏点绝不能传进来，否则评估就泄漏了。
    """
    import copy

    m = copy.deepcopy(model)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    mode = net_input_mode(meta)

    n = len(pts)
    if n < 4:
        return model

    xs = np.array([p.x for p in pts])
    ys = np.array([p.y for p in pts])
    vs = np.array([p.rssi for p in pts], dtype=np.float32)
    ij = [grid.xy_to_idx(x, y) for x, y in zip(xs, ys)]
    jjs = np.array([t[0] for t in ij])
    iis = np.array([t[1] for t in ij])

    for _ in range(epochs):
        k = int(rng.integers(max(3, int(n * min_obs_ratio)), max(4, n - 2)))
        sel = rng.choice(n, size=min(k, n - 2), replace=False)
        sup = np.setdiff1d(np.arange(n), sel)

        obs_v = np.zeros(grid.shape, np.float32)
        obs_m = np.zeros(grid.shape, np.float32)
        obs_v[jjs[sel], iis[sel]] = vs[sel]
        obs_m[jjs[sel], iis[sel]] = 1.0

        sup_m = np.zeros(grid.shape, np.float32)
        sup_m[jjs[sup], iis[sup]] = 1.0
        y_val = np.zeros(grid.shape, np.float32)
        y_val[jjs, iis] = vs

        x, pr = _net_inputs(mode, obs_v, obs_m, room, grid, norm)
        xt = torch.from_numpy(x[None]).to(device)
        yt = torch.from_numpy((norm.encode(y_val) * sup_m)[None, None]).to(device)
        mt = torch.from_numpy(sup_m[None, None]).to(device)

        if mode in ('prior', 'multiprior'):
            pt = torch.from_numpy(prepare_prior(pr, room, norm)[None, None]).to(device)
            pred = m(xt, pt)
        else:
            pred = m(xt)

        loss = masked_mse(pred, yt, mt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()

    m.eval()
    return m


# ==========================================================================
# 真实数据评估
# ==========================================================================
def eval_real(root: str = '.', device: str = 'cpu',
              model_names=('ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr'),
              baselines=('nearest', 'idw', 'rbf', 'gp'),
              repeats: int = 20, seed: int = 123,
              finetune_epochs: int = 300,
              out_path: str | None = None, tta: bool = False) -> dict:
    grid = DEFAULT_GRID
    room = grid.room_mask().astype(np.float32)
    X, Y = grid.centers()
    pts = D.load(root=root)
    rng = np.random.default_rng(seed)

    result = {'protocols': {}, 'sweep': {}, 'baseline_labels': BASELINE_LABELS,
              'model_labels': MODEL_LABELS, 'n_points_total': len(pts)}

    loaded = {}
    for nm in model_names:
        try:
            loaded[nm] = load_trained(nm, root=root, device=device)
        except FileNotFoundError:
            print(f'  [跳过] 未找到模型 {nm}', flush=True)

    # ---------- 固定协议 ----------
    for proto in ('corner', 'center'):
        inp, hid = split_points(pts, proto)
        obs_v, obs_m = build_obs(inp, grid)
        truth = [p.rssi for p in hid]
        entry = {
            'label': PROTOCOL_LABELS[proto],
            'n_input': len(inp), 'n_hidden': len(hid),
            'hidden_points': [[p.x_data, p.y_data, p.rssi] for p in hid],
            'methods': {},
        }

        for nm, (model, meta, norm) in loaded.items():
            t_ms = time_inference(lambda m=model, mt=meta, n=norm:
                                 nn_predict_grid(m, mt, n, obs_v, obs_m, room,
                                                 grid, device))
            pred = nn_predict_grid(model, meta, norm, obs_v, obs_m, room, grid, device)
            preds = [bilinear_sample(pred, p.x, p.y, grid) for p in hid]
            m = metrics(preds, truth)
            m['infer_ms'] = t_ms
            m['per_point'] = [[p.x_data, p.y_data, p.rssi,
                               round(float(v), 3) if v == v else None]
                              for p, v in zip(hid, preds)]
            entry['methods'][nm] = m

            # ---- 测试时增强（TTA，只做实时推理、不训练） ----
            if tta:
                t_ms_tta = time_inference(lambda m=model, mt=meta, n=norm:
                                          nn_predict_grid(m, mt, n, obs_v, obs_m,
                                                          room, grid, device,
                                                          tta=True), repeats=10)
                pred_t = nn_predict_grid(model, meta, norm, obs_v, obs_m, room,
                                         grid, device, tta=True)
                preds_t = [bilinear_sample(pred_t, p.x, p.y, grid) for p in hid]
                m_t = metrics(preds_t, truth)
                m_t['infer_ms'] = t_ms_tta
                m_t['per_point'] = [[p.x_data, p.y_data, p.rssi,
                                     round(float(v), 3) if v == v else None]
                                    for p, v in zip(hid, preds_t)]
                entry['methods'][nm + '_tta'] = m_t

            # ---- 实测微调（只用输入点，隐藏点不参与） ----
            ft = finetune_real(model, meta, norm, inp, grid, room,
                               epochs=finetune_epochs, device=device, seed=seed)
            pred_ft = nn_predict_grid(ft, meta, norm, obs_v, obs_m, room, grid, device)
            preds_ft = [bilinear_sample(pred_ft, p.x, p.y, grid) for p in hid]
            m_ft = metrics(preds_ft, truth)
            m_ft['infer_ms'] = time_inference(
                lambda mm=ft, mt=meta, n=norm: nn_predict_grid(mm, mt, n, obs_v,
                                                               obs_m, room, grid, device))
            m_ft['per_point'] = [[p.x_data, p.y_data, p.rssi,
                                  round(float(v), 3) if v == v else None]
                                 for p, v in zip(hid, preds_ft)]
            entry['methods'][nm + '_ft'] = m_ft

        for bn in baselines:
            fn = BASELINE_ZOO[bn]
            js, iss = np.nonzero(obs_m > 0.5)
            pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
            pv = obs_v[js, iss]
            pred = fn(pxy, pv, X, Y, room > 0.5)
            preds = [bilinear_sample(pred, p.x, p.y, grid) for p in hid]
            m = metrics(preds, truth)
            m['infer_ms'] = time_inference(
                lambda f=fn, a=pxy, b=pv: f(a, b, X, Y, room > 0.5))
            m['per_point'] = [[p.x_data, p.y_data, p.rssi,
                               round(float(v), 3) if v == v else None]
                              for p, v in zip(hid, preds)]
            entry['methods'][bn] = m

        result['protocols'][proto] = entry
        print(f'  [{proto}] 输入 {len(inp)} 隐藏 {len(hid)}', flush=True)

    # ---------- 输入点数扫描 ----------
    all_xy = np.array([[p.x_data, p.y_data] for p in pts])
    for K in (5, 10, 15, 20, 30):
        acc = {k: [] for k in list(loaded) + list(baselines)}
        for _ in range(repeats):
            sel = rng.choice(len(pts), size=K, replace=False)
            mask = np.zeros(len(pts), bool)
            mask[sel] = True
            inp = [p for p, m in zip(pts, mask) if m]
            hid = [p for p, m in zip(pts, mask) if not m]
            if not hid:
                continue
            obs_v, obs_m = build_obs(inp, grid)
            truth = [p.rssi for p in hid]

            for nm, (model, meta, norm) in loaded.items():
                pred = nn_predict_grid(model, meta, norm, obs_v, obs_m, room, grid, device)
                preds = [bilinear_sample(pred, p.x, p.y, grid) for p in hid]
                acc[nm].append(metrics(preds, truth)['rmse'])

            js, iss = np.nonzero(obs_m > 0.5)
            pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
            pv = obs_v[js, iss]
            for bn in baselines:
                pred = BASELINE_ZOO[bn](pxy, pv, X, Y, room > 0.5)
                preds = [bilinear_sample(pred, p.x, p.y, grid) for p in hid]
                acc[bn].append(metrics(preds, truth)['rmse'])

        result['sweep'][str(K)] = {
            k: {'rmse_mean': round(float(np.nanmean(v)), 3),
                'rmse_std': round(float(np.nanstd(v)), 3),
                'runs': len(v)}
            for k, v in acc.items() if v
        }
        print(f'  [sweep K={K}] ' +
              '  '.join(f'{k}={np.nanmean(v):.2f}' for k, v in acc.items() if v),
              flush=True)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print('  写出 ->', out_path, flush=True)
    return result


# ==========================================================================
# 仿真（跨户型）评估
# ==========================================================================
def eval_sim(root: str = '.', device: str = 'cpu',
             model_names=('ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr'),
             files=None, out_path: str | None = None,
             max_samples: int | None = 200, tta: bool = False) -> dict:
    """在仿真测试集上评估：**跨户型、跨 AP** 的补全能力。

    两个测试集都用，因为方案四的结论必须"两头看"：

    * ``mixed``  —— 常规布点测试集（与方案一/二/三训练时同分布）；
    * ``scatter``—— 散落布点测试集。它与 ``mixed`` **逐元素共享同一批户型和真值**，
      只有观测点位置不同，因此两者的差异可以干净地归因到"布点方式"。

    只看一个会得出片面结论：只用 mixed 评，会低估散落训练的价值；
    只用 scatter 评，又会掩盖"专精散落布点是否牺牲了常规布点性能"。
    """
    grid = DEFAULT_GRID
    X, Y = grid.centers()
    if files is None:
        files = [('data/sim/test.npz', 'mixed'),
                 ('data/sim_scatter/test.npz', 'scatter')]

    loaded = {}
    for nm in model_names:
        try:
            loaded[nm] = load_trained(nm, root=root, device=device)
        except FileNotFoundError:
            print(f'  [跳过] 未找到模型 {nm}', flush=True)

    result = {'model_labels': MODEL_LABELS, 'baseline_labels': BASELINE_LABELS,
              'tta_enabled': bool(tta), 'testsets': {}}

    for rel, tag in files:
        d = np.load(os.path.join(root, rel))
        total = d['obs_value'].shape[0]
        n = total if max_samples is None else min(max_samples, total)
        room_all = d['room'].astype(np.float32)

        names = list(loaded) + ['nearest', 'idw', 'rbf', 'gp']
        if tta:
            names += [k + '_tta' for k in loaded]
        accum = {k: [] for k in names}
        obs_counts = []

        for k in range(n):
            obs_v = d['obs_value'][k].astype(np.float32)
            obs_m = d['obs_mask'][k].astype(np.float32)
            room = room_all[k]
            target = d['target'][k].astype(np.float64)
            obs_counts.append(int(obs_m.sum()))

            for nm, (model, meta, norm) in loaded.items():
                for suffix, use_tta in (('', False), ('_tta', True)):
                    if suffix and not tta:
                        continue
                    pred = nn_predict_grid(model, meta, norm, obs_v, obs_m, room,
                                           grid, device, tta=use_tta)
                    m = (room > 0.5) & (obs_m < 0.5) & np.isfinite(pred)
                    accum[nm + suffix].append(
                        float(np.sqrt(((pred[m] - target[m]) ** 2).mean())))

            js, iss = np.nonzero(obs_m > 0.5)
            pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
            pv = obs_v[js, iss]
            for bn in ('nearest', 'idw', 'rbf', 'gp'):
                pred = BASELINE_ZOO[bn](pxy, pv, X, Y, room > 0.5)
                m = (room > 0.5) & (obs_m < 0.5) & np.isfinite(pred)
                accum[bn].append(
                    float(np.sqrt(((pred[m] - target[m]) ** 2).mean())))

            if (k + 1) % 50 == 0:
                print(f'  [{tag}] {k + 1}/{n}', flush=True)

        entry = {'source': rel, 'n_samples': int(n),
                 'obs_count_mean': round(float(np.mean(obs_counts)), 2),
                 'obs_count_range': [int(min(obs_counts)), int(max(obs_counts))],
                 'methods': {}}
        for kname, v in accum.items():
            entry['methods'][kname] = {
                'rmse_mean': round(float(np.mean(v)), 3),
                'rmse_median': round(float(np.median(v)), 3),
                'rmse_p90': round(float(np.percentile(v, 90)), 3),
                'runs': len(v),
            }
        result['testsets'][tag] = entry

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print('  写出 ->', out_path, flush=True)
    return result


def eval_stratified(root: str = '.', device: str = 'cpu',
                    model_names=('unet', 'unet_scatter', 'pgmpr'),
                    files=None, n_bins: int = 3, max_samples: int | None = 200,
                    out_path: str | None = None) -> dict:
    """按"样本自身的布点散落程度"分层，看各方案在哪一档上更强。

    动机：方案四（散落布点训练）在整体均值上并没有赢过方案二。但"均值平手"
    不等于"处处平手" —— 完全可能是"在偏规整的样本上略差、在最散落的样本上更好"，
    两者一平均就互相抵消了。只有把测试样本按散落度分层，才能看出这一点。

    散落度用**每个样本**未观测区内"到最近观测点距离"的 P95 来度量（单位 m）：
    数值越大，说明该样本的空白区越宽、外推压力越大。
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    grid = DEFAULT_GRID
    if files is None:
        files = [('data/sim_scatter/test.npz', 'scatter')]

    loaded = {}
    for nm in model_names:
        try:
            loaded[nm] = load_trained(nm, root=root, device=device)
        except FileNotFoundError:
            print(f'  [跳过] 未找到模型 {nm}', flush=True)

    result = {'n_bins': int(n_bins), 'testsets': {}}

    for rel, tag in files:
        d = np.load(os.path.join(root, rel))
        total = d['obs_value'].shape[0]
        n = total if max_samples is None else min(max_samples, total)
        room_all = d['room'].astype(np.float32)

        spread = np.zeros(n)
        per_model = {k: np.zeros(n) for k in loaded}

        for k in range(n):
            ov = d['obs_value'][k].astype(np.float32)
            om = d['obs_mask'][k].astype(np.float32)
            room = room_all[k]
            tgt = d['target'][k].astype(np.float64)
            unknown = (room > 0.5) & (om < 0.5)
            dist = distance_transform_edt(~(om > 0.5)) * grid.cell
            spread[k] = float(np.percentile(dist[unknown], 95)) if unknown.any() else 0.0

            for nm, (model, meta, norm) in loaded.items():
                pred = nn_predict_grid(model, meta, norm, ov, om, room, grid, device)
                m = unknown & np.isfinite(pred)
                per_model[nm][k] = float(np.sqrt(((pred[m] - tgt[m]) ** 2).mean()))

            if (k + 1) % 50 == 0:
                print(f'  [{tag}] {k + 1}/{n}', flush=True)

        order = np.argsort(spread)
        bins = [b for b in np.array_split(order, n_bins) if len(b)]
        labels = ['最规整', '中等', '最散落']
        entry = {'source': rel, 'n_samples': int(n), 'bins': []}
        for bi, idx in enumerate(bins):
            entry['bins'].append({
                'label': labels[bi] if bi < len(labels) else f'第{bi + 1}档',
                'spread_mean': round(float(spread[idx].mean()), 2),
                'spread_range': [round(float(spread[idx].min()), 2),
                                 round(float(spread[idx].max()), 2)],
                'n': int(len(idx)),
                'methods': {nm: round(float(per_model[nm][idx].mean()), 3)
                            for nm in loaded},
            })
        result['testsets'][tag] = entry
        print(f'  [{tag}] 分档完成：' + ' | '.join(
            f"{b['label']}(P95={b['spread_mean']}m) n={b['n']}" for b in entry['bins']),
            flush=True)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print('  写出 ->', out_path, flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description='RF Map 评估')
    ap.add_argument('--what', default='real,sim')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--root', default='.')
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--out-dir', default='outputs/rfmap-results')
    ap.add_argument('--tta', action='store_true',
                    help='额外输出测试时增强（TTA）的对照结果')
    args = ap.parse_args(argv)

    if 'real' in args.what:
        print('== 真实数据评估 ==', flush=True)
        eval_real(root=args.root, device=args.device, repeats=args.repeats,
                  tta=args.tta,
                  out_path=os.path.join(args.out_dir, 'metrics_real.json'))
    if 'sim' in args.what:
        print('== 仿真跨户型评估 ==', flush=True)
        eval_sim(root=args.root, device=args.device, tta=args.tta,
                 out_path=os.path.join(args.out_dir, 'metrics_sim.json'))
    if 'strat' in args.what:
        print('== 散落度分层评估 ==', flush=True)
        eval_stratified(root=args.root, device=args.device,
                        out_path=os.path.join(args.out_dir,
                                              'metrics_stratified.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
