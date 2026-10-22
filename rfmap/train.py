"""训练流程：仿真预训练 + 实测微调。

为什么必须"先仿真、后实测"
--------------------------
任务书明确要求区分训练 / 验证 / 测试**场景组**，并指出：

    若只有一张固定地图，反复遮挡只能用于程序调试，不能证明模型学会了未知地图补全。

本项目手里只有 **1 张真实参考地图**（49 点），因此：

1. **仿真预训练**在 1200 张随机户型 / 随机 AP 的地图上进行，
   模型见到的"户型"和"AP 位置"都是新的，这才构成"补全新地图"的能力来源；
2. **实测微调**只用真实 49 点中的**输入点**，隐藏点从不进入输入通道；
3. 报告里把两类结果分开陈述，不把"同房间稀疏重建"说成"跨房间泛化"。

损失与掩码
----------
所有损失都是 **masked MSE**：只在 ``room == 1`` 的格点上计算。
未观测格点的占位值 0 先乘观测掩码，绝不参与梯度——直接落实任务书
"未测位置只是占位值，不能被理解为零信号"。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from . import data as D
from . import geometry as G
from .grid import Grid, DEFAULT as DEFAULT_GRID
from .models import (MODEL_ZOO, MODEL_LABELS, Normalizer, build_model,
                     count_params, idw_predict, build_input_tensor, prepare_prior,
                     log_distance_trend, nearest_obs_distance, obs_density,
                     build_input_tensor_v5, V5_DIST_SCALE)

# 需要 IDW 先验通道的模型（方案三、方案五）
PRIOR_MODELS = ('resunet', 'pgmpr')
# 还需要 trend / 最近观测点距离 / 观测密度 三个额外通道的模型（方案五）
MULTIPRIOR_MODELS = ('pgmpr',)
# 观测加权损失的强度：权重 = 1 + DENSITY_WEIGHT * 观测密度（并归一到均值 1）
DENSITY_WEIGHT = 2.0


# ==========================================================================
# 数据
# ==========================================================================
def _precompute_idw(obs_value, obs_mask, grid, room, power=2.0, max_radius=6.0,
                    log_every=200):
    """为每个样本预算一张 IDW 先验图（方案三 / 方案五的引导通道）。"""
    X, Y = grid.centers()
    n = obs_value.shape[0]
    priors = np.empty((n, grid.ny, grid.nx), dtype=np.float32)
    for k in range(n):
        js, iss = np.nonzero(obs_mask[k] > 0.5)
        pts = np.stack([X[js, iss], Y[js, iss]], axis=1)
        vals = obs_value[k][js, iss]
        priors[k] = idw_predict(pts, vals, X, Y, room[k] > 0.5,
                                power=power, max_radius=max_radius)
        if log_every and (k + 1) % log_every == 0:
            print(f'    IDW 先验 {k + 1}/{n}', flush=True)
    return priors


def _precompute_extra(obs_value, obs_mask, grid, room, log_every=400):
    """方案五的 3 路附加先验图，返回 (N, 3, ny, nx) float32。

    ======  =========================================================
    通道    含义
    ======  =========================================================
    0       对数距离物理趋势（dBm，房间外 NaN）—— 全局衰减形状
    1       到最近观测点距离（m）—— 外推难度的显式度量
    2       观测密度（0~1）—— 区分孤立点与密集测量区
    ======  =========================================================

    第 0 路刻意保留 dBm 原始量纲，标准化统一放在
    :func:`rfmap.models.build_input_tensor_v5` 里做，避免两处口径不一致。
    """
    X, Y = grid.centers()
    n = obs_value.shape[0]
    extra = np.zeros((n, 3, grid.ny, grid.nx), dtype=np.float32)
    for k in range(n):
        m = room[k] > 0.5
        js, iss = np.nonzero(obs_mask[k] > 0.5)
        pts = np.stack([X[js, iss], Y[js, iss]], axis=1)
        vals = obs_value[k][js, iss]
        extra[k, 0] = log_distance_trend(pts, vals, X, Y, m)
        extra[k, 1] = nearest_obs_distance(obs_mask[k], grid.cell)
        extra[k, 2] = obs_density(obs_mask[k])
        if log_every and (k + 1) % log_every == 0:
            print(f'    附加先验 {k + 1}/{n}', flush=True)
    return extra


def _cache_file(cache_dir, src_path, tag):
    """先验图缓存路径。

    必须带上**父目录名**：``data/sim/train.npz`` 与 ``data/sim_scatter/train.npz``
    的文件名相同，只用 basename 会让方案四误用方案二算好的先验图，
    从而污染整个数据消融实验（这是个真实的坑，已在此规避）。
    """
    if not cache_dir:
        return None
    p = os.path.abspath(src_path)
    parent = os.path.basename(os.path.dirname(p))
    stem = os.path.splitext(os.path.basename(p))[0]
    return os.path.join(cache_dir, f'{parent}_{stem}_{tag}.npy')


class SimDataset(torch.utils.data.Dataset):
    """仿真数据集，支持多份 npz 拼接与三种输入模式。

    Parameters
    ----------
    npz_paths : str | list[str]
        一份或多份数据集（多份会按样本维拼接，用于方案五合并
        "常规布点 + 散落布点"两类训练数据）。
    mode : str
        ``'base'`` 3 通道（方案一 / 二 / 四）；
        ``'prior'`` 3 通道 + IDW 先验（方案三）；
        ``'multiprior'`` 6 通道 + IDW 先验 + 观测加权（方案五）。
    """

    def __init__(self, npz_paths, norm: Normalizer, grid: Grid,
                 mode: str = 'base', cache_dir: str | None = None):
        if isinstance(npz_paths, str):
            npz_paths = [npz_paths]
        self.paths = list(npz_paths)

        ov, om, rm, tg = [], [], [], []
        for p in self.paths:
            d = np.load(p)
            ov.append(d['obs_value'].astype(np.float32))
            om.append(d['obs_mask'].astype(np.float32))
            rm.append(d['room'].astype(np.float32))
            tg.append(d['target'].astype(np.float32))
        self.obs_value = np.concatenate(ov)
        self.obs_mask = np.concatenate(om)
        self.room = np.concatenate(rm)
        self.target = np.concatenate(tg)

        self.norm = norm
        self.grid = grid
        self.mode = mode
        self.prior = None
        self.extra = None
        self.weight = None

        if mode in ('prior', 'multiprior'):
            self.prior = self._gather('prior', lambda ov, om, rm: _precompute_idw(
                ov, om, grid, rm), cache_dir)
        if mode == 'multiprior':
            self.extra = self._gather('extra', lambda ov, om, rm: _precompute_extra(
                ov, om, grid, rm), cache_dir)
            # 观测加权：观测点附近权重更高（那里信息最可信），再归一到均值 1，
            # 使加权前后损失的绝对尺度可比，学习率不必跟着改。
            w = 1.0 + DENSITY_WEIGHT * self.extra[:, 2]
            m = (self.room > 0.5).astype(np.float32)
            denom = np.maximum((w * m).sum(axis=(1, 2), keepdims=True), 1e-6)
            self.weight = (w * m / (denom / np.maximum(m.sum(axis=(1, 2),
                                                              keepdims=True), 1.0))
                           ).astype(np.float32)

    def _gather(self, tag, fn, cache_dir):
        """按文件逐个取缓存，缺失才计算 —— 避免每次训练重算先验图。"""
        chunks = []
        for p in self.paths:
            cf = _cache_file(cache_dir, p, tag)
            source = None
            if cf and os.path.exists(cf):
                chunks.append(np.load(cf))
                continue
            if source is None:
                d = np.load(p)
                source = (d['obs_value'].astype(np.float32),
                          d['obs_mask'].astype(np.float32),
                          d['room'].astype(np.float32))
            print(f'  计算 {tag} 先验: {p}', flush=True)
            arr = fn(*source)
            chunks.append(arr)
            if cf:
                os.makedirs(os.path.dirname(cf), exist_ok=True)
                np.save(cf, arr)
        return np.concatenate(chunks)

    def __len__(self):
        return self.obs_value.shape[0]

    def __getitem__(self, k):
        if self.mode == 'multiprior':
            x = build_input_tensor_v5(self.obs_value[k], self.obs_mask[k],
                                      self.room[k], self.norm,
                                      self.extra[k, 0], self.extra[k, 1],
                                      self.extra[k, 2])
        else:
            x = build_input_tensor(self.obs_value[k], self.obs_mask[k],
                                   self.room[k], self.norm)
        y = (self.norm.encode(self.target[k]) * (self.room[k] > 0.5)).astype(np.float32)
        prior = (prepare_prior(self.prior[k], self.room[k], self.norm)[None]
                 if self.prior is not None
                 else np.zeros((1,) + self.grid.shape, np.float32))
        w = (self.weight[k][None] if self.weight is not None
             else np.ones((1,) + self.grid.shape, np.float32))
        return x, prior, y[None], self.room[k][None], w


# ==========================================================================
# 损失
# ==========================================================================
def masked_mse(pred, target, room):
    """只在房间有效区内计算 MSE（房间外与柱体内不计入）。"""
    m = (room > 0.5).float()
    denom = m.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * m).sum() / denom


def weighted_masked_mse(pred, target, room, weight):
    """观测密度加权的 masked MSE（方案五使用）。

    动机：观测点附近是全图唯一有"真值锚点"的地方，这里的误差更该被惩罚；
    而远离观测点的格点本来就不确定，若一视同仁，网络可能为了压低大片
    不确定区的误差，把靠近观测点的应得精度也一起牺牲掉。

    权重图由 :class:`SimDataset` 预先算好（``1 + 2 × 观测密度``）并归一到
    房间内均值为 1，因此损失的绝对尺度与普通 masked MSE 可比。
    """
    m = (room > 0.5).float()
    w = m * weight
    return (((pred - target) ** 2) * w).sum() / w.sum().clamp_min(1e-6)


# ==========================================================================
# 训练循环
# ==========================================================================
def _forward(model, name, x, prior):
    if name in PRIOR_MODELS:
        return model(x, prior)
    return model(x)


def _dataset_mode(model_name: str) -> str:
    if model_name in MULTIPRIOR_MODELS:
        return 'multiprior'
    if model_name in PRIOR_MODELS:
        return 'prior'
    return 'base'


def train_sim(model_name: str, epochs: int = 120, width: int = 24,
              batch_size: int = 16, lr: float = 2e-3, seed: int = 7,
              device: str = 'cpu', root: str = '.', patience: int = 30,
              out_dir: str | None = None, verbose: bool = True,
              data_dir: str | None = None, train_files=None,
              loss: str = 'auto') -> dict:
    """仿真预训练。返回训练摘要（含历史曲线与最佳验证损失）。

    Parameters
    ----------
    data_dir : str | None
        训练/验证数据所在目录（默认 ``data/sim``）。方案四改成散落布点数据集时
        只需切到 ``data/sim_scatter``，网络与超参一字不改 —— 保证"唯一变量是数据"。
    train_files : list[str] | None
        显式指定训练文件（可多份拼接，方案五用）。给定时优先于 ``data_dir``。
    loss : str
        ``'auto'`` 时方案五用观测加权损失、其余用普通 masked MSE；
        也可强制 ``'mse'`` / ``'weighted'`` 以便做损失消融。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    grid = DEFAULT_GRID
    mode = _dataset_mode(model_name)
    has_prior = mode in ('prior', 'multiprior')
    if loss == 'auto':
        loss = 'weighted' if mode == 'multiprior' else 'mse'
    cache_dir = os.path.join(root, 'data/sim/.cache')

    ddir = data_dir or os.path.join(root, 'data/sim')
    tr_paths = (train_files if train_files
                else [os.path.join(ddir, 'train.npz')])
    tr_paths = [p if os.path.isabs(p) else os.path.join(root, p) for p in tr_paths]
    va_path = os.path.join(ddir, 'val.npz')

    tr_raw = np.load(tr_paths[0])
    norm = Normalizer.fit(tr_raw['target'], tr_raw['room'])

    ds_tr = SimDataset(tr_paths, norm, grid, mode, cache_dir)
    ds_va = SimDataset(va_path, norm, grid, mode, cache_dir)

    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size,
                                        shuffle=True, num_workers=0, drop_last=True)
    ld_va = torch.utils.data.DataLoader(ds_va, batch_size=batch_size,
                                        shuffle=False, num_workers=0)

    model = build_model(model_name, width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    use_weighted = (loss == 'weighted')

    def _loss(pred, y, room, w):
        return (weighted_masked_mse(pred, y, room, w) if use_weighted
                else masked_mse(pred, y, room))

    best = float('inf')
    best_state = None
    best_epoch = -1
    hist = {'epoch': [], 'train': [], 'val': [], 'lr': []}
    bad = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        tl, tn = 0.0, 0
        for x, prior, y, room, w in ld_tr:
            x, prior, y, room, w = (x.to(device), prior.to(device), y.to(device),
                                    room.to(device), w.to(device))
            opt.zero_grad()
            pred = _forward(model, model_name, x, prior)
            loss_v = _loss(pred, y, room, w)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss_v.item() * x.shape[0]
            tn += x.shape[0]
        sched.step()
        tr_loss = tl / max(tn, 1)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, prior, y, room, w in ld_va:
                x, prior, y, room, w = (x.to(device), prior.to(device), y.to(device),
                                        room.to(device), w.to(device))
                pred = _forward(model, model_name, x, prior)
                # 验证一律用未加权 masked MSE，使不同方案的曲线彼此可比，
                # 也与评估阶段的 RMSE 口径一致。
                vl += masked_mse(pred, y, room).item() * x.shape[0]
                vn += x.shape[0]
        va_loss = vl / max(vn, 1)

        hist['epoch'].append(ep)
        hist['train'].append(tr_loss)
        hist['val'].append(va_loss)
        hist['lr'].append(opt.param_groups[0]['lr'])

        if va_loss < best - 1e-6:
            best, best_epoch = va_loss, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f'  ep {ep:4d}  train {tr_loss:.5f}  val {va_loss:.5f}  '
                  f'best {best:.5f}@{best_epoch}  {time.time() - t0:.0f}s', flush=True)

        if bad >= patience:
            if verbose:
                print(f'  早停于 epoch {ep}（best={best:.5f} @ {best_epoch}）', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    out = out_dir or os.path.join(root, f'models/{model_name}')
    os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, 'model.pt'))
    meta = {
        'model': model_name,
        'label': MODEL_LABELS[model_name],
        'width': width,
        'params': count_params(model),
        'normalizer': norm.to_dict(),
        'grid': grid.to_dict(),
        'has_prior': has_prior,
        'input_mode': mode,
        'n_input_channels': (6 + 1) if mode == 'multiprior' else (3 + 1 if has_prior else 3),
        'loss': loss,
        'train_files': [os.path.relpath(p, root) for p in tr_paths],
        'val_file': os.path.relpath(va_path, root),
        'train_size': len(ds_tr),
        'val_size': len(ds_va),
        'epochs_run': len(hist['epoch']),
        'best_epoch': best_epoch,
        'best_val_loss': best,
        'seed': seed,
        'lr': lr,
        'batch_size': batch_size,
        'device': device,
        'seconds': round(time.time() - t0, 1),
    }
    with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(hist, f, indent=2)

    if verbose:
        print(f'  保存 -> {out}  (best val {best:.5f} @ ep {best_epoch}, '
              f'{meta["seconds"]}s)', flush=True)
    return meta


def load_trained(model_name: str, root: str = '.', device: str = 'cpu'):
    """加载已训练模型与元数据。"""
    out = os.path.join(root, f'models/{model_name}')
    with open(os.path.join(out, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    model = build_model(model_name, width=meta['width'])
    model.load_state_dict(torch.load(os.path.join(out, 'model.pt'),
                                     map_location=device))
    model.to(device).eval()
    return model, meta, Normalizer.from_dict(meta['normalizer'])


# ==========================================================================
# CLI
# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description='RF Map 训练')
    ap.add_argument('--stage', choices=['sim'], default='sim')
    ap.add_argument('--models', default='ae,unet,resunet')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--width', type=int, default=24)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--root', default='.')
    ap.add_argument('--data-dir', default=None,
                    help='训练数据目录（默认 data/sim；方案四用 data/sim_scatter）')
    ap.add_argument('--out-dir', default=None,
                    help='权重输出目录（默认 models/<model>）')
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--loss', default='auto', choices=['auto', 'mse', 'weighted'])
    args = ap.parse_args(argv)

    names = [s.strip() for s in args.models.split(',') if s.strip()]
    results = {}
    for nm in names:
        print(f'== 训练 {MODEL_LABELS[nm]} ({nm}) ==', flush=True)
        out_dir = args.out_dir or os.path.join(args.root, f'models/{nm}')
        results[nm] = train_sim(nm, epochs=args.epochs, width=args.width,
                                batch_size=args.batch, lr=args.lr,
                                seed=args.seed, device=args.device,
                                root=args.root, patience=args.patience,
                                data_dir=args.data_dir, out_dir=out_dir,
                                loss=args.loss)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
