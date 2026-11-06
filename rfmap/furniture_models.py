"""五种家具感知 RF Map 网络、输入构造与推理。"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn

from .grid import Grid, DEFAULT as DEFAULT_GRID
from .models import (
    CompletionAE, SmallUNet, Normalizer, build_input_tensor, build_input_tensor_v5,
    prepare_prior, idw_predict, log_distance_trend, nearest_obs_distance, obs_density,
    count_params,
)
from .train import _precompute_idw, _precompute_extra, DENSITY_WEIGHT

MODEL_ORDER = ['ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr']
MODEL_LABELS = {
    'ae': '方案一 · 卷积补全自编码器',
    'unet': '方案二 · 小型 U-Net',
    'resunet': '方案三 · 插值引导残差 U-Net',
    'unet_scatter': '方案四 · 散落布点训练的 U-Net',
    'pgmpr': '方案五 · 物理引导多先验残差 U-Net',
}
BASELINE_LABELS = {
    'nearest': '基线 · 最近邻',
    'idw': '基线 · 反距离加权（IDW, p=2）',
    'rbf': '基线 · 零均值 RBF',
    'gp': '基线 · 对数距离趋势 + RBF 残差',
}
PRIOR_MODELS = {'resunet', 'pgmpr'}


class FurnitureResidualUNet(nn.Module):
    """以 IDW 为残差基准；x 的最后拼接由 forward 统一完成。"""

    def __init__(self, width: int, base_channels: int):
        super().__init__()
        self.net = SmallUNet(width=width, in_ch=base_channels + 1, out_ch=1)

    def forward(self, x, prior):
        return prior + self.net(torch.cat([x, prior], dim=1))


def build_furniture_model(name: str, width: int = 24) -> nn.Module:
    if name == 'ae':
        return CompletionAE(width=width, in_ch=4)
    if name in ('unet', 'unet_scatter'):
        return SmallUNet(width=width, in_ch=4, out_ch=1)
    if name == 'resunet':
        return FurnitureResidualUNet(width=width, base_channels=4)
    if name == 'pgmpr':
        return FurnitureResidualUNet(width=width, base_channels=7)
    raise KeyError(f'未知模型：{name}')


def input_mode(name_or_meta) -> str:
    if isinstance(name_or_meta, dict):
        return name_or_meta.get('input_mode', 'base')
    name = name_or_meta
    if name == 'pgmpr':
        return 'multiprior'
    if name in PRIOR_MODELS:
        return 'prior'
    return 'base'


def n_parameters(model: nn.Module) -> int:
    return count_params(model)


def _base_with_furniture(obs_v, obs_m, room, furniture, norm):
    x = build_input_tensor(obs_v, obs_m, room, norm)
    f = (furniture > 0.5).astype(np.float32)[None]
    return np.concatenate([x[:3], f], axis=0).astype(np.float32)


def make_input(name_or_meta, obs_v, obs_m, room, furniture, norm,
               grid: Grid = DEFAULT_GRID, prior=None):
    """构造基础通道；家具始终位于索引 3，IDW prior 单独返回。

    方案一/二/四：4 通道 = 原三通道 + 家具。
    方案三：上述 4 通道 + forward 内拼 IDW，共 5 通道。
    方案五：原三通道 + 家具 + 趋势/距离/密度 + IDW，共 8 通道。
    """
    mode = input_mode(name_or_meta)
    X, Y = grid.centers()
    if mode == 'multiprior':
        js, iss = np.nonzero(obs_m > 0.5)
        pts = np.stack([X[js, iss], Y[js, iss]], axis=1)
        vals = obs_v[js, iss]
        trend = log_distance_trend(pts, vals, X, Y, room > 0.5)
        dist = nearest_obs_distance(obs_m, grid.cell)
        density = obs_density(obs_m)
        x6 = build_input_tensor_v5(obs_v, obs_m, room, norm, trend, dist, density)
        x = np.concatenate([x6[:3], (furniture > 0.5).astype(np.float32)[None],
                            x6[3:]], axis=0).astype(np.float32)
        if prior is None:
            prior = idw_predict(pts, vals, X, Y, room > 0.5)
        return x, prior
    x = _base_with_furniture(obs_v, obs_m, room, furniture, norm)
    if mode == 'prior':
        if prior is None:
            js, iss = np.nonzero(obs_m > 0.5)
            pts = np.stack([X[js, iss], Y[js, iss]], axis=1)
            prior = idw_predict(pts, obs_v[js, iss], X, Y, room > 0.5)
        return x, prior
    return x, None


class FurnitureDataset(torch.utils.data.Dataset):
    """仿真数据集；先验只计算一次，训练时不在每个 epoch 重算。"""

    def __init__(self, paths, norm: Normalizer, model_name: str,
                 grid: Grid = DEFAULT_GRID):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        arrays = {k: [] for k in ('obs_value', 'obs_mask', 'room', 'target', 'furniture')}
        for path in self.paths:
            d = np.load(path)
            for key in arrays:
                arrays[key].append(d[key].astype(np.float32))
        for key in arrays:
            setattr(self, key, np.concatenate(arrays[key]))
        self.norm = norm
        self.grid = grid
        self.model_name = model_name
        self.mode = input_mode(model_name)
        self.prior = None
        self.extra = None
        self.weight = None
        if self.mode in ('prior', 'multiprior'):
            self.prior = np.concatenate([
                _precompute_idw(d['obs_value'].astype(np.float32),
                                d['obs_mask'].astype(np.float32), grid,
                                d['room'].astype(np.float32), log_every=400)
                for d in (np.load(p) for p in self.paths)
            ])
        if self.mode == 'multiprior':
            self.extra = np.concatenate([
                _precompute_extra(d['obs_value'].astype(np.float32),
                                  d['obs_mask'].astype(np.float32), grid,
                                  d['room'].astype(np.float32), log_every=400)
                for d in (np.load(p) for p in self.paths)
            ])
            w = 1.0 + DENSITY_WEIGHT * self.extra[:, 2]
            m = (self.room > 0.5).astype(np.float32)
            denom = np.maximum((w * m).sum(axis=(1, 2), keepdims=True), 1e-6)
            self.weight = (w * m / (denom / np.maximum(
                m.sum(axis=(1, 2), keepdims=True), 1.0))).astype(np.float32)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, k):
        ov, om, rm, fm = self.obs_value[k], self.obs_mask[k], self.room[k], self.furniture[k]
        if self.mode == 'multiprior':
            x6 = build_input_tensor_v5(ov, om, rm, self.norm, self.extra[k, 0],
                                       self.extra[k, 1], self.extra[k, 2])
            x = np.concatenate([x6[:3], fm[None], x6[3:]], axis=0).astype(np.float32)
        else:
            x = _base_with_furniture(ov, om, rm, fm, self.norm)
        y = (self.norm.encode(self.target[k]) * (rm > 0.5)).astype(np.float32)
        prior = (prepare_prior(self.prior[k], rm, self.norm)[None]
                 if self.prior is not None else np.zeros((1,) + self.grid.shape, np.float32))
        weight = (self.weight[k][None] if self.weight is not None
                  else np.ones((1,) + self.grid.shape, np.float32))
        return x, prior, y[None], rm[None], weight


def forward_grid(model, meta, norm, obs_v, obs_m, room, furniture,
                 grid: Grid = DEFAULT_GRID, device='cpu', prior=None):
    mode = input_mode(meta)
    x, pr = make_input(meta, obs_v, obs_m, room, furniture, norm, grid, prior)
    xt = torch.from_numpy(x[None]).to(device)
    with torch.no_grad():
        if mode in ('prior', 'multiprior'):
            pt = torch.from_numpy(prepare_prior(pr, room, norm)[None, None]).to(device)
            out = model(xt, pt)
        else:
            out = model(xt)
    pred = norm.decode(out[0, 0].detach().cpu().numpy()).astype(np.float64)
    return np.where(room > 0.5, pred, np.nan)


_TTA = (
    lambda a: a,
    np.flipud,
    np.fliplr,
    lambda a: np.flipud(np.fliplr(a)),
)


def predict_grid(model, meta, norm, obs_v, obs_m, room, furniture,
                 grid: Grid = DEFAULT_GRID, device='cpu', prior=None, tta=False):
    if not tta:
        return forward_grid(model, meta, norm, obs_v, obs_m, room, furniture,
                            grid, device, prior)
    acc = np.zeros(grid.shape, dtype=float)
    for transform in _TTA:
        p = forward_grid(model, meta, norm, transform(obs_v), transform(obs_m),
                         transform(room), transform(furniture), grid, device,
                         transform(prior) if prior is not None else None)
        acc += transform(np.nan_to_num(p, nan=0.0))
    return np.where(room > 0.5, acc / len(_TTA), np.nan)


def load_trained(model_dir: str, device='cpu', finetuned=False):
    with open(os.path.join(model_dir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    model = build_furniture_model(meta['model'], width=meta['width'])
    filename = 'model_finetuned.pt' if finetuned else 'model.pt'
    state = torch.load(os.path.join(model_dir, filename), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    norm = Normalizer.from_dict(meta['normalizer'])
    return model, meta, norm
