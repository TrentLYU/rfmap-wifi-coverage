"""放家具场景：五模型训练、实测微调、评估与最终整图预测。

默认完整运行：
    .venv/bin/python -m rfmap.furniture_experiment --stage all --device mps

所有输出写入独立目录 outputs/rfmap-furniture，不覆盖上一版空房实验。
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import time

import numpy as np
import torch

from . import geometry as G
from .evaluate import bilinear_sample, build_obs, metrics
from .grid import DEFAULT as GRID
from .models import BASELINE_ZOO, Normalizer
from .train import masked_mse, weighted_masked_mse
from .furniture_data import (
    read_workbook, write_normalized, generate_datasets, furniture_mask,
    point_furniture_distance, SIM_SEEDS, SCATTER_SEEDS, Y_SCALE,
)
from .furniture_models import (
    MODEL_ORDER, MODEL_LABELS, BASELINE_LABELS, FurnitureDataset,
    build_furniture_model, input_mode, n_parameters, predict_grid, load_trained,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, 'outputs/rfmap-furniture')
DEFAULT_WORK = os.path.join(ROOT, '.deepworks/tmp/rfmap-furniture')
INPUT_XLSX = os.path.join(ROOT, '放家具数据.xlsx')


def _json_dump(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_data(out_dir):
    pts, furniture, quality = read_workbook(INPUT_XLSX)
    write_normalized(pts, furniture, quality, os.path.join(out_dir, 'data'))
    print(f'实测数据：{len(pts)} 点；家具 {len(furniture)} 件；'
          f'RSSI {quality["rssi_min"]:.0f}~{quality["rssi_max"]:.0f} dBm；'
          f'符号修正 {quality["raw_positive_count"]} 点', flush=True)
    return pts, furniture, quality


def _forward(model, name, x, prior):
    return model(x, prior) if input_mode(name) in ('prior', 'multiprior') else model(x)


def train_one(name, train_paths, val_path, out_root, device='mps', epochs=120,
              patience=35, width=24, batch_size=16, lr=2e-3, seed=7):
    torch.manual_seed(seed)
    np.random.seed(seed)
    raw = np.load(train_paths[0])
    norm = Normalizer.fit(raw['target'], raw['room'])
    ds_tr = FurnitureDataset(train_paths, norm, name, GRID)
    ds_va = FurnitureDataset(val_path, norm, name, GRID)
    ld_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                                        num_workers=0, drop_last=True)
    ld_va = torch.utils.data.DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                                        num_workers=0)
    model = build_furniture_model(name, width=width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    weighted = name == 'pgmpr'
    hist = {'epoch': [], 'train': [], 'val': [], 'lr': []}
    best, best_epoch, best_state, bad = float('inf'), -1, None, 0
    t0 = time.time()
    print(f'\n== {MODEL_LABELS[name]} ==', flush=True)
    for ep in range(epochs):
        model.train()
        total, count = 0.0, 0
        for x, prior, y, room, weight in ld_tr:
            x, prior, y, room, weight = (x.to(device), prior.to(device), y.to(device),
                                         room.to(device), weight.to(device))
            opt.zero_grad()
            pred = _forward(model, name, x, prior)
            loss = (weighted_masked_mse(pred, y, room, weight) if weighted
                    else masked_mse(pred, y, room))
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name} epoch {ep} 出现非有限损失：{loss}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * x.shape[0]
            count += x.shape[0]
        sched.step()
        train_loss = total / max(count, 1)

        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, prior, y, room, weight in ld_va:
                x, prior, y, room = x.to(device), prior.to(device), y.to(device), room.to(device)
                pred = _forward(model, name, x, prior)
                total += float(masked_mse(pred, y, room).item()) * x.shape[0]
                count += x.shape[0]
        val_loss = total / max(count, 1)
        hist['epoch'].append(ep)
        hist['train'].append(train_loss)
        hist['val'].append(val_loss)
        hist['lr'].append(opt.param_groups[0]['lr'])
        if val_loss < best - 1e-6:
            best, best_epoch = val_loss, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'  ep {ep:3d} train={train_loss:.5f} val={val_loss:.5f} '
                  f'best={best:.5f}@{best_epoch} {time.time()-t0:.0f}s', flush=True)
        if bad >= patience:
            print(f'  早停：epoch {ep}', flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    out_dir = os.path.join(out_root, 'models', name)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    mode = input_mode(name)
    meta = {
        'model': name,
        'label': MODEL_LABELS[name],
        'width': width,
        'params': n_parameters(model),
        'normalizer': norm.to_dict(),
        'grid': GRID.to_dict(),
        'input_mode': mode,
        'furniture_channel_index': 3,
        'n_input_channels': {'base': 4, 'prior': 5, 'multiprior': 8}[mode],
        'loss': 'weighted_masked_mse' if weighted else 'masked_mse',
        'train_files': train_paths,
        'val_file': val_path,
        'train_size': len(ds_tr),
        'val_size': len(ds_va),
        'epochs_max': epochs,
        'epochs_run': len(hist['epoch']),
        'best_epoch': best_epoch,
        'best_val_loss': best,
        'seed': seed,
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': 1e-5,
        'gradient_clip': 5.0,
        'patience': patience,
        'device': device,
        'seconds': round(time.time() - t0, 1),
        'framework': f'torch-{torch.__version__}',
    }
    _json_dump(meta, os.path.join(out_dir, 'meta.json'))
    _json_dump(hist, os.path.join(out_dir, 'history.json'))
    print(f'  保存 {out_dir}；最佳 {best:.5f}@ep{best_epoch}', flush=True)
    return meta


def train_all(paths, out_root, device='mps', epochs=120, patience=35, only=None):
    specs = {
        'ae': [paths['mixed']['train']],
        'unet': [paths['mixed']['train']],
        'resunet': [paths['mixed']['train']],
        'unet_scatter': [paths['scatter']['train']],
        'pgmpr': [paths['mixed']['train'], paths['scatter']['train']],
    }
    summary_path = os.path.join(out_root, 'training_summary.json')
    summary = {}
    if only and os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            summary = json.load(f)
    for name in MODEL_ORDER:
        if only and name not in only:
            continue
        # CompletionAE 在本数据上约 40 轮后才脱离初始平台期；默认给它更长窗口，
        # 其余模型保持 120/35。显式传入非默认值（例如冒烟测试 2/2）时不覆盖。
        name_epochs = 160 if name == 'ae' and epochs == 120 else epochs
        name_patience = 80 if name == 'ae' and patience == 35 else patience
        summary[name] = train_one(name, specs[name], paths['mixed']['val'], out_root,
                                  device=device, epochs=name_epochs, patience=name_patience)
    _json_dump(summary, summary_path)
    return summary


def finetune_real(model, meta, norm, pts, furniture, device='mps', epochs=300,
                  lr=3e-4, seed=123):
    """只用允许的输入点做自监督微调；输入集合与监督集合每轮均不相交。"""
    m = copy.deepcopy(model)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    room = GRID.room_mask().astype(np.float32)
    fm = furniture_mask(furniture, GRID, room)
    n = len(pts)
    if n < 4:
        return m
    ij = [GRID.xy_to_idx(p.x, p.y) for p in pts]
    jjs = np.array([v[0] for v in ij])
    iis = np.array([v[1] for v in ij])
    vals = np.array([p.rssi for p in pts], dtype=np.float32)
    for _ in range(epochs):
        k = int(rng.integers(max(3, int(n * 0.5)), max(4, n - 2)))
        sel = rng.choice(n, size=min(k, n - 2), replace=False)
        sup = np.setdiff1d(np.arange(n), sel)
        obs_v = np.zeros(GRID.shape, np.float32)
        obs_m = np.zeros(GRID.shape, np.float32)
        obs_v[jjs[sel], iis[sel]] = vals[sel]
        obs_m[jjs[sel], iis[sel]] = 1.0
        sup_m = np.zeros(GRID.shape, np.float32)
        sup_m[jjs[sup], iis[sup]] = 1.0
        y_val = np.zeros(GRID.shape, np.float32)
        y_val[jjs, iis] = vals
        x, pr = __import__('rfmap.furniture_models', fromlist=['make_input']).make_input(
            meta, obs_v, obs_m, room, fm, norm, GRID)
        xt = torch.from_numpy(x[None]).to(device)
        yt = torch.from_numpy((norm.encode(y_val) * sup_m)[None, None]).to(device)
        mt = torch.from_numpy(sup_m[None, None]).to(device)
        if input_mode(meta) in ('prior', 'multiprior'):
            from .models import prepare_prior
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


def _sync(device):
    if device == 'mps' and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _time_fn(fn, device, repeats=5):
    fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    _sync(device)
    return round((time.perf_counter() - t0) / repeats * 1000, 2)


def _protocol_split(pts, furniture, protocol):
    if protocol == 'corner':
        keys = {(x, y) for x in (0.9, 1.09) for y in (9.81, 10.9, 12.0)}
        hidden = [p for p in pts if (p.x_data, p.y_data) in keys]
    elif protocol == 'center':
        keys = {(x, y) for x in (4.09, 5.09, 6.09)
                for y in (4.36, 5.45, 6.54)}
        hidden = [p for p in pts if (p.x_data, p.y_data) in keys]
    elif protocol == 'furniture':
        distances = point_furniture_distance(pts, furniture)
        hidden = [p for p, d in zip(pts, distances) if d <= 0.6]
    else:
        raise KeyError(protocol)
    hidden_ids = {p.idx for p in hidden}
    inp = [p for p in pts if p.idx not in hidden_ids]
    return inp, hidden


def _method_metrics(pred, hid, furniture, infer_ms=None):
    preds = [bilinear_sample(pred, p.x, p.y, GRID) for p in hid]
    truth = [p.rssi for p in hid]
    out = metrics(preds, truth)
    out['infer_ms'] = infer_ms
    dist = point_furniture_distance(hid, furniture)
    near = [i for i, d in enumerate(dist) if d <= 0.6]
    far = [i for i, d in enumerate(dist) if d > 0.6]
    out['near_furniture'] = metrics([preds[i] for i in near], [truth[i] for i in near])
    out['far_furniture'] = metrics([preds[i] for i in far], [truth[i] for i in far])
    out['per_point'] = [[p.x_data, p.y_data, p.rssi,
                         round(float(v), 4) if np.isfinite(v) else None,
                         round(float(d), 4)]
                        for p, v, d in zip(hid, preds, dist)]
    return out


def evaluate_real(out_root, pts, furniture, device='mps', finetune_epochs=300,
                  sweep_repeats=20, timing_repeats=5):
    room = GRID.room_mask().astype(np.float32)
    fm = furniture_mask(furniture, GRID, room)
    X, Y = GRID.centers()
    loaded = {name: load_trained(os.path.join(out_root, 'models', name), device)
              for name in MODEL_ORDER}
    result = {
        'model_labels': MODEL_LABELS, 'baseline_labels': BASELINE_LABELS,
        'n_points_total': len(pts), 'finetune_epochs': finetune_epochs,
        'protocols': {}, 'sweep': {}, 'near_furniture_definition': '距家具表面 <=0.6 m（约等于距中心 <=1.0 m）',
    }
    labels = {'corner': '角块留出（外推）', 'center': '中心块留出（插值）',
              'furniture': '家具附近留出（遮挡压力测试）'}
    for protocol in ('corner', 'center', 'furniture'):
        inp, hid = _protocol_split(pts, furniture, protocol)
        if len(hid) < 2:
            raise RuntimeError(f'{protocol} 隐藏点不足：{len(hid)}')
        obs_v, obs_m = build_obs(inp, GRID)
        entry = {'label': labels[protocol], 'n_input': len(inp), 'n_hidden': len(hid),
                 'hidden_points': [[p.x_data, p.y_data, p.rssi] for p in hid],
                 'methods': {}}
        print(f'真实协议 {protocol}: 输入 {len(inp)}，隐藏 {len(hid)}', flush=True)
        for name, (model, meta, norm) in loaded.items():
            fn = lambda m=model, mt=meta, n=norm: predict_grid(
                m, mt, n, obs_v, obs_m, room, fm, GRID, device)
            pred = fn()
            entry['methods'][name] = _method_metrics(
                pred, hid, furniture, _time_fn(fn, device, timing_repeats))
            ft = finetune_real(model, meta, norm, inp, furniture, device=device,
                               epochs=finetune_epochs, seed=123)
            fn_ft = lambda m=ft, mt=meta, n=norm: predict_grid(
                m, mt, n, obs_v, obs_m, room, fm, GRID, device)
            pred_ft = fn_ft()
            entry['methods'][name + '_ft'] = _method_metrics(
                pred_ft, hid, furniture, _time_fn(fn_ft, device, timing_repeats))
            if name == 'pgmpr':
                pred_tta = predict_grid(model, meta, norm, obs_v, obs_m, room, fm,
                                        GRID, device, tta=True)
                entry['methods']['pgmpr_tta'] = _method_metrics(pred_tta, hid, furniture)
                pred_ft_tta = predict_grid(ft, meta, norm, obs_v, obs_m, room, fm,
                                           GRID, device, tta=True)
                entry['methods']['pgmpr_ft_tta'] = _method_metrics(pred_ft_tta, hid, furniture)
        js, iss = np.nonzero(obs_m > 0.5)
        pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
        pv = obs_v[js, iss]
        for bn, fn_base in BASELINE_ZOO.items():
            pred = fn_base(pxy, pv, X, Y, room > 0.5)
            entry['methods'][bn] = _method_metrics(
                pred, hid, furniture,
                _time_fn(lambda f=fn_base: f(pxy, pv, X, Y, room > 0.5), device,
                         timing_repeats))
        result['protocols'][protocol] = entry

    rng = np.random.default_rng(123)
    for k in (10, 20, 40, 60, 80, 100):
        acc = {name: [] for name in MODEL_ORDER + list(BASELINE_ZOO)}
        for _ in range(sweep_repeats):
            sel = rng.choice(len(pts), size=min(k, len(pts) - 2), replace=False)
            selected = set(sel.tolist())
            inp = [p for i, p in enumerate(pts) if i in selected]
            hid = [p for i, p in enumerate(pts) if i not in selected]
            obs_v, obs_m = build_obs(inp, GRID)
            for name, (model, meta, norm) in loaded.items():
                pred = predict_grid(model, meta, norm, obs_v, obs_m, room, fm, GRID, device)
                vals = [bilinear_sample(pred, p.x, p.y, GRID) for p in hid]
                acc[name].append(metrics(vals, [p.rssi for p in hid])['rmse'])
            js, iss = np.nonzero(obs_m > 0.5)
            pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
            pv = obs_v[js, iss]
            for bn, fn_base in BASELINE_ZOO.items():
                pred = fn_base(pxy, pv, X, Y, room > 0.5)
                vals = [bilinear_sample(pred, p.x, p.y, GRID) for p in hid]
                acc[bn].append(metrics(vals, [p.rssi for p in hid])['rmse'])
        result['sweep'][str(k)] = {
            name: {'rmse_mean': round(float(np.nanmean(v)), 4),
                   'rmse_std': round(float(np.nanstd(v)), 4), 'runs': len(v)}
            for name, v in acc.items()
        }
        print(f'  点数扫描 K={k} 完成', flush=True)
    _json_dump(result, os.path.join(out_root, 'results/metrics_real.json'))
    return result


def evaluate_sim(out_root, paths, device='mps'):
    X, Y = GRID.centers()
    loaded = {name: load_trained(os.path.join(out_root, 'models', name), device)
              for name in MODEL_ORDER}
    result = {'model_labels': MODEL_LABELS, 'baseline_labels': BASELINE_LABELS,
              'testsets': {}}
    for tag in ('mixed', 'scatter'):
        d = np.load(paths[tag]['test'])
        n = d['target'].shape[0]
        names = MODEL_ORDER + list(BASELINE_ZOO)
        acc = {name: {'rmse': [], 'mae': []} for name in names}
        obs_counts = []
        for k in range(n):
            ov, om = d['obs_value'][k], d['obs_mask'][k]
            room, fm, target = d['room'][k], d['furniture'][k], d['target'][k]
            unknown = (room > 0.5) & (om < 0.5)
            obs_counts.append(int(om.sum()))
            for name, (model, meta, norm) in loaded.items():
                pred = predict_grid(model, meta, norm, ov, om, room, fm, GRID, device)
                mask = unknown & np.isfinite(pred)
                err = pred[mask] - target[mask]
                acc[name]['rmse'].append(float(np.sqrt((err ** 2).mean())))
                acc[name]['mae'].append(float(np.abs(err).mean()))
            js, iss = np.nonzero(om > 0.5)
            pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
            pv = ov[js, iss]
            for bn, fn_base in BASELINE_ZOO.items():
                pred = fn_base(pxy, pv, X, Y, room > 0.5)
                mask = unknown & np.isfinite(pred)
                err = pred[mask] - target[mask]
                acc[bn]['rmse'].append(float(np.sqrt((err ** 2).mean())))
                acc[bn]['mae'].append(float(np.abs(err).mean()))
            if (k + 1) % 50 == 0:
                print(f'  仿真 {tag}: {k + 1}/{n}', flush=True)
        methods = {}
        for name, values in acc.items():
            rm = np.array(values['rmse'])
            mae = np.array(values['mae'])
            methods[name] = {
                'rmse_mean': round(float(rm.mean()), 4),
                'rmse_median': round(float(np.median(rm)), 4),
                'rmse_p90': round(float(np.percentile(rm, 90)), 4),
                'mae_mean': round(float(mae.mean()), 4), 'runs': n,
            }
        result['testsets'][tag] = {
            'source': paths[tag]['test'], 'n_samples': n,
            'obs_count_mean': round(float(np.mean(obs_counts)), 2),
            'obs_count_range': [int(min(obs_counts)), int(max(obs_counts))],
            'methods': methods,
        }
    _json_dump(result, os.path.join(out_root, 'results/metrics_sim.json'))
    return result


def fit_final_models(out_root, pts, furniture, device='mps', epochs=300):
    """部署用模型用全部 140 点微调；其整图不用于声称独立测试精度。"""
    room = GRID.room_mask().astype(np.float32)
    fm = furniture_mask(furniture, GRID, room)
    obs_v, obs_m = build_obs(pts, GRID)
    X, Y = GRID.centers()
    summary = {}
    for name in MODEL_ORDER:
        model_dir = os.path.join(out_root, 'models', name)
        model, meta, norm = load_trained(model_dir, device)
        t0 = time.time()
        ft = finetune_real(model, meta, norm, pts, furniture, device=device,
                           epochs=epochs, seed=123)
        torch.save(ft.state_dict(), os.path.join(model_dir, 'model_finetuned.pt'))
        ft_meta = {'epochs': epochs, 'lr': 3e-4, 'seed': 123,
                   'n_real_points': len(pts), 'seconds': round(time.time() - t0, 1),
                   'purpose': '全部实测点微调后的部署整图；不是独立留出测试'}
        _json_dump(ft_meta, os.path.join(model_dir, 'finetune.json'))
        pred = predict_grid(ft, meta, norm, obs_v, obs_m, room, fm, GRID, device)
        os.makedirs(os.path.join(out_root, 'predictions'), exist_ok=True)
        np.savez_compressed(os.path.join(out_root, 'predictions', name + '_full_map.npz'),
                            rssi=pred.astype(np.float32), room=room, furniture=fm,
                            x=GRID.x_coords(), y=GRID.y_coords())
        with open(os.path.join(out_root, 'predictions', name + '_full_map.csv'), 'w',
                  newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['x_m', 'y_m', 'rssi_pred_dbm', 'inside_room', 'furniture'])
            for j in range(GRID.ny):
                for i in range(GRID.nx):
                    if room[j, i] > 0.5:
                        w.writerow([round(float(X[j, i]), 3), round(float(Y[j, i]), 3),
                                    round(float(pred[j, i]), 4), 1, int(fm[j, i] > 0.5)])
        summary[name] = ft_meta
        print(f'  最终模型与整图：{name}', flush=True)
    _json_dump(summary, os.path.join(out_root, 'results/final_finetune.json'))
    return summary


def write_manifest(out_root, args, quality):
    training_path = os.path.join(out_root, 'training_summary.json')
    training = {}
    if os.path.exists(training_path):
        with open(training_path, encoding='utf-8') as f:
            training = json.load(f)
    manifest = {
        'generated_at': dt.datetime.now().isoformat(timespec='seconds'),
        'resolved_training': {name: {'epochs_max': m.get('epochs_max', 120),
                                     'patience': m.get('patience', 35),
                                     'best_epoch': m.get('best_epoch')}
                              for name, m in training.items()},
        'input_file': os.path.basename(INPUT_XLSX),
        'input_sha256': _sha256(INPUT_XLSX),
        'furniture_figure': '实验室户型图（加入家具版本）.png',
        'floorplan_xy_figure': '实验地户型 （xy轴）.png',
        'coordinate_convention': quality['coordinate_convention'],
        'orientation_assumption': quality['orientation_assumption'],
        'rssi_rule': quality['rssi_rule'],
        'sim_seeds': SIM_SEEDS,
        'scatter_seeds': SCATTER_SEEDS,
        'train_seed': 7,
        'eval_seed': 123,
        'config': vars(args),
        'environment': {
            'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'platform': platform.platform(),
        },
        'code_files': {},
    }
    for filename in ('furniture_data.py', 'furniture_models.py',
                     'furniture_experiment.py', 'furniture_report.py',
                     'furniture_selfcheck.py'):
        path = os.path.join(ROOT, 'rfmap', filename)
        if os.path.exists(path):
            manifest['code_files'][filename] = _sha256(path)
    _json_dump(manifest, os.path.join(out_root, 'run_manifest.json'))


def _paths(work_dir):
    return {tag: {split: os.path.join(work_dir, directory, split + '.npz')
                  for split in ('train', 'val', 'test')}
            for tag, directory in (('mixed', 'sim'), ('scatter', 'sim_scatter'))}


def main(argv=None):
    ap = argparse.ArgumentParser(description='放家具 RF Map 五模型实验')
    ap.add_argument('--stage', choices=['all', 'data', 'sim', 'train', 'eval', 'final', 'report'],
                    default='all')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--out-root', default=DEFAULT_OUT)
    ap.add_argument('--work-dir', default=DEFAULT_WORK)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--patience', type=int, default=35)
    ap.add_argument('--finetune-epochs', type=int, default=300)
    ap.add_argument('--n-train', type=int, default=1200)
    ap.add_argument('--n-val', type=int, default=200)
    ap.add_argument('--n-test', type=int, default=200)
    ap.add_argument('--sweep-repeats', type=int, default=20)
    ap.add_argument('--timing-repeats', type=int, default=5)
    ap.add_argument('--only', nargs='*', choices=MODEL_ORDER, default=None,
                    help='仅重训指定方案并合并训练摘要；评估仍统一跑五种模型')
    args = ap.parse_args(argv)
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print('MPS 不可用，改用 CPU', flush=True)
        args.device = 'cpu'
    _json_dump(vars(args), os.path.join(args.out_root, 'run_config.json'))
    pts, furniture, quality = prepare_data(args.out_root)
    paths = _paths(args.work_dir)
    if args.stage in ('all', 'sim'):
        paths = generate_datasets(args.work_dir, args.n_train, args.n_val, args.n_test, GRID)
    if args.stage in ('all', 'train'):
        train_all(paths, args.out_root, args.device, args.epochs, args.patience, args.only)
    if args.stage in ('all', 'eval'):
        evaluate_sim(args.out_root, paths, args.device)
        evaluate_real(args.out_root, pts, furniture, args.device,
                      args.finetune_epochs, args.sweep_repeats, args.timing_repeats)
    if args.stage in ('all', 'final'):
        fit_final_models(args.out_root, pts, furniture, args.device, args.finetune_epochs)
    if args.stage in ('all', 'report'):
        from .furniture_report import build_all
        build_all(args.out_root)
    write_manifest(args.out_root, args, quality)
    print(f'\n完成。交付目录：{args.out_root}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
