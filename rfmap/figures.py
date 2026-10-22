"""一键生成实验报告所需的全部图。

只负责"读结果 -> 出图"，不重新训练、不重新评估；所有数字都来自
``outputs/rfmap-results/*.json`` 与 ``models/*/history.json``，
保证图与报告里的数值同源。
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import data as D
from . import evaluate as E
from . import plotting as P
from .grid import DEFAULT as DEFAULT_GRID
from .models import BASELINE_ZOO, BASELINE_LABELS, MODEL_LABELS, idw_predict
from .train import load_trained


class MarkedPoint:
    """给图上的测点加个颜色标记，便于区分输入点 / 隐藏点。"""

    def __init__(self, p, mark):
        self.x, self.y, self.rssi = p.x, p.y, p.rssi
        self.x_data, self.y_data = p.x_data, p.y_data
        self.mark = mark


def _pred_fields(root, grid, room, obs_v, obs_m, device='cpu',
                 model_names=('ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr'),
                 finetune_pts=None, finetune_epochs=300, tta=()):
    """算出各方法的预测场，返回 {key: field}。"""
    fields = {}
    for nm in model_names:
        try:
            model, meta, norm = load_trained(nm, root=root, device=device)
        except FileNotFoundError:
            continue
        fields[nm] = E.nn_predict_grid(model, meta, norm, obs_v, obs_m, room,
                                       grid, device)
        if nm in tta:
            fields[nm + '_tta'] = E.nn_predict_grid(model, meta, norm, obs_v,
                                                    obs_m, room, grid, device,
                                                    tta=True)
        if finetune_pts:
            ft = E.finetune_real(model, meta, norm, finetune_pts, grid, room,
                                 epochs=finetune_epochs, device=device)
            fields[nm + '_ft'] = E.nn_predict_grid(ft, meta, norm, obs_v, obs_m,
                                                   room, grid, device)

    X, Y = grid.centers()
    js, iss = np.nonzero(obs_m > 0.5)
    pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
    pv = obs_v[js, iss]
    for bn in ('idw', 'gp', 'nearest'):
        fields[bn] = BASELINE_ZOO[bn](pxy, pv, X, Y, room > 0.5)
    return fields


# 报告正文里"五个方案"的固定展示顺序（方案一~五）
SCHEME_ORDER = ('ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr')

# 两个仿真测试集的显示名。它们逐元素共享同一批户型和真值，只有布点方式不同。
TEST_LABELS = {
    'mixed': '常规布点测试集',
    'scatter': '散落布点测试集',
}


def make_all_figures(root: str = '.', out_dir: str = 'outputs/figures',
                     device: str = 'cpu') -> dict:
    P.setup_fonts()
    os.makedirs(out_dir, exist_ok=True)
    grid = DEFAULT_GRID
    room = grid.room_mask().astype(np.float32)
    pts = D.load(root=root)
    made = {}
    labels = dict(BASELINE_LABELS)
    labels.update(MODEL_LABELS)

    # ---- 0. 布点方式对比 + 方案五结构 ----
    a = os.path.join(root, 'data/sim/train.npz')
    b = os.path.join(root, 'data/sim_scatter/train.npz')
    if os.path.exists(a) and os.path.exists(b):
        made['obs_layout'] = P.fig_obs_layout(
            [('常规布点（方案一 / 二 / 三 的训练数据）', a),
             ('散落布点（方案四的训练数据）', b)],
            f'{out_dir}/fig_obs_layout.png',
            suptitle='布点方式对比：同一批户型、同样的观测点数，只改位置')
    made['arch_v5'] = P.fig_arch_v5(f'{out_dir}/fig_arch_v5.png')

    # ---- 1. 几何 / 数据画像 ----
    made['geometry'] = P.fig_geometry(pts, f'{out_dir}/fig_geometry.png', grid)
    made['data_profile'] = P.fig_data_profile(pts, f'{out_dir}/fig_data_profile.png')

    # ---- 2. 训练曲线（五个方案） ----
    hists, hlabels = {}, {}
    for nm in SCHEME_ORDER:
        p = os.path.join(root, f'models/{nm}/history.json')
        if os.path.exists(p):
            hists[nm] = json.load(open(p, encoding='utf-8'))
            hlabels[nm] = MODEL_LABELS[nm]
    if hists:
        made['train_curves'] = P.fig_training_curves(
            hists, f'{out_dir}/fig_training_curves.png', hlabels)

    # ---- 3. 输入点数扫描 ----
    mreal = os.path.join(root, 'outputs/rfmap-results/metrics_real.json')
    if os.path.exists(mreal):
        d = json.load(open(mreal, encoding='utf-8'))
        lab = dict(labels)
        lab.update({k + '_ft': v + '（实测微调）' for k, v in MODEL_LABELS.items()})
        made['sweep'] = P.fig_sweep(
            d['sweep'], f'{out_dir}/fig_sweep.png', lab,
            only=['unet', 'unet_scatter', 'pgmpr', 'resunet', 'idw', 'gp'])

    # ---- 4. 五方案跨户型对比（两个测试集） ----
    msim = os.path.join(root, 'outputs/rfmap-results/metrics_sim.json')
    if os.path.exists(msim):
        d = json.load(open(msim, encoding='utf-8'))
        testsets = d.get('testsets') or {}
        groups = []
        for tag in ('mixed', 'scatter'):
            e = testsets.get(tag)
            if not e:
                continue
            items = []
            for k, v in e['methods'].items():
                if k.endswith('_tta') or k == 'rbf':
                    continue
                name = labels.get(k, k)
                if k in BASELINE_LABELS:
                    name = MODEL_LABELS.get(k, name)
                items.append((name, v['rmse_mean'], k == 'pgmpr'))
            groups.append((TEST_LABELS[tag], items))
        if groups:
            # 零均值 RBF 是对照基线，数值高达 20~36 dB，放进同一张线性坐标图
            # 会把其余方法的差异压扁。图里剔除它，表里保留完整数值。
            made['scheme_compare'] = P.fig_compare_bars(
                groups, f'{out_dir}/fig_scheme_compare.png',
                ylabel='未观测格点 RMSE (dB)',
                suptitle='五个方案与插值基线的跨户型补全精度（200 个未见过的户型）\n'
                         '注：零均值 RBF 对照项数值过大（21.6 / 35.7 dB），未在图中显示，'
                         '完整数值见报告表格')

        # 方案二 / 方案四 配对消融（唯一变量 = 布点方式）
        pair_groups = []
        for tag in ('mixed', 'scatter'):
            e = testsets.get(tag)
            if not e:
                continue
            items = []
            for k in ('unet', 'unet_scatter'):
                if k in e['methods']:
                    items.append((MODEL_LABELS[k], e['methods'][k]['rmse_mean'],
                                  k == 'unet_scatter'))
            if items:
                pair_groups.append((TEST_LABELS[tag], items))
        if pair_groups:
            made['scatter_ablation'] = P.fig_compare_bars(
                pair_groups, f'{out_dir}/fig_scatter_ablation.png',
                ylabel='未观测格点 RMSE (dB)',
                suptitle='方案四的数据消融：网络与超参完全相同，只有布点方式不同')

        # 按散落度分层（回答"方案四是不是只在最难的样本上更强"）
        mstrat = os.path.join(root, 'outputs/rfmap-results/metrics_stratified.json')
        if os.path.exists(mstrat):
            st = json.load(open(mstrat, encoding='utf-8'))
            made['strata'] = P.fig_strata(
                st, f'{out_dir}/fig_strata.png',
                labels={'unet': MODEL_LABELS['unet'],
                        'unet_scatter': MODEL_LABELS['unet_scatter'],
                        'pgmpr': MODEL_LABELS['pgmpr']},
                suptitle='按布点散落程度分层：各方案在不同外推压力下的表现')

    # ---- 5. 真实数据：两种协议下的方法对比 ----
    real_order = ['unet', 'unet_scatter', 'resunet', 'pgmpr', 'idw', 'gp']
    for proto in ('corner', 'center'):
        inp, hid = E.split_points(pts, proto)
        obs_v, obs_m = E.build_obs(inp, grid)
        fields = _pred_fields(root, grid, room, obs_v, obs_m, device)

        obs_field = np.where(obs_m > 0.5, obs_v, np.nan)
        panels = [('输入观测点（白点 = 已测）', obs_field,
                   [MarkedPoint(p, 'white') for p in inp])]
        for k in real_order:
            if k not in fields:
                continue
            title = BASELINE_LABELS.get(k, MODEL_LABELS.get(k, k))
            panels.append((title, fields[k],
                           [MarkedPoint(p, 'white') for p in inp]))
        truth_field = np.full(grid.shape, np.nan)
        for p in hid:
            j, i = grid.xy_to_idx(p.x, p.y)
            truth_field[j, i] = p.rssi
        panels.append(('隐藏点真值（仅用于评分）', truth_field,
                       [MarkedPoint(p, 'red') for p in hid]))

        made[f'real_{proto}'] = P.fig_heatmaps(
            panels, f'{out_dir}/fig_real_{proto}.png',
            P.REAL_VMIN, P.REAL_VMAX, grid, ncols=4,
            suptitle=f'真实数据 · {E.PROTOCOL_LABELS[proto]}：'
                     f'输入 {len(inp)} 点 / 隐藏 {len(hid)} 点（统一色标 '
                     f'{P.REAL_VMIN:.0f}~{P.REAL_VMAX:.0f} dBm）')

        # ---- 6. 隐藏点误差图 ----
        err_panels = []
        for k in real_order:
            if k not in fields:
                continue
            title = BASELINE_LABELS.get(k, MODEL_LABELS.get(k, k))
            errs = []
            for p in hid:
                v = E.bilinear_sample(fields[k], p.x, p.y, grid)
                if v == v:
                    errs.append((p.x_data, p.y_data, float(v) - p.rssi))
            err_panels.append((title, errs))
        made[f'err_{proto}'] = P.fig_error_points(
            err_panels, f'{out_dir}/fig_err_{proto}.png', grid, ncols=3,
            suptitle=f'隐藏点误差（标注为 预测−真值，单位 dB）· '
                     f'{E.PROTOCOL_LABELS[proto]}')

    # ---- 7. 方案五的定性案例 ----
    sim_tr = os.path.join(root, 'data/sim/train.npz')
    if os.path.exists(sim_tr):
        made['sim_samples'] = P.fig_sim_samples(
            sim_tr, f'{out_dir}/fig_sim_samples.png', grid, k=4)

    sim_te = os.path.join(root, 'data/sim/test.npz')
    if os.path.exists(sim_te):
        d = np.load(sim_te)
        idx = 0
        obs_v = d['obs_value'][idx].astype(np.float32)
        obs_m = d['obs_mask'][idx].astype(np.float32)
        r = d['room'][idx].astype(np.float32)
        tgt = np.where(r > 0.5, d['target'][idx], np.nan)
        fields = _pred_fields(root, grid, r, obs_v, obs_m, device,
                              model_names=('unet', 'unet_scatter', 'pgmpr'),
                              tta=('pgmpr',))
        obs_field = np.where(obs_m > 0.5, obs_v, np.nan)
        hidden = (r > 0.5) & (obs_m < 0.5)
        panels = [('① 真值（模型看不到）', tgt, None),
                  ('② 输入：稀疏观测点', obs_field, None)]
        for k in ('idw', 'unet', 'unet_scatter', 'pgmpr', 'pgmpr_tta'):
            if k in fields:
                nm = BASELINE_LABELS.get(k, MODEL_LABELS.get(k.split('_tta')[0], k))
                if k.endswith('_tta'):
                    nm += '（+TTA）'
                panels.append((f'③ {nm}', fields[k], None))
        err = np.abs(fields['pgmpr'] - tgt) if 'pgmpr' in fields else None
        if err is not None:
            err = np.where(hidden, err, np.nan)
        made['scheme5_case'] = P.fig_heatmaps(
            panels, f'{out_dir}/fig_scheme5_case.png',
            P.SIM_VMIN, P.SIM_VMAX, grid, ncols=4,
            suptitle='方案五定性对比（仿真测试集样本 0）：同一组稀疏观测点下各方法的整图预测')
        if err is not None:
            made['scheme5_err'] = P.fig_heatmaps(
                [('方案五 |误差|（仅未观测区）', err, None)],
                f'{out_dir}/fig_scheme5_err.png', 0.0, 12.0, grid,
                ncols=1, cmap='magma_r',
                suptitle='方案五在未观测格点上的绝对误差分布')

        # ---- 8. 跨户型样本对比（含方案五） ----
        panels2 = []
        for k in (0, 1):
            ov = d['obs_value'][k].astype(np.float32)
            om = d['obs_mask'][k].astype(np.float32)
            rr = d['room'][k].astype(np.float32)
            tt = np.where(rr > 0.5, d['target'][k], np.nan)
            panels2.append((f'样本{k} 真值', tt, None))
            f2 = _pred_fields(root, grid, rr, ov, om, device)
            for kk in ('unet', 'resunet', 'pgmpr', 'idw', 'gp'):
                if kk in f2:
                    panels2.append((f'样本{k} ' + BASELINE_LABELS.get(
                        kk, MODEL_LABELS.get(kk, kk)), f2[kk], None))
        made['sim_compare'] = P.fig_heatmaps(
            panels2, f'{out_dir}/fig_sim_compare.png', P.SIM_VMIN, P.SIM_VMAX,
            grid, ncols=6, suptitle='仿真跨户型测试：同一组输入点下各方法对比')

    # ---- 9. 五级分级的 64 x 64 像素块图 ----
    # 这里用**全部 49 点**作为观测，对应"产品最终形态"，不是评分场景。
    obs_v, obs_m = E.build_obs(pts, grid)
    X, Y = grid.centers()
    js, iss = np.nonzero(obs_m > 0.5)
    pxy = np.stack([X[js, iss], Y[js, iss]], axis=1)
    pv = obs_v[js, iss]
    idw_field = idw_predict(pxy, pv, X, Y, room > 0.5)

    lvl_panels = [('(a) 实测 49 点 IDW 插值', idw_field)]
    lvl_fields = {'idw': idw_field}
    try:
        model5, meta5, norm5 = load_trained('pgmpr', root=root, device=device)
        ft5 = E.finetune_real(model5, meta5, norm5, pts, grid, room,
                              epochs=300, device=device)
        f5 = E.nn_predict_grid(ft5, meta5, norm5, obs_v, obs_m, room, grid,
                               device)
        lvl_panels.append(('(b) 方案五 PG-MPR-UNet（49 点微调后预测）', f5))
        lvl_fields['pgmpr_ft'] = f5
    except Exception as ex:                                   # pragma: no cover
        print('  [warn] 方案五分级图跳过:', ex)

    made['levels64'] = P.fig_levels64(
        lvl_panels, f'{out_dir}/fig_levels64.png', grid,
        points=pts,
        suptitle='Wi-Fi RSSI 覆盖五级分级图：64 x 64 像素块（每块 0.2 m x 0.2 m，共 4096 块），'
                 '颜色越深信号越强',
        note='黄点为 49 个实测点位；分级取值来自插值 / 模型预测，仅用于可视化，'
             '评分一律使用留出的隐藏实测点。')

    # 等级面积统计：分级图背后到底各占多少格点，写成 JSON 供报告取数
    stat = {'edges': P.LEVEL_EDGES, 'grid': {'nx': grid.nx, 'ny': grid.ny,
                                             'cell': grid.cell},
            'names': P.LEVEL_NAMES, 'ranges': P.LEVEL_RANGES, 'sources': {}}
    for key, fld in lvl_fields.items():
        cnt = P.level_counts(fld)
        tot = int(sum(cnt))
        stat['sources'][key] = {
            'counts': cnt, 'total': tot,
            'share': [round(c / tot, 4) if tot else 0.0 for c in cnt]}
    # 顺带记下实测点自身落在哪一级（这是真实测量值，不是插值）
    obs_idx = P.level_index(np.where(obs_m > 0.5, obs_v, np.nan))
    oc = [int((obs_idx == k).sum()) for k in range(5)]
    stat['measured_points'] = {'counts': oc, 'total': int(sum(oc))}
    with open(os.path.join(root, 'outputs/rfmap-results/levels64.json'), 'w',
              encoding='utf-8') as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

    return made


if __name__ == '__main__':
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else 'cpu'
    res = make_all_figures(device=dev)
    print(json.dumps(res, indent=2, ensure_ascii=False))
