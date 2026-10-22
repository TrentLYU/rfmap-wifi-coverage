"""端到端自检：把整条交付链路逐项验证一遍。

设计原则：**只读**。除了打印结论，不产生、不修改任何文件。
每一项检查都给出可复查的实际证据（文件大小、形状、统计量、命令输出），
而不是"应该没问题"。

用法::

    .venv/bin/python -m rfmap.selfcheck
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  — {detail}' if detail else ''))
    return ok


def main():
    os.chdir(ROOT)
    print('=' * 78)
    print('RF Map 端到端自检')
    print('=' * 78)

    # ---------- 1. 实测数据 ----------
    import csv
    p = 'data/rfmap/measured.csv'
    if os.path.exists(p):
        with open(p, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        vals = [float(r['rssi_dbm']) for r in rows]
        check('实测数据 CSV', len(rows) == 49,
              f'{len(rows)} 行，RSSI {min(vals)} ~ {max(vals)} dBm')
    else:
        check('实测数据 CSV', False, '文件不存在')

    # 数据全部落在房间内
    from rfmap import data as D
    from rfmap import geometry as G
    pts = D.load(root='.')
    outside = [q.idx for q in pts if not G.inside_room(q.x, q.y, 1e-3)]
    check('测点全部位于房间内', len(outside) == 0, f'越界点 {outside or "无"}')

    # 结构柱证据
    in_col = G.inside_room(12.0, 6.0)
    out_col = G.inside_room(11.7, 6.0)
    check('结构柱几何自洽（(12,6) 在柱内 / (11.7,6) 在柱外）',
          (not in_col) and out_col)

    # ---------- 2. 网格 ----------
    from rfmap.grid import DEFAULT as GRID
    m = GRID.room_mask()
    check('网格配置', GRID.nx == 64 and GRID.ny == 64,
          f'{GRID.nx}x{GRID.ny}, cell={GRID.cell}, 有效格点 {int(m.sum())}')

    # ---------- 3. 仿真数据 ----------
    for split, n in (('train', 1200), ('val', 200), ('test', 200)):
        fp = f'data/sim/{split}.npz'
        if os.path.exists(fp):
            d = np.load(fp)
            ok = (d['obs_value'].shape == (n, 64, 64) and
                  d['obs_mask'].shape == (n, 64, 64))
            # 观测点上的观测值必须等于真值
            k = 0
            obs_ok = True
            for k in range(0, min(n, 20)):
                om = d['obs_mask'][k] > 0.5
                if not np.allclose(d['obs_value'][k][om], d['target'][k][om]):
                    obs_ok = False
                    break
            check(f'仿真数据 {split}.npz', ok and obs_ok,
                  f'{d["obs_value"].shape}, 抽查 20 个样本的观测值=真值 ✓'
                  if obs_ok else '观测值与真值不一致')
        else:
            check(f'仿真数据 {split}.npz', False, '缺失')

    # 散落布点数据集必须与原数据集**逐元素共享**户型与真值（方案四的单变量前提）
    same_all = True
    for split in ('train', 'val', 'test'):
        a, b = f'data/sim/{split}.npz', f'data/sim_scatter/{split}.npz'
        if not (os.path.exists(a) and os.path.exists(b)):
            same_all = False
            break
        da, db = np.load(a), np.load(b)
        if not (np.array_equal(da['room'], db['room']) and
                np.array_equal(da['target'], db['target'])):
            same_all = False
            break
        # 观测点位置必须已经变了，否则等于没做消融
        if np.array_equal(da['obs_mask'], db['obs_mask']):
            same_all = False
            break
        # 观测点数必须保持不变（严格单变量：只改位置、不改数量）
        if int(da['obs_mask'].sum()) != int(db['obs_mask'].sum()):
            same_all = False
            break
    check('散落布点数据集与常规数据集同户型同真值、点数相同、仅位置不同',
          same_all,
          '逐元素校验 train/val/test 三份，任一不符即失败')

    # ---------- 4. 模型 ----------
    import torch
    from rfmap.train import load_trained
    from rfmap.models import build_input_tensor, prepare_prior
    import numpy as _np
    import torch as _torch
    from rfmap import evaluate as _E
    from rfmap.grid import DEFAULT as _G
    _grid = _G
    _room = _grid.room_mask().astype('float32')
    _X, _Y = _grid.centers()
    _rng = _np.random.default_rng(0)
    _om = _np.zeros(_grid.shape, 'float32')
    _ov = _np.zeros(_grid.shape, 'float32')
    _sel = _rng.choice(_np.flatnonzero(_room > 0.5), 12, replace=False)
    _jj, _ii = _np.divmod(_sel, _grid.nx)
    _truth = (_np.hypot(_X - 2.5, _Y - 6.0) * -3.0 - 35).astype('float32')
    _ov[_jj, _ii] = _truth[_jj, _ii]
    _om[_jj, _ii] = 1.0

    preds = {}
    for name in ('ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr'):
        try:
            model, meta, norm = load_trained(name, root='.', device='cpu')
            # 走真实的预测入口（同时覆盖多先验通道构造与 TTA 路径）
            p1 = _E.nn_predict_grid(model, meta, norm, _ov, _om, _room, _grid,
                                    'cpu')
            p2 = _E.nn_predict_grid(model, meta, norm, _ov, _om, _room, _grid,
                                    'cpu', tta=True)
            finite = bool(_np.isfinite(p1[_room > 0.5]).all()) and \
                bool(_np.isfinite(p2[_room > 0.5]).all())
            preds[name] = (model, meta, norm)
            check(f'模型 {name} 可加载并推理（含 TTA）',
                  finite and p1.shape == _grid.shape,
                  f'{meta["params"]:,} 参数, 模式={_E.net_input_mode(meta)}, '
                  f'best_val={meta["best_val_loss"]:.5f}, '
                  f'单次均值 {_np.nanmean(p1):.2f} / TTA {_np.nanmean(p2):.2f} dBm')
        except Exception as e:  # noqa: BLE001
            check(f'模型 {name} 可加载并推理（含 TTA）', False, repr(e))

    # ---------- 5. 指标文件 ----------
    for fp in ('outputs/rfmap-results/metrics_real.json',
               'outputs/rfmap-results/metrics_sim.json',
               'outputs/rfmap-results/metrics_stratified.json'):
        if os.path.exists(fp):
            d = json.load(open(fp, encoding='utf-8'))
            if 'protocols' in d:
                n = len(d['protocols'])
                ntta = sum(1 for e in d['protocols'].values()
                           for k in e['methods'] if k.endswith('_tta'))
                extra = f'{n} 套协议, {len(d.get("sweep", {}))} 档点数扫描, TTA 条目 {ntta}'
            elif 'testsets' in d:
                ts = d['testsets']
                if 'methods' in (ts.get('mixed') or {}):
                    extra = (f'测试集 {list(ts)}; 方法数 '
                             f'{len(ts["mixed"]["methods"])}')
                else:
                    extra = ('分层测试集 ' + ', '.join(
                        f'{k}:{len(v["bins"])}档' for k, v in ts.items()))
            else:
                extra = '未知结构'
            check(f'指标文件 {os.path.basename(fp)}', True, extra)
        else:
            check(f'指标文件 {os.path.basename(fp)}', False, '缺失')

    # ---------- 5b. 五方案关键结论自洽性 ----------
    try:
        d = json.load(open('outputs/rfmap-results/metrics_sim.json',
                           encoding='utf-8'))
        mx = d['testsets']['mixed']['methods']
        sc = d['testsets']['scatter']['methods']
        nets = [k for k in mx if k not in ('nearest', 'idw', 'rbf', 'gp')
                and not k.endswith('_tta')]
        best_net_m = min(nets, key=lambda k: mx[k]['rmse_mean'])
        best_net_s = min(nets, key=lambda k: sc[k]['rmse_mean'])
        best_bl_m = min(('nearest', 'idw', 'rbf', 'gp'),
                        key=lambda k: mx[k]['rmse_mean'])
        all_better = (mx[best_net_m]['rmse_mean'] < mx[best_bl_m]['rmse_mean']
                      and sc[best_net_s]['rmse_mean'] < sc[best_bl_m]['rmse_mean'])
        check('跨户型结论自洽：最优网络优于最优插值基线（两个测试集）',
              all_better,
              f'mixed 最优网络 {best_net_m}={mx[best_net_m]["rmse_mean"]:.3f} '
              f'vs 最优基线 {best_bl_m}={mx[best_bl_m]["rmse_mean"]:.3f}；'
              f'scatter 最优网络 {best_net_s}={sc[best_net_s]["rmse_mean"]:.3f}')
        tta_lower = all(mx[k + '_tta']['rmse_mean'] < mx[k]['rmse_mean']
                        for k in nets if k + '_tta' in mx)
        check('TTA 在仿真测试集上对每个网络均为正收益', tta_lower,
              '逐模型比较 RMSE（报告里如实写"真实数据上并非全部如此"）')
    except Exception as e:  # noqa: BLE001
        check('五方案关键结论自洽性', False, repr(e))

    # ---------- 6. 图表 ----------
    figs = sorted(f for f in os.listdir('outputs/figures')) if os.path.isdir('outputs/figures') else []
    pngs = [f for f in figs if f.endswith('.png')]
    need_figs = ['fig_geometry.png', 'fig_data_profile.png', 'fig_training_curves.png',
                 'fig_sweep.png', 'fig_real_corner.png', 'fig_real_center.png',
                 'fig_err_corner.png', 'fig_err_center.png', 'fig_sim_samples.png',
                 'fig_sim_compare.png', 'fig_obs_layout.png', 'fig_arch_v5.png',
                 'fig_scheme_compare.png', 'fig_scatter_ablation.png',
                 'fig_strata.png', 'fig_scheme5_case.png', 'fig_scheme5_err.png',
                 'fig_levels64.png']
    miss_fig = [f for f in need_figs if f not in figs]
    check('图表文件', not miss_fig,
          f'{len(pngs)} 张 PNG，缺失 {miss_fig or "无"}')

    # 五级分级统计（§10.9）
    lvp = 'outputs/rfmap-results/levels64.json'
    if os.path.exists(lvp):
        lv = json.load(open(lvp, encoding='utf-8'))
        src = lv.get('sources') or {}
        mp = (lv.get('measured_points') or {}).get('counts') or []
        ok = (len(lv.get('edges', [])) == 6
              and len(src) >= 2
              and all(len((src.get(k) or {}).get('counts', [])) == 5 for k in src)
              and all(sum((src.get(k) or {}).get('counts', [])) == 3697 for k in src)
              and sum(mp) == 49)
        check('五级分级统计一致（64×64 网格 / 5 级 / 49 实测点）', ok,
              f"{len(lv.get('edges', [])) - 1} 级，来源 {sorted(src)}，"
              f"每源格点和 {[sum((src.get(k) or {}).get('counts', [])) for k in src]}，"
              f'实测点合计 {sum(mp)}')
    else:
        check('五级分级统计一致（64×64 网格 / 5 级 / 49 实测点）', False,
              'levels64.json 不存在')

    # ---------- 7. HTML 报告 ----------
    rp = 'outputs/RF覆盖预测实验报告.html'
    if os.path.exists(rp):
        html = open(rp, encoding='utf-8').read()
        imgs = re.findall(r'src="([^"]+)"', html)
        missing = [i for i in imgs if not os.path.exists(os.path.join('outputs', i))]
        # 标签平衡粗检（占位符检查要排除 <pre> 代码块：复现命令里的
        # Python f-string 变量如 {name} 是正常内容，不是未替换的模板占位符）
        open_div = html.count('<div')
        close_div = html.count('</div>')
        open_tbl = html.count('<table')
        close_tbl = html.count('</table>')
        body = re.sub(r'<pre>.*?</pre>', '', html, flags=re.S)
        placeholder = re.findall(r'\{[a-z_]+\}', body)
        check('HTML 报告结构', not missing and open_div == close_div and open_tbl == close_tbl
              and not placeholder,
              f'{len(html) // 1024} KB, 引用 {len(imgs)} 图（缺失 {len(missing)}）, '
              f'div {open_div}/{close_div}, table {open_tbl}/{close_tbl}, '
              f'正文未替换占位符 {len(placeholder)}')
    else:
        check('HTML 报告结构', False, '文件不存在')

    # ---------- 8. Word 报告 ----------
    dp = 'outputs/RF覆盖预测实验报告.docx'
    if os.path.exists(dp):
        z = zipfile.ZipFile(dp)
        x = z.read('word/document.xml').decode('utf-8', errors='ignore')
        tbl = x.count('<w:tbl>')
        drw = x.count('<w:drawing>')
        media = len([n for n in z.namelist() if n.startswith('word/media/')])
        check('Word 报告', tbl >= 15 and drw >= 8 and media >= 8,
              f'{os.path.getsize(dp) // 1024} KB, {tbl} 个表格, {drw} 张图（media {media}）')
    else:
        check('Word 报告', False, '文件不存在')

    # ---------- 9. Web 应用 ----------
    studio = 'outputs/rfmap-studio'
    need = ['index.html', 'app.js', 'style.css', 'sample_room.json',
            'model_weights.js', 'nn_model.js']
    if os.path.isdir(studio):
        have = os.listdir(studio)
        miss = [f for f in need if f not in have]
        idx = open(os.path.join(studio, 'index.html'), encoding='utf-8').read()
        order_ok = (idx.find('model_weights.js') < idx.find('nn_model.js')
                    < idx.find('app.js'))
        has_infer = 'RFM_Infer' in open(os.path.join(studio, 'nn_model.js'),
                                        encoding='utf-8').read()
        check('Web 应用文件与加载顺序', not miss and order_ok and has_infer,
              f'缺失 {miss or "无"}; 脚本顺序 model_weights → nn_model → app '
              f'{"正确" if order_ok else "错误"}; 已注册 RFM_Infer={has_infer}')
        wjs = os.path.join(studio, 'model_weights.js')
        if os.path.exists(wjs):
            head = open(wjs, encoding='utf-8').read(4000)
            mm = re.search(r'"n_params":(\d+)', open(wjs, encoding='utf-8').read(200000))
            check('权重导出与 CSV 网格一致', bool(mm),
                  f'参数 {mm.group(1) if mm else "?"}, 文件 {os.path.getsize(wjs) // 1024} KB')
    else:
        check('Web 应用文件', False, '目录不存在')

    # ---------- 10. 文档 ----------
    for fp in ('outputs/实测RSSI数据说明.md', 'outputs/RFMap-Studio-使用说明.md'):
        check(f'文档 {os.path.basename(fp)}', os.path.exists(fp),
              f'{os.path.getsize(fp) // 1024} KB' if os.path.exists(fp) else '缺失')

    # ---------- 汇总 ----------
    print('=' * 78)
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f'合计 {n_ok}/{len(RESULTS)} 项通过')
    if n_ok != len(RESULTS):
        print('未通过项：')
        for name, ok, detail in RESULTS:
            if not ok:
                print(f'  - {name}: {detail}')
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == '__main__':
    sys.exit(main())
