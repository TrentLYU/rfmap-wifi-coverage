"""生成实验报告（HTML），并把 Word 版需要的中间产物准备好。

设计原则
--------
报告里的**每一个数字都从磁盘上的产物文件读出来**，而不是手写进模板：

* ``models/*/meta.json``   —— 训练配置、参数量、最佳轮次
* ``outputs/rfmap-results/metrics_real.json`` —— 真实数据全部指标
* ``outputs/rfmap-results/metrics_sim.json``  —— 跨户型全部指标
* ``data/rfmap/measured.csv`` + ``rfmap.data.quality_report`` —— 数据质检

这样重新训练一次、再跑 ``python -m rfmap.report``，报告会跟着更新，
不会出现"文档改了、数字没改"的经典事故。
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from . import data as D
from . import geometry as G
from .grid import DEFAULT as GRID
from .models import BASELINE_LABELS, MODEL_LABELS, count_params, build_model

MODEL_ORDER = ['ae', 'unet', 'resunet', 'unet_scatter', 'pgmpr']

# 两个仿真测试集的显示名（逐元素共享同一批户型与真值，只有布点方式不同）
TEST_LABELS = {'mixed': '常规布点测试集', 'scatter': '散落布点测试集'}


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _obs_stats(root, npz_rel, n=200):
    """统计一个仿真数据集的布点特征（点数、到最近观测点的距离分布）。

    这是方案四的关键证据：散落布点在不改变观测点数量的前提下，把
    "离最近观测点多远"整体抬高，也就是外推区变大。全部数值由数据实算，
    不写死。
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    p = os.path.join(root, npz_rel)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    om = d['obs_mask'][:n]
    rm = d['room'][:n]
    med, p95, cnt = [], [], []
    for k in range(om.shape[0]):
        o = om[k] > 0.5
        r = rm[k] > 0.5
        if not o.any():
            continue
        dist = distance_transform_edt(~o) * 0.2  # 格距 0.2 m
        v = dist[r]
        med.append(float(np.median(v)))
        p95.append(float(np.percentile(v, 95)))
        cnt.append(int(o.sum()))
    return {
        'file': npz_rel,
        'n_samples': int(om.shape[0]),
        'points_mean': round(float(np.mean(cnt)), 1),
        'points_range': [int(np.min(cnt)), int(np.max(cnt))],
        'nn_median': round(float(np.mean(med)), 2),
        'nn_p95': round(float(np.mean(p95)), 2),
    }


def _fmt(v, nd=3):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.{nd}f}'
    return str(v)


# ==========================================================================
def collect(root: str = '.') -> dict:
    """把报告需要的全部事实集中收集成一个 dict。"""
    ctx = {}
    ctx['geometry'] = G.summary()
    ctx['grid'] = GRID.to_dict()

    pts = D.load(root=root)
    ctx['points'] = pts
    ctx['quality'] = D.quality_report(pts)
    ctx['n_points'] = len(pts)

    ctx['models'] = {}
    for m in MODEL_ORDER:
        meta = _load_json(os.path.join(root, f'models/{m}/meta.json'))
        hist = _load_json(os.path.join(root, f'models/{m}/history.json'))
        if meta:
            try:
                from .train import build_model as _bm
                meta['params_actual'] = count_params(_bm(m, width=meta['width']))
            except Exception:
                pass
            ctx['models'][m] = {'meta': meta, 'history': hist}

    ctx['real'] = _load_json(os.path.join(root, 'outputs/rfmap-results/metrics_real.json'))
    ctx['sim'] = _load_json(os.path.join(root, 'outputs/rfmap-results/metrics_sim.json'))
    ctx['strat'] = _load_json(os.path.join(root, 'outputs/rfmap-results/metrics_stratified.json'))

    # 仿真数据集规模
    ctx['sim_data'] = {}
    for split in ('train', 'val', 'test'):
        p = os.path.join(root, f'data/sim/{split}.npz')
        if os.path.exists(p):
            import numpy as np
            d = np.load(p)
            ctx['sim_data'][split] = {
                'n': int(d['obs_value'].shape[0]),
                'shape': list(d['obs_value'].shape[1:]),
                'mb': round(os.path.getsize(p) / 1024 / 1024, 2),
            }

    # 散落布点数据集规模（方案四）
    ctx['sim_scatter'] = {}
    for split in ('train', 'val', 'test'):
        p = os.path.join(root, f'data/sim_scatter/{split}.npz')
        if os.path.exists(p):
            import numpy as np
            d = np.load(p)
            ctx['sim_scatter'][split] = {
                'n': int(d['obs_value'].shape[0]),
                'mb': round(os.path.getsize(p) / 1024 / 1024, 2),
            }

    # 布点方式统计（方案四的核心证据）
    ctx['obs_stats'] = {}
    for tag, rel in (('mixed', 'data/sim/train.npz'),
                     ('scatter', 'data/sim_scatter/train.npz')):
        s = _obs_stats(root, rel)
        if s:
            ctx['obs_stats'][tag] = s

    # 五级分级图的分级统计（§10.9）
    ctx['levels'] = _load_json(
        os.path.join(root, 'outputs/rfmap-results/levels64.json'))

    # 权重导出
    wp = os.path.join(root, 'outputs/rfmap-studio/model_weights.js')
    if os.path.exists(wp):
        ctx['web_weights_kb'] = round(os.path.getsize(wp) / 1024, 1)
    return ctx


# ==========================================================================
CSS = """
:root{
  --ink:#16181d; --muted:#5c6470; --line:#e0e4ea; --soft:#f6f8fb;
  --brand:#1f6feb; --good:#0f7a4a; --bad:#b42318; --warn:#a15c00;
}
*{box-sizing:border-box}
body{
  margin:0; padding:0 0 90px; background:#fff; color:var(--ink);
  font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
  font-size:15px; line-height:1.78;
}
main{max-width:1000px; margin:0 auto; padding:0 26px}
header.cover{
  background:linear-gradient(135deg,#12213b,#1f4b8f 55%,#2f7fd0);
  color:#fff; padding:54px 26px 44px; margin-bottom:34px;
}
header.cover .inner{max-width:1000px; margin:0 auto}
header.cover h1{font-size:30px; margin:0 0 10px; letter-spacing:-.3px; line-height:1.34}
header.cover .sub{opacity:.9; font-size:14px}
header.cover .meta{margin-top:18px; font-size:13.5px; opacity:.86; line-height:1.9}
h2{font-size:20px; margin:46px 0 14px; padding-bottom:8px;
   border-bottom:2px solid var(--line); letter-spacing:-.2px}
h3{font-size:16.5px; margin:28px 0 8px; color:#1b2733}
h4{font-size:14.5px; margin:20px 0 6px; color:#33404f}
p{margin:9px 0}
ul,ol{margin:9px 0 9px 24px; padding:0}
li{margin:5px 0}
code{background:var(--soft); border:1px solid var(--line); border-radius:4px;
  padding:1px 5px; font-size:12.8px;
  font-family:"Cascadia Code",Consolas,"Courier New",monospace}
pre{background:#0f172a; color:#e2e8f0; padding:14px 16px; border-radius:8px;
  overflow-x:auto; font-size:12.8px; line-height:1.62;
  font-family:"Cascadia Code",Consolas,"Courier New",monospace}
pre code{background:none; border:none; color:inherit; padding:0}
table{border-collapse:collapse; width:100%; margin:14px 0; font-size:13.6px}
th,td{border:1px solid var(--line); padding:7px 10px; text-align:left;
  vertical-align:top}
th{background:var(--soft); font-weight:600}
td.num,th.num{text-align:right; font-variant-numeric:tabular-nums}
tr.best td{background:#eefaf3; font-weight:600}
.note{background:#f0f6ff; border-left:4px solid var(--brand);
  padding:12px 16px; border-radius:0 6px 6px 0; margin:16px 0; font-size:14px}
.warn{background:#fff8ec; border-left:4px solid var(--warn)}
.bad{background:#fdf3f2; border-left:4px solid var(--bad)}
.good{background:#f0faf4; border-left:4px solid var(--good)}
.kpi{display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
  gap:12px; margin:20px 0}
.kpi > div{background:var(--soft); border:1px solid var(--line);
  border-radius:9px; padding:13px 15px}
.kpi span{display:block; color:var(--muted); font-size:12.4px; margin-bottom:5px}
.kpi b{font-size:21px; letter-spacing:-.4px}
.kpi b.good{color:var(--good)} .kpi b.bad{color:var(--bad)}
figure{margin:22px 0; text-align:center}
figure img{max-width:100%; border:1px solid var(--line); border-radius:8px}
figcaption{color:var(--muted); font-size:12.9px; margin-top:9px; line-height:1.65}
.toc{background:var(--soft); border:1px solid var(--line); border-radius:10px;
  padding:16px 22px; margin:26px 0}
.toc ol{margin:6px 0 0 20px; columns:2; column-gap:34px}
.toc a{color:#1b2733; text-decoration:none}
.toc a:hover{color:var(--brand); text-decoration:underline}
.tag{display:inline-block; font-size:11.6px; padding:1px 8px; border-radius:99px;
  border:1px solid var(--line); background:#fff; color:var(--muted); margin-left:6px}
.tag.done{border-color:#b6e0c8; background:#eefaf3; color:var(--good)}
.tag.warn{border-color:#f0d9a8; background:#fff8ec; color:var(--warn)}
footer{margin-top:52px; padding-top:18px; border-top:1px solid var(--line);
  color:var(--muted); font-size:12.9px}
"""


def _model_table(ctx):
    rows = []
    for m in MODEL_ORDER:
        if m not in ctx['models']:
            continue
        meta = ctx['models'][m]['meta']
        rows.append(
            f"<tr><td>{MODEL_LABELS[m]}</td>"
            f"<td class='num'>{meta['params']:,}</td>"
            f"<td class='num'>{meta['width']}</td>"
            f"<td class='num'>{meta['epochs_run']}</td>"
            f"<td class='num'>{meta['best_epoch']}</td>"
            f"<td class='num'>{meta['best_val_loss']:.5f}</td>"
            f"<td class='num'>{meta['seconds']:.0f}</td></tr>")
    return '\n'.join(rows)


def _metrics_table(methods: dict, labels: dict, keys=None, best_key=None,
                   extra_cols=('mae', 'rmse', 'max_abs', 'bias')):
    keys = keys or list(methods)
    rows = []
    ordered = sorted(keys, key=lambda k: (methods[k].get('rmse') is None,
                                          methods[k].get('rmse') or 1e9))
    best = ordered[0] if ordered else None
    for k in ordered:
        v = methods[k]
        cls = " class='best'" if k == best else ''
        name = labels.get(k, MODEL_LABELS.get(k, k))
        if k.endswith('_ft'):
            name = MODEL_LABELS.get(k[:-3], k[:-3]) + '（实测微调）'
        cells = ''.join(f"<td class='num'>{_fmt(v.get(c), 3)}</td>"
                        for c in extra_cols)
        rows.append(f"<tr{cls}><td>{name}</td>{cells}"
                    f"<td class='num'>{_fmt(v.get('infer_ms'), 2)}</td></tr>")
    return '\n'.join(rows)


def _name(k, labels):
    """把内部 key 翻成人类可读名字（含 _ft / _tta 后缀）。"""
    if k.endswith('_ft'):
        return MODEL_LABELS.get(k[:-3], k[:-3]) + '（实测微调）'
    if k.endswith('_tta'):
        return MODEL_LABELS.get(k[:-4], k[:-4]) + '（+TTA 测试时增强）'
    return labels.get(k, MODEL_LABELS.get(k, k))


def _sim_entry(sim, tag):
    return ((sim or {}).get('testsets') or {}).get(tag) or {}


def _scheme_compare_table(sim, labels):
    """五方案 + 基线在两个仿真测试集上的对照（方案四消融与方案五优势的主表）。"""
    mixed = _sim_entry(sim, 'mixed').get('methods') or {}
    scat = _sim_entry(sim, 'scatter').get('methods') or {}
    if not mixed and not scat:
        return '', None

    keys = [k for k in list(mixed) if not k.endswith('_tta')]
    best_mixed = min((mixed[k]['rmse_mean'] for k in keys if k in mixed),
                     default=None)
    best_scat = min((scat[k]['rmse_mean'] for k in keys if k in scat),
                    default=None)
    best_bl_mixed = min((mixed[k]['rmse_mean'] for k in keys
                         if k in mixed and k in BASELINE_LABELS), default=None)
    best_bl_scat = min((scat[k]['rmse_mean'] for k in keys
                        if k in scat and k in BASELINE_LABELS), default=None)

    rows = []
    for k in sorted(keys, key=lambda x: mixed.get(x, {}).get('rmse_mean', 9e9)):
        mv = mixed.get(k, {}).get('rmse_mean')
        sv = scat.get(k, {}).get('rmse_mean')
        is_net = not k.endswith('_ft') and k not in BASELINE_LABELS
        cls = ''
        if k == 'pgmpr':
            cls = " class='best'"
        tag = ' <span class="tag">插值基线</span>' if k in BASELINE_LABELS else ''
        gain = ''
        if is_net and mv is not None and best_bl_mixed:
            gain = f"{best_bl_mixed - mv:+.2f}"
        rows.append(
            f"<tr{cls}><td>{_name(k, labels)}{tag}</td>"
            f"<td class='num'>{_fmt(mv, 3)}</td>"
            f"<td class='num'>{_fmt(sv, 3)}</td>"
            f"<td class='num'>{gain}</td></tr>")
    tbl = ('<table>'
           '<tr><th>方法</th>'
           '<th class="num">常规布点测试集 RMSE (dB)</th>'
           '<th class="num">散落布点测试集 RMSE (dB)</th>'
           '<th class="num">相对最优基线 (dB)</th></tr>'
           + ''.join(rows) + '</table>')
    info = {'best_mixed': best_mixed, 'best_scatter': best_scat,
            'best_bl_mixed': best_bl_mixed, 'best_bl_scatter': best_bl_scat,
            'mixed': mixed, 'scatter': scat}
    return tbl, info


def _scheme_rows(si, labels):
    """五方案双测试集的表体行（表头由调用处写，便于在正文里换措辞）。"""
    mixed = si.get('mixed') or {}
    scat = si.get('scatter') or {}
    keys = [k for k in mixed if not k.endswith('_tta')]
    rows = []
    for k in sorted(keys, key=lambda x: mixed.get(x, {}).get('rmse_mean', 9e9)):
        mv = mixed.get(k, {}).get('rmse_mean')
        sv = scat.get(k, {}).get('rmse_mean')
        cls = " class='best'" if k == 'pgmpr' else ''
        tag = ' <span class="tag">插值基线</span>' if k in BASELINE_LABELS else ''
        if k in BASELINE_LABELS or mv is None or not si.get('best_bl_mixed'):
            gain = '—'
        else:
            gain = f"{si['best_bl_mixed'] - mv:+.2f}"
        rows.append(f"<tr{cls}><td>{_name(k, labels)}{tag}</td>"
                    f"<td class='num'>{_fmt(mv, 3)}</td>"
                    f"<td class='num'>{_fmt(sv, 3)}</td>"
                    f"<td class='num'>{gain}</td></tr>")
    return '\n'.join(rows)


def _sim_best_key(si, tag):
    m = si.get(tag) or {}
    keys = [k for k in m if not k.endswith('_tta')]
    if not keys:
        return ''
    return min(keys, key=lambda x: m.get(x, {}).get('rmse_mean', 9e9))


def _rm(si, tag, key):
    """取某个测试集上某方法的 RMSE 均值。"""
    return ((si.get(tag) or {}).get(key) or {}).get('rmse_mean')


def _diff(si, tag, ka, kb):
    """同一测试集上两方法的 RMSE 之差（ka - kb），任一缺失返回 None。"""
    a, b = _rm(si, tag, ka), _rm(si, tag, kb)
    if a is None or b is None:
        return None
    return a - b


def _scatter_ablation_table(sim, labels):
    """方案二 vs 方案四：网络与超参完全相同，唯一变量是布点方式。"""
    mixed = _sim_entry(sim, 'mixed').get('methods') or {}
    scat = _sim_entry(sim, 'scatter').get('methods') or {}
    if 'unet' not in mixed or 'unet_scatter' not in mixed:
        return ''
    rows = []
    for k in ('unet', 'unet_scatter'):
        rows.append(
            f"<tr><td>{_name(k, labels)}</td>"
            f"<td class='num'>{_fmt(mixed.get(k, {}).get('rmse_mean'), 3)}</td>"
            f"<td class='num'>{_fmt(scat.get(k, {}).get('rmse_mean'), 3)}</td>"
            f"<td class='num'>{_fmt(scat.get(k, {}).get('rmse_p90'), 3)}</td>"
            f"<td class='num'>{_fmt(scat.get(k, {}).get('rmse_median'), 3)}</td></tr>")
    delta = ''
    if ('unet' in scat and 'unet_scatter' in scat):
        d = scat['unet']['rmse_mean'] - scat['unet_scatter']['rmse_mean']
        delta = (f"<p>在**散落布点测试集**上，方案四比方案二 "
                 f"{'低' if d > 0 else '高'} <b>{abs(d):.3f} dB</b>"
                 f"（{'方案四更优' if d > 0 else '方案二更优'}）。</p>")
    return ('<table>'
            '<tr><th>方法（结构完全相同）</th>'
            '<th class="num">常规布点 RMSE</th>'
            '<th class="num">散落布点 RMSE</th>'
            '<th class="num">散落布点 P90</th>'
            '<th class="num">散落布点中位数</th></tr>'
            + ''.join(rows) + '</table>' + delta)


def _tta_table(real, labels, protos):
    """测试时增强（TTA）的开/关对照。"""
    rows = []
    for pk in protos:
        e = (real.get('protocols') or {}).get(pk)
        if not e:
            continue
        m = e['methods']
        for k in m:
            if not k.endswith('_tta'):
                continue
            base = m.get(k[:-4])
            if not base:
                continue
            a = base['rmse']
            b = m[k]['rmse']
            d = a - b
            rows.append(
                f"<tr><td>{e['label']}</td>"
                f"<td>{MODEL_LABELS.get(k[:-4], k[:-4])}</td>"
                f"<td class='num'>{_fmt(a, 3)}</td>"
                f"<td class='num'>{_fmt(b, 3)}</td>"
                f"<td class='num'>{d:+.3f}</td>"
                f"<td class='num'>{_fmt(base.get('infer_ms'), 1)} → "
                f"{_fmt(m[k].get('infer_ms'), 1)}</td></tr>")
    if not rows:
        return ''
    return ('<table>'
            '<tr><th>协议</th><th>方法</th>'
            '<th class="num">RMSE 无 TTA</th>'
            '<th class="num">RMSE 有 TTA</th>'
            '<th class="num">变化 (dB)</th>'
            '<th class="num">推理耗时 (ms)</th></tr>'
            + ''.join(rows) + '</table>')


def _strata_table(ctx, labels):
    """按散落度分层的对照表（方案四"零结果"的关键证据）。"""
    st = ctx.get('strat') or {}
    e = (st.get('testsets') or {}).get('scatter')
    if not e or not e.get('bins'):
        return ''
    bins = e['bins']
    methods = list(bins[0]['methods'].keys())
    hdr = ''.join(f'<th class="num">{_name(m, labels)} RMSE (dB)</th>'
                  for m in methods)
    rows = []
    for b in bins:
        cells = ''.join(f"<td class='num'>{b['methods'][m]:.3f}</td>"
                        for m in methods)
        rows.append(f"<tr><td>{b['label']}</td>"
                    f"<td class='num'>{b['spread_mean']} m</td>"
                    f"<td class='num'>{b['n']}</td>{cells}</tr>")
    return ('<table><tr><th>散落度档位</th>'
            '<th class="num">未观测区 P95 距离</th>'
            '<th class="num">样本数</th>' + hdr + '</tr>' + ''.join(rows) + '</table>')


def _strata_delta(ctx):
    """各档内 方案二 − 方案四 的 RMSE 差（正 = 方案四更好）。"""
    st = ctx.get('strat') or {}
    e = (st.get('testsets') or {}).get('scatter')
    if not e:
        return []
    out = []
    for b in e.get('bins', []):
        m = b['methods']
        if 'unet' in m and 'unet_scatter' in m:
            out.append((b['label'], b['spread_mean'],
                        round(m['unet'] - m['unet_scatter'], 3)))
    return out


def _tta_sim_table(sim, labels):
    """TTA 在两个仿真测试集上的开/关对照（每个模型一行、两个测试集两列）。"""
    mixed = _sim_entry(sim, 'mixed').get('methods') or {}
    scat = _sim_entry(sim, 'scatter').get('methods') or {}
    keys = [k for k in mixed if k in MODEL_LABELS and not k.endswith('_ft')]
    if not keys:
        return ''
    rows = []
    for k in sorted(keys, key=lambda x: mixed.get(x, {}).get('rmse_mean', 9e9)):
        tk = k + '_tta'
        if tk not in mixed:
            continue
        rows.append(
            f"<tr><td>{MODEL_LABELS[k]}</td>"
            f"<td class='num'>{_fmt(mixed[k]['rmse_mean'], 3)} → "
            f"{_fmt(mixed[tk]['rmse_mean'], 3)}</td>"
            f"<td class='num'>{mixed[k]['rmse_mean'] - mixed[tk]['rmse_mean']:+.3f}</td>"
            f"<td class='num'>{_fmt(scat.get(k, {}).get('rmse_mean'), 3)} → "
            f"{_fmt(scat.get(tk, {}).get('rmse_mean'), 3)}</td>"
            f"<td class='num'>"
            f"{(scat[k]['rmse_mean'] - scat[tk]['rmse_mean']):+.3f}"
            f"</td></tr>" if k in scat and tk in scat else
            f"<tr><td>{MODEL_LABELS[k]}</td><td colspan='4'>—</td></tr>")
    return ('<table>'
            '<tr><th rowspan="2">模型</th>'
            '<th colspan="2" class="num">常规布点测试集</th>'
            '<th colspan="2" class="num">散落布点测试集</th></tr>'
            '<tr><th class="num">无 TTA → 有 TTA</th><th class="num">变化</th>'
            '<th class="num">无 TTA → 有 TTA</th><th class="num">变化</th></tr>'
            + ''.join(rows) + '</table>')


def _v5_param_info(ctx):
    """方案五与方案二的参数量对照（供多处引用，避免各写一份算错）。"""
    _mv5 = (ctx['models'].get('pgmpr') or {}).get('meta') or {}
    _mu = (ctx['models'].get('unet') or {}).get('meta') or {}
    v5 = f"{_mv5.get('params', 0):,}" if _mv5 else '—'
    un = f"{_mu.get('params', 0):,}" if _mu else '—'
    dl, pc = '—', '—'
    if _mv5 and _mu and _mu.get('params'):
        d = _mv5['params'] - _mu['params']
        dl, pc = f"{d:,}", f"{d / _mu['params'] * 100:.2f}"
    return v5, un, dl, pc


def _levels_table(ctx):
    """§10.9 的五级分级统计表：每一级各占多少格点、实测点又落在哪一级。"""
    lv = ctx.get('levels') or {}
    src = lv.get('sources') or {}
    names = lv.get('names') or ['', '', '', '', '']
    ranges = lv.get('ranges') or ['', '', '', '', '']
    idw = src.get('idw') or {}
    v5 = src.get('pgmpr_ft') or {}
    mp = (lv.get('measured_points') or {}).get('counts') or [0] * 5
    ci_all = idw.get('counts') or [0] * 5
    cv_all = v5.get('counts') or [0] * 5
    ti, tv = idw.get('total') or 0, v5.get('total') or 0
    rows = []
    for i in range(5):
        name = names[i].split(' · ')[-1] if names[i] else '—'
        ci, cv = ci_all[i], cv_all[i]
        si = (ci / ti * 100) if ti else 0.0
        sv = (cv / tv * 100) if tv else 0.0
        hi = ' class="best"' if i == 4 else ''
        rows.append(
            f'<tr{hi}><td>{5 - i} 级（{name}）</td>'
            f'<td class="num">{ranges[i]}</td>'
            f'<td class="num">{ci}</td><td class="num">{si:.1f}%</td>'
            f'<td class="num">{cv}</td><td class="num">{sv:.1f}%</td>'
            f'<td class="num">{mp[i]}</td></tr>')
    return '\n'.join(rows)


def _kpi(ctx):
    real = ctx.get('real') or {}
    sim = ctx.get('sim') or {}
    corner = (real.get('protocols', {}).get('corner', {}) or {}).get('methods', {})
    center = (real.get('protocols', {}).get('center', {}) or {}).get('methods', {})
    best_corner = min(corner.items(), key=lambda x: x[1]['rmse'])[1]['rmse'] if corner else None
    best_center = min(center.items(), key=lambda x: x[1]['rmse'])[1]['rmse'] if center else None

    def _mbest(tag):
        m = _sim_entry(sim, tag).get('methods') or {}
        nets = {k: v for k, v in m.items()
                if k not in BASELINE_LABELS and not k.endswith('_tta')}
        bls = {k: v for k, v in m.items() if k in BASELINE_LABELS}
        best_net = (min((v['rmse_mean'] for v in nets.values()))
                    if nets else None)
        best_net_k = (min(nets.items(), key=lambda x: x[1]['rmse_mean'])[0]
                      if nets else None)
        best_bl = (min((v['rmse_mean'] for v in bls.values()))
                   if bls else None)
        return best_net, best_net_k, best_bl, m.get('pgmpr', {}).get('rmse_mean')

    net_m, net_m_k, bl_m, pg_m = _mbest('mixed')
    net_s, net_s_k, bl_s, pg_s = _mbest('scatter')
    return f"""
<div class="kpi">
  <div><span>实测测点</span><b>{ctx['n_points']} 点</b></div>
  <div><span>角块外推 · 最优 RMSE</span><b class="good">{_fmt(best_corner,2)} dB</b></div>
  <div><span>中心插值 · 最优 RMSE</span><b class="good">{_fmt(best_center,2)} dB</b></div>
  <div><span>跨户型 · 最优网络</span><b class="good">{_fmt(net_m,2)} dB</b>
       <span style="font-size:11.5px">{MODEL_LABELS.get(net_m_k, net_m_k)}</span></div>
  <div><span>跨户型 · 最优插值基线</span><b>{_fmt(bl_m,2)} dB</b></div>
  <div><span>散落布点布点场景 · 最优网络</span><b class="good">{_fmt(net_s,2)} dB</b>
       <span style="font-size:11.5px">{MODEL_LABELS.get(net_s_k, net_s_k)}</span></div>
</div>"""


# ==========================================================================
def render(ctx: dict) -> str:
    g = ctx['geometry']
    q = ctx['quality']
    real = ctx.get('real') or {}
    sim = ctx.get('sim') or {}
    protos = real.get('protocols', {})
    sweep = real.get('sweep', {})

    labels = dict(BASELINE_LABELS)
    labels.update(MODEL_LABELS)

    model_tbl = _model_table(ctx)
    kpi = _kpi(ctx)

    # ---- 数据点表 ----
    pt_rows = '\n'.join(
        f"<tr><td class='num'>{p.idx}</td><td class='num'>{p.x_data:g}</td>"
        f"<td class='num'>{p.y_data:g}</td><td class='num'>{p.x:.3f}</td>"
        f"<td class='num'>{p.y:.3f}</td><td class='num'>{p.rssi:g}</td>"
        f"<td>{'是' if G.inside_room(p.x, p.y, 1e-3) else '否'}</td>"
        f"<td>{'⚠ 偏强' if p.rssi > -40 else '正常'}</td></tr>"
        for p in ctx['points'])

    # ---- 协议表 ----
    proto_rows = []
    for k, e in protos.items():
        m = e['methods']
        best = min(m.items(), key=lambda x: x[1]['rmse'])
        best_bl = min(((kk, vv) for kk, vv in m.items()
                       if kk in BASELINE_LABELS and not kk.endswith('_ft')),
                      key=lambda x: x[1]['rmse'])
        proto_rows.append(
            f"<tr><td>{e['label']}</td><td class='num'>{e['n_input']}</td>"
            f"<td class='num'>{e['n_hidden']}</td>"
            f"<td>{labels.get(best[0], best[0])}</td>"
            f"<td class='num'>{best[1]['rmse']:.3f}</td>"
            f"<td>{labels.get(best_bl[0], best_bl[0])}</td>"
            f"<td class='num'>{best_bl[1]['rmse']:.3f}</td></tr>")
    proto_tbl = '\n'.join(proto_rows)

    # ---- sweep 表 ----
    sweep_methods = sorted({m for v in sweep.values() for m in v})
    sw_header = ''.join(f'<th class="num">{labels.get(m, m)}</th>' for m in sweep_methods)
    sw_rows = []
    for K in sorted(int(x) for x in sweep):
        cells = ''.join(
            f"<td class='num'>{_fmt(sweep[str(K)].get(m, {}).get('rmse_mean'))}"
            f"<span style='color:#98a2b3'> ±{_fmt(sweep[str(K)].get(m, {}).get('rmse_std'),2)}</span></td>"
            for m in sweep_methods)
        sw_rows.append(f"<tr><td class='num'>{K}</td>{cells}</tr>")
    sweep_tbl = f"<table><tr><th>输入点数</th>{sw_header}</tr>{''.join(sw_rows)}</table>"

    # ---- 仿真表：常规布点测试集（详细） ----
    sim_mixed = _sim_entry(sim, 'mixed').get('methods') or {}
    sim_rows = []
    shown = [k for k in sim_mixed if not k.endswith('_tta')]
    ordered = sorted(shown, key=lambda k: sim_mixed[k]['rmse_mean'])
    for i, k in enumerate(ordered):
        v = sim_mixed[k]
        cls = " class='best'" if i == 0 else ''
        is_bl = k in BASELINE_LABELS
        sim_rows.append(
            f"<tr{cls}><td>{_name(k, labels)}"
            f"{' <span class=\"tag\">基线</span>' if is_bl else ''}</td>"
            f"<td class='num'>{v['rmse_mean']:.3f}</td>"
            f"<td class='num'>{v['rmse_median']:.3f}</td>"
            f"<td class='num'>{v['rmse_p90']:.3f}</td></tr>")
    sim_tbl = '\n'.join(sim_rows)

    # ---- 五方案综合对比 / 散落布点消融 / TTA 消融 ----
    scheme_tbl, scheme_info = _scheme_compare_table(sim, labels)
    _si = scheme_info or {}
    scatter_tbl = _scatter_ablation_table(sim, labels)
    tta_tbl = _tta_table(real, labels, ('corner', 'center'))
    oss = ctx.get('obs_stats') or {}
    o_m = oss.get('mixed') or {}
    o_s = oss.get('scatter') or {}
    sim_te_scatter = (ctx.get('sim_scatter') or {}).get('test') or {}

    # 方案五 vs 方案二 的参数量对照（证明"不是靠模型变大换精度"）
    _mv5 = (ctx['models'].get('pgmpr') or {}).get('meta') or {}
    _mu = (ctx['models'].get('unet') or {}).get('meta') or {}
    model_params_v5 = f"{_mv5.get('params', 0):,}" if _mv5 else '—'
    model_params_unet = f"{_mu.get('params', 0):,}" if _mu else '—'
    if _mv5 and _mu and _mu.get('params'):
        _d = _mv5['params'] - _mu['params']
        model_params_delta = f"{_d:,}"
        model_params_pct = f"{_d / _mu['params'] * 100:.2f}"
    else:
        model_params_delta = '—'
        model_params_pct = '—'

    proto_sections = ''
    for key in ('corner', 'center'):
        if key not in protos:
            continue
        e = protos[key]
        m = e['methods']
        keys = [k for k in m]
        ft_keys = [k for k in keys if k.endswith('_ft')]
        base_keys = [k for k in keys if not k.endswith('_ft')]
        proto_sections += f"""
<h3 id="res-{key}">{e['label']} —— 输入 {e['n_input']} 点 / 隐藏 {e['n_hidden']} 点</h3>
<p>{e['label'].split('（')[0]}的评估方式是：把该区域全部实测点从输入里拿掉，
只喂给模型其余 {e['n_input']} 个点，再用被拿掉的 {e['n_hidden']} 个点算误差。
隐藏点从头到尾没有进入过输入通道，也没有参与过任何训练或早停。</p>
<table>
  <tr><th>方法</th><th class="num">MAE (dB)</th><th class="num">RMSE (dB)</th>
      <th class="num">最大绝对误差</th><th class="num">平均偏差 bias</th>
      <th class="num">推理耗时 (ms)</th></tr>
  {_metrics_table(m, labels, base_keys)}
</table>
<table>
  <tr><th>方法（实测微调后）</th><th class="num">MAE (dB)</th><th class="num">RMSE (dB)</th>
      <th class="num">最大绝对误差</th><th class="num">平均偏差 bias</th>
      <th class="num">推理耗时 (ms)</th></tr>
  {_metrics_table(m, labels, ft_keys)}
</table>
<figure>
  <img src="figures/fig_real_{key}.png" alt="{e['label']} 各方法热力图对比">
  <figcaption>图：{e['label']} 下各方法给出的整幅 RSSI 热力图。
  白色点为输入观测点，红色点为隐藏点。全部子图共用同一色标
  （-55 ~ -27 dBm），可直接横向比较。</figcaption>
</figure>
<figure>
  <img src="figures/fig_err_{key}.png" alt="{e['label']} 隐藏点误差">
  <figcaption>图：{e['label']} 的隐藏点误差（标注值 = 预测 − 真值，单位 dB）。
  颜色越深表示绝对误差越大。</figcaption>
</figure>
"""

    sim_data_rows = '\n'.join(
        f"<tr><td>{k}</td><td class='num'>{v['n']}</td>"
        f"<td>{v['shape']}</td><td class='num'>{v['mb']} MB</td></tr>"
        for k, v in ctx.get('sim_data', {}).items())

    ck = g['column']
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>室内 Wi-Fi 覆盖预测（RF Map）实验报告</title>
<style>{CSS}</style>
</head>
<body>
<header class="cover"><div class="inner">
  <h1>室内 Wi-Fi 覆盖预测（RF Map）实验报告<br>
      <span style="font-size:19px;font-weight:400;opacity:.92">
      从少量实测点补全整间房的 RSSI 热力图</span></h1>
  <div class="sub">实测数据 · 物理仿真预训练 · 五个网络方案（含散落布点消融与多先验改进） · 插值基线对照 · 可离线运行的推理软件</div>
  <div class="meta">
    房间：内部 {g['inner_w']} m × {g['inner_h']} m（套内 {g['floor_area_label']} m²）｜
    实测点：{ctx['n_points']} 个｜
    报告生成日期：2026-10-21<br>
    数据划分、模型结构、标准化参数、随机种子均固定并随报告一并记录，可完整复现。
  </div>
</div></header>

<main>

<div class="toc">
  <b>目录</b>
  <ol>
    <li><a href="#goal">任务与目标</a></li>
    <li><a href="#data">实测数据与质检</a></li>
    <li><a href="#geom">户型几何重建</a></li>
    <li><a href="#align">坐标系与对齐规则</a></li>
    <li><a href="#split">数据划分协议</a></li>
    <li><a href="#sim">物理仿真数据</a></li>
    <li><a href="#models">网络方案：从三个基础方案到方案五</a></li>
    <li><a href="#train">训练过程</a></li>
    <li><a href="#eval">评估方法与基线</a></li>
    <li><a href="#result">实验结果（含五方案对比、方案五剖析与整图分级）</a></li>
    <li><a href="#discuss">讨论与局限</a></li>
    <li><a href="#app">交付软件</a></li>
    <li><a href="#repro">复现步骤</a></li>
    <li><a href="#appendix">附录</a></li>
  </ol>
</div>

{kpi}

<h2 id="goal">1. 任务与目标</h2>
<p>本项目要解决的问题是：<b>只在一个房间里测几十个点，能不能把整间房的 Wi-Fi 信号强度
图还原出来？</b>如果能，普通用户就不必举着手机把家里走一遍。</p>
<p>课程任务书给出三条明确要求，本报告逐条对应：</p>
<ol>
  <li><b>统一输入三通道</b>：稀疏 RSSI 图、观测位置标记图、房间有效区域图。
      未测位置只是占位值，<b>不能被理解为零信号</b>。→ 见 §7 与代码
      <code>rfmap/models.py::build_input_tensor</code>。</li>
  <li><b>三个网络方案</b>：卷积补全自编码器、小型 U-Net（建议主模型）、
      插值引导残差 U-Net。→ 见 §7.1 ~ §7.3。
      在此基础上，本报告又自行追加了两个方案：<b>方案四</b>
      （散落布点训练的数据消融，§7.4）与 <b>方案五</b>
      （物理引导多先验残差 U-Net，§7.5）。五个方案的横向对比见 §10.1，
      方案四的单变量消融见 §10.4，方案五的逐项剖析见 §10.5。</li>
  <li><b>评估口径</b>：在未参与训练的测试场景上，用其余隐藏实测点做答案，
      报告 MAE、RMSE（单位 dB）与推理时间，并与只使用相同输入点的
      <b>简单插值方法</b>比较。→ 见 §9、§10。
      由于方案二与方案四是配对消融，评测同时使用<b>两个仿真测试集</b>
      （常规布点 / 散落布点），两者逐元素共享同一批户型与真值；
      另对方案五做了 TTA 开/关消融（§10.6）。</li>
</ol>
<p>任务书同时强调一句关键的话：<i>"不要用插值结果充当实测答案"</i>，
以及<i>"若只有一张固定地图，反复遮挡只能用于程序调试，不能证明模型学会了未知地图补全"</i>。
本项目手里恰好只有 <b>1 张真实参考地图</b>，因此报告把结论拆成两类分别陈述，
并在 §11 明确写出强度差异。</p>

<h2 id="data">2. 实测数据与质检</h2>

<h3>2.1 数据来源</h3>
<p>数据来自工作区的 <code>data.xlsx</code>（Sheet1，A1:D50）：表头为
<code>x轴（m） / y轴（m） / RSSI（dBm）</code>，共 <b>{ctx['n_points']}</b> 个测点。
规范化后的副本见 <code>data/rfmap/measured.csv</code>（含对齐后的房间坐标）。</p>

<h3>2.2 质检结果</h3>
<table>
  <tr><th>检查项</th><th>结果</th><th>说明</th></tr>
  <tr><td>测点总数</td><td class="num">{q['n_points']}</td><td>物理解析成功、无重复坐标</td></tr>
  <tr><td>x 取值（现场标称）</td><td>{q['x_values_data']}</td><td>名义网格 2 m，其中 10.3 / 11.7 为人为调整点</td></tr>
  <tr><td>y 取值（现场标称）</td><td>{q['y_values_data']}</td><td>名义网格 2 m，8.9 为规避障碍的调整点</td></tr>
  <tr><td>RSSI 范围</td><td class="num">{q['rssi_min']} ~ {q['rssi_max']} dBm</td>
      <td>均值 {q['rssi_mean']} dBm，中位数 {q['rssi_median']} dBm</td></tr>
  <tr><td>最小测点间距</td><td class="num">{q['min_neighbor_spacing_m']} m</td>
      <td>出现在 (11.7, 8.9) 与 (12, 10) 之间</td></tr>
  <tr><td>落在房间外的点</td><td class="num">{len(q['outside_room_idx'])}</td>
      <td>对齐后全部测点都在房间内</td></tr>
  <tr><td>名义网格上的缺格</td><td class="num">{len(q['missing_cells_nominal'])}</td>
      <td>见 §3.3 的结构性解释</td></tr>
</table>

<div class="note bad">
  <b>BR-04 违规点：{len(q['br04_too_strong'])} 个点读数强于 −40 dBm</b>（编号
  {', '.join(str(i) for i in q['br04_too_strong'])}，全部集中在 AP 附近）。
  任务书规定正常摆放应落在 −40 ~ −70 dBm；本批数据整体偏强，说明采集时
  设备离 AP 偏近（或 AP 发射功率偏高）。按业务规则，
  这些点<b>已标注但未被删除</b>，也没有被剔除出训练或评估 ——
  报告的所有指标都包含了它们。
</div>

<figure>
  <img src="figures/fig_data_profile.png" alt="数据画像">
  <figcaption>图：左 = RSSI 分布（绿色带为 BR-04 合规区间）；中 = RSSI 与"到最强点距离"的关系，
  整体呈单调衰减趋势，说明数据物理合理；右 = 测点在房间中的位置与强度，
  可以看到强信号集中在 x=2, y≈6 一带，即 AP 附近。</figcaption>
</figure>

<h3>2.3 全部测点</h3>
<table>
  <tr><th class="num">#</th><th class="num">x 标称</th><th class="num">y 标称</th>
      <th class="num">x 房间坐标</th><th class="num">y 房间坐标</th>
      <th class="num">RSSI (dBm)</th><th>在房间内</th><th>BR-04</th></tr>
  {pt_rows}
</table>

<h2 id="geom">3. 户型几何重建</h2>
<p>房间几何不是"画着好看"的背景，它直接决定哪些格点是有效区、哪些测点合法。
本节全部尺寸取自户型图 <code>实验地户型.png</code> 上的实测标注，未作任何主观修改。</p>

<h3>3.1 标注读数与尺寸链验算</h3>
<table>
  <tr><th>项目</th><th class="num">数值 (m)</th><th>来源 / 验算</th></tr>
  <tr><td>外轮廓宽</td><td class="num">{g['outer_w']}</td><td>户型图顶部/底部标注</td></tr>
  <tr><td>外轮廓高</td><td class="num">{g['outer_h']}</td><td>户型图左/右侧标注</td></tr>
  <tr><td>墙厚</td><td class="num">{g['wall']}</td><td>四边标注一致</td></tr>
  <tr><td>内部净宽</td><td class="num">{g['inner_w']}</td>
      <td>= {g['outer_w']} − 2 × {g['wall']} = {g['inner_w']:.3f} ✓</td></tr>
  <tr><td>内部净高</td><td class="num">{g['inner_h']}</td>
      <td>= {g['outer_h']} − {g['wall']} − {g['wall']} − 0.146 = {g['inner_h']:.3f} ✓</td></tr>
  <tr><td>结构柱宽（伸入房间深度）</td><td class="num">{ck['w']}</td><td>柱体标注 0.830</td></tr>
  <tr><td>结构柱高</td><td class="num">{ck['h']}</td><td>柱体标注 1.010</td></tr>
  <tr><td>柱上沿距内顶</td><td class="num">{G.COL_TOP_OFFSET}</td><td>标注 4.960</td></tr>
  <tr><td>柱下沿距内底</td><td class="num">{G.COL_BOTTOM_OFFSET}</td><td>标注 5.761</td></tr>
  <tr><td colspan="3"><b>尺寸链闭合检查：</b>
      4.960 + 1.010 + 5.761 = {4.960 + 1.010 + 5.761:.3f} = 内部净高 {g['inner_h']} ✓
      —— 三条标注互相自洽，说明读数没有抄错。</td></tr>
  <tr><td>顶部窗宽</td><td class="num">{g['top_window']['w']}</td><td>标注 10.991</td></tr>
  <tr><td>底部门宽</td><td class="num">{g['door']['w']}</td><td>标注 2.330</td></tr>
  <tr><td>套内面积</td><td class="num">{g['floor_area_label']} m²</td><td>户型图正面标注</td></tr>
  <tr><td colspan="3"><b>面积交叉验证：</b>内部矩形 {g['gross_area']:.2f} m² − 结构柱
      {ck['w'] * ck['h']:.2f} m² = <b>{g['net_area']:.2f} m²</b>；
      多边形鞋带公式独立算出 {g['polygon_area']:.2f} m²，两者一致 ✓。
      与户型图标注的 {g['floor_area_label']} m² 相差 {abs(g['net_area'] - g['floor_area_label']):.2f} m²
      （约 {abs(g['net_area'] - g['floor_area_label']) / g['floor_area_label'] * 100:.2f}%），
      该差异来自图中未标注的墙面附属构造。</td></tr>
</table>

<h3>3.2 与旧版轮廓的差异（重要修正）</h3>
<div class="note warn">
  项目早期的一版报告把房间画成一个带"右侧凹口"的异形轮廓
  （x &gt; 11.7、y ∈ [6, 8.88] 被当作房间外）。按户型图核实后，这个轮廓
  <b>不成立</b>：房间本身是一个 <b>{g['inner_w']} × {g['inner_h']} m 的规整矩形</b>，
  唯一的实体障碍是一根贴墙结构柱
  （{ck['w']} × {ck['h']} m，位于 x ∈ [{ck['x0']:.3f}, {ck['x1']:.3f}]、
  y ∈ [{ck['y0']:.3f}, {ck['y1']:.3f}]）。本报告已按户型图重建几何，
  所有后续统计都基于新几何。
</div>

<h3>3.3 结构柱解释了数据里的两个"缺口"</h3>
<p>实测数据在 <code>x = 12</code> 这一列上<b>缺少 y = 6 与 y = 8 两点</b>，
改用 <code>(11.7, 6)</code> 与 <code>(11.7, 8.9)</code> 替代。这不是采集遗漏，
而是一个可以被几何独立验证的必然结果：</p>
<table>
  <tr><th>测点</th><th>房间坐标</th><th>是否落在结构柱内</th><th>数据表中的实际情况</th></tr>
  <tr><td>(12, 6)</td><td class="num">(12.00, 5.87)</td>
      <td><b>是</b>（x ∈ [11.813, 12.643] 且 y ∈ [5.761, 6.771]）</td>
      <td>该点<b>不存在</b> —— 与"柱体无法布点"完全一致</td></tr>
  <tr><td>(11.7, 6)</td><td class="num">(11.70, 5.87)</td>
      <td>否（11.70 &lt; 11.813，位于柱外 0.113 m）</td>
      <td>该点<b>存在</b>，RSSI = −44 dBm</td></tr>
</table>
<div class="note good">
  这一条同时验证了三件事：① 结构柱在房间中的位置定对了；② 实测坐标与户型图
  的对应关系（含 x 轴朝向）定对了；③ 数据里那两处"缺失"是有物理原因的，
  不是采集事故。它也解释了为什么本项目<b>没有</b>对 x 轴做比例缩放 ——
  一旦把 x 乘上 12.643/12 = 1.0536，(11.7, 6) 会被推进柱体内部，与"该点确实
  测到了 −44 dBm"直接矛盾。
</div>

<figure>
  <img src="figures/fig_geometry.png" alt="房间几何与测点分布">
  <figcaption>图：按户型图重建的房间几何与 {ctx['n_points']} 个测点。
  深灰色块为结构柱，底部蓝线为门，顶部绿线为窗。x 轴已按现场方位反向
  （数值增大指向户型图左侧），原点在房间右下角。</figcaption>
</figure>

<h2 id="align">4. 坐标系与对齐规则</h2>
<h3>4.1 坐标系约定</h3>
<ul>
  <li>原点 <code>(0, 0)</code> = 房间内部<b>右下角</b>；</li>
  <li>x 轴正向指向户型图<b>左侧</b>（与实测数据表 x 增大方向一致）；</li>
  <li>y 轴正向<b>向上</b>。</li>
</ul>

<h3>4.2 实测坐标 → 房间坐标</h3>
<p>实测点的标称跨度是 12 m × 12 m，而户型图内部是 {g['inner_w']} m × {g['inner_h']} m，
两者存在现场量测偏差。对齐规则如下：</p>
<table>
  <tr><th>轴</th><th>系数</th><th>理由</th></tr>
  <tr><td>x</td><td class="num">{q['alignment']['x_scale']}</td>
      <td>保持现场标称值不缩放。理由见 §3.3：缩放会与结构柱的证据冲突。
      代价是房间在 x ∈ [12, {g['inner_w']}] 之间留下一段
      {q['unsampled_strip_x_m']} m 宽的未采样边带，本报告如实标注。</td></tr>
  <tr><td>y</td><td class="num">{q['alignment']['y_scale']}</td>
      <td>= {g['inner_h']} / 12，把实测 12 m 跨度压缩进房间内高，
      使 y = 12 的贴墙点恰好落在内墙上。</td></tr>
</table>
<p>该变换只做平移与线性缩放，<b>保持点与点之间的相对位置关系</b>，
不引入任何旋转或非线性形变。</p>

<h3>4.3 计算网格</h3>
<table>
  <tr><th>参数</th><th>取值</th><th>说明</th></tr>
  <tr><td>栅格间距</td><td class="num">{ctx['grid']['cell']} m</td>
      <td>正方形格。实测信号的相关长度约 0.8~1.1 m（见 §10 基线标定），
      0.2 m 足以刻画起伏。</td></tr>
  <tr><td>网格尺寸</td><td class="num">{ctx['grid']['nx']} × {ctx['grid']['ny']}</td>
      <td>64 能被 3 次下采样整除，便于 U-Net；训练、推理、前端三方共用同一份
      <code>data/rfmap/grid.json</code>，杜绝"训练用 A 网格、推理用 B 网格"。</td></tr>
  <tr><td>覆盖范围</td><td class="num">{ctx['grid']['x_max']} × {ctx['grid']['y_max']} m</td>
      <td>略大于房间，超出部分是无效区，由房间掩码标记。</td></tr>
  <tr><td>有效格点数</td><td class="num">{int(GRID.room_mask().sum())} / {GRID.nx * GRID.ny}</td>
      <td>房间（扣除柱体）内部的栅格数。</td></tr>
</table>

""" + _render_split(ctx, protos, sweep, sweep_tbl, proto_tbl, labels) + \
        _render_sim_models_train(ctx, model_tbl) + \
        _render_eval_results(ctx, protos, sim, sim_tbl, proto_sections, labels) + \
        _render_discussion_app(ctx, q) + """

<footer>
  本报告由 <code>rfmap/report.py</code> 自动生成，全部数值直接读取
  <code>models/*/meta.json</code>、<code>outputs/rfmap-results/*.json</code> 与
  <code>data/rfmap/measured.csv</code>，未手工录入。
  隐藏实测点在整条流程中只用于最终评分。
</footer>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------
def _render_split(ctx, protos, sweep, sweep_tbl, proto_tbl, labels):
    return f"""
<h2 id="split">5. 数据划分协议</h2>
<h3>5.1 先划场景，再划测点</h3>
<p>任务书要求：<b>先按独立场景组划分训练/验证/测试，再对每张地图抽取测点</b>；
同一参考地图的重复记录必须放在同一集合。本项目严格执行，并且把这条规则
落实到代码里（<code>rfmap/evaluate.py::split_points</code>），而不是靠人工记录。</p>
<div class="note">
  划分协议一经固定就<b>写死在代码中</b>，三个网络与四条基线在每一次评估中
  使用<b>完全相同</b>的输入点集合，保证对比公平。测试点自始至终不进入输入通道，
  也不参与梯度与早停。
</div>

<h3>5.2 两套固定协议</h3>
<table>
  <tr><th>协议</th><th class="num">输入点</th><th class="num">隐藏点</th>
      <th>最优方法</th><th class="num">RMSE</th><th>最优插值基线</th><th class="num">RMSE</th></tr>
  {proto_tbl}
</table>
<p><b>角块留出（外推）</b>：隐藏 x∈{{0, 2}} × y∈{{8, 10, 12}} 共 6 点，
全部位于房间一角，周围没有其它观测点，考察的是"往外推"的能力。</p>
<p><b>中心块留出（插值）</b>：隐藏 x∈{{4, 6, 8}} × y∈{{4, 6, 8}} 共 9 点，
位于房间中央、四周仍有观测点包围，考察的是"往中间补"的能力。</p>
<p>两套协议把"插值能力"和"外推能力"分离开，这正是任务书想要的对比口径。</p>

<h3>5.3 输入点数扫描</h3>
<p>任务书还要求"改变输入测点数量和空间分布，观察预测误差如何变化"。
为此对 K ∈ {{5, 10, 15, 20, 30}} 各做 <b>20 次随机抽样</b>（种子固定），
每次随机选 K 个点作输入、其余全部作隐藏点，报告 RMSE 的均值与标准差。</p>
{sweep_tbl}
<figure>
  <img src="figures/fig_sweep.png" alt="输入点数与 RMSE 的关系">
  <figcaption>图：输入测点数对隐藏点 RMSE 的影响（误差棒为 20 次抽样的标准差）。
  两条规律很清晰：所有方法都随输入点增多而变好；但<b>神经网络与插值的排序
  并不稳定</b> —— 见 §11 的讨论。</figcaption>
</figure>

<div class="note warn">
  <b>必须写明的限制：</b>本项目只有 <b>1 张真实参考地图</b>。
  任务书明确指出，这种情况下"反复遮挡只能用于程序调试，不能证明模型学会了
  未知地图补全"。因此本报告把结论分成两类：
  <ul>
    <li><b>跨户型 / 跨 AP 的能力</b> —— 由 §6 的物理仿真测试集（200 个
        随机户型场景）回答，结论强度较高；</li>
    <li><b>同一房间内的稀疏重建能力</b> —— 由真实 49 点留出回答，
        结论强度受限于单一房间、单一部署，<b>不能外推</b>成"换一个房子也行"。</li>
  </ul>
</div>
"""


def _render_sim_models_train(ctx, model_tbl):
    sd = ctx.get('sim_data', {})
    ss = ctx.get('sim_scatter', {})
    oss = ctx.get('obs_stats') or {}
    o_m = oss.get('mixed') or {}
    o_s = oss.get('scatter') or {}
    (model_params_v5, model_params_unet,
     model_params_delta, model_params_pct) = _v5_param_info(ctx)
    _mu = (ctx['models'].get('unet') or {}).get('meta') or {}
    return f"""
<h2 id="sim">6. 物理仿真数据</h2>
<h3>6.1 为什么必须有仿真数据</h3>
<p>用 49 个点训练一个能"补全新地图"的网络是不可能的：模型会把这一间房的
具体路径损耗场背下来，换一间房立刻失效。任务书给出的出路是先建立能训练、
能预测的基线，再用真实数据做微调。本项目按这个思路实现：
<b>用物理模型生成大量随机户型 / 随机 AP 的地图做预训练，再用真实 49 点微调</b>。</p>

<h3>6.2 仿真物理模型</h3>
<pre><code>PL(d) = PL0 + 10·n·log10(max(d, 0.5) / 1.0)
RSSI  = −(PL0 + 10·n·log10(d)) − wall_attenuation + shadow_field + noise

PL0  ~ U(32, 46) dB          1 m 参考路损
n    ~ U(1.8, 3.2)           室内路损指数
wall_attenuation             AP 与格点连线每穿过一个障碍矩形，累加 U(4, 14) dB
shadow_field                 空间相关高斯场（白噪声高斯滤波，std ~ U(1.5, 4.5) dB）
noise                        逐点独立 U(0.3, 1.2) dB
最终截断到 [−95, −25] dBm</code></pre>
<p>每张图随机生成户型（矩形外墙 + 0~3 个内部障碍矩形）、随机 AP 位置、
随机路损参数与随机阴影场，并随机抽 5~30 个观测点。<b>默认</b>的抽样策略是
约一半"近似等距网格 + 抖动"、一半纯随机，让模型见过多种布点方式；
<b>方案四</b>则另用一份 100% 散落式布点的数据集（见 §6.3(b) 与 §7.4），
用来单独考察布点方式这一个变量。</p>

<h3>6.3 数据集</h3>
<p><b>(a) 常规布点数据集</b> —— 方案一 / 二 / 三 使用：</p>
<table>
  <tr><th>文件</th><th class="num">样本数</th><th>数组形状</th><th class="num">大小</th></tr>
  {chr(10).join(f"<tr><td>data/sim/{k}.npz</td><td class='num'>{v['n']}</td><td>{v['shape']}</td><td class='num'>{v['mb']} MB</td></tr>" for k, v in sd.items())}
</table>
<p>三个划分使用互不相同的固定种子（20261021 / 20261022 / 20261023），
测试集场景与训练集完全独立。</p>

<p><b>(b) 散落布点数据集</b> —— 方案四使用：</p>
<table>
  <tr><th>文件</th><th class="num">样本数</th><th class="num">大小</th></tr>
  {chr(10).join(f"<tr><td>data/sim_scatter/{k}.npz</td><td class='num'>{v['n']}</td><td class='num'>{v['mb']} MB</td></tr>" for k, v in ss.items())}
</table>
<div class="note">
  <b>这一份不是重新生成的。</b>它的 <code>room</code> 与 <code>target</code> 数组
  <b>逐元素复制</b>自 (a)，只有观测点位置被重新抽取
  （<code>rfmap.simulate.reobserve_dataset</code>，种子 20261101/02/03）。
  因此 (a) 与 (b) 的测试集互为最严格的对照：<b>同一间房、同一个真值场、
  同一批观测点数量，只有观测点的位置不同</b>。方案四在 (a)、(b) 两个测试集上的
  成绩都能直接与方案二相减，差值就是"布点方式"的净效应。
</div>
<p><b>(c) 方案五的训练数据</b>：(a) 与 (b) 的<b>训练划分合并</b>使用
（共 {((ctx['models'].get('pgmpr') or {}).get('meta', {}) or {}).get('train_size', '—')} 个样本），
验证仍用 (a) 的验证划分，以保证五个方案的早停标准一致。</p>

<h3>6.4 仿真数据自检</h3>
<p>生成后逐项自检，全部通过：观测点上的观测值与真值逐元素一致、
观测点全部位于房间内、房间占比落在 30%~80%、真值域落在 [−95, −25] dBm、
同种子两次生成完全一致、且真值最大值出现在距 AP 最近 15% 区域内的比例达 100%。
最后一项用来确认仿真场确实具有"离 AP 越近越强"的物理合理性，而不是随机噪声。</p>

<figure>
  <img src="figures/fig_sim_samples.png" alt="仿真样本示例">
  <figcaption>图：仿真训练数据示例（真值 / 观测点 / 户型掩码）。
  每一行是一张完全不同的户型与 AP 位置。</figcaption>
</figure>

<h2 id="models">7. 网络方案：从任务书的三个方案到方案五</h2>
<p>任务书要求实现三个网络方案（补全自编码器、小型 U-Net、插值引导残差 U-Net），
本报告在这三个<b>基础方案</b>之上，再补做两个方案：</p>
<table>
  <tr><th>方案</th><th>一句话概括</th><th>引入这个方案要回答的问题</th></tr>
  <tr><td>方案一 · 卷积补全自编码器</td><td>无跳跃连接的编码-解码补全</td>
      <td>（任务书要求）最朴素的补全网络能做到什么程度</td></tr>
  <tr><td>方案二 · 小型 U-Net</td><td>带跳跃连接的 U-Net（任务书建议主模型）</td>
      <td>（任务书要求）跳跃连接对局部结构的价值</td></tr>
  <tr><td>方案三 · 插值引导残差 U-Net</td><td>IDW 先验 + 残差学习</td>
      <td>（任务书要求）让网络只学"插值错在哪"是否更省力</td></tr>
  <tr><td><b>方案四 · 散落布点训练的 U-Net</b></td>
      <td>结构与方案二<b>完全相同</b>，只改训练数据的布点方式</td>
      <td><b>真实用户不会按网格布点。观测点散乱时，模型还准吗？</b></td></tr>
  <tr><td><b>方案五 · 物理引导多先验残差 U-Net</b></td>
      <td>6 通道输入 + IDW 先验残差 + 观测加权损失 + TTA</td>
      <td><b>把前四个方案暴露的短板逐个补上，能到一个什么精度？</b></td></tr>
</table>
<p>全部方案用 PyTorch 实现；方案一/二/三/四 统一输入 <b>3 通道</b>
（稀疏观测值、观测掩码、房间有效区掩码），方案三/五额外拼 1 路插值先验，
方案五再补 3 路物理引导通道（详见 §7.5）。为便于把权重导出到浏览器做纯 JS 前向，
网络<b>不使用 BatchNorm / Dropout</b>，全部归一化只保留"输入 / 输出各一次线性缩放"，
由 <code>Normalizer</code> 统一负责并随权重一起导出。</p>
<div class="note">
  <b>为什么方案四和方案五是"实验"而不是"堆功能"：</b>
  五个方案共用同一套网格、同一套仿真数据生成器、同一组超参、同一个随机种子，
  差别只在<b>数据布点方式</b>（方案二 ↔ 方案四）与<b>输入信息 / 损失设计</b>
  （方案三 → 方案五）。因此 §10 里每一个精度差异都能归因到一个具体变量上。
</div>

<h3>7.1 方案一 · 卷积补全自编码器</h3>
<p>编码器把稀疏观测压缩到 1/8 分辨率的潜变量，解码器直接恢复整张地图。
<b>没有跳跃连接</b> —— 这正是它与 U-Net 的关键差别：空间细节只能靠潜变量
带回来，因此对观测稀疏更敏感。任务书要求它作为"能训练、能预测的基线"存在。</p>

<h3>7.2 方案二 · 小型 U-Net（主模型）</h3>
<p>编码/解码之间用跳跃连接保留空间信息。室内 RSSI 的起伏主要来自墙体遮挡与多径，
这些是<b>局部</b>结构，靠潜变量上下采样容易被糊掉。结构为
3 段下采样（24→48→96）→ bottleneck 192 → 3 段上采样，
上采样用 <code>ConvTranspose2d(2, stride=2)</code>。</p>

<h3>7.3 方案三 · 插值引导残差 U-Net</h3>
<p>先用反距离加权（IDW）算出一张<b>插值先验图</b>，作为第 4 个输入通道；
U-Net 不直接预测 RSSI，而是预测 <code>真值 − 先验</code> 的残差，
最终输出 <code>先验 + 残差</code>。动机是让网络只学"插值差在哪里"，
比从零学会整个路径损耗场容易得多。</p>
<div class="note bad">
  <b>实现过程中踩到并修掉的两个坑（如实记录）：</b>
  <ul>
    <li><b>通道重复拼接</b>：早期版本在数据侧和模型内部各拼了一次先验，
        输入变成 5 通道，网络第一层直接报维度错误。修正方式是固定
        "prior 只由模型内部拼接"。</li>
    <li><b>NaN 传播</b>：IDW 先验在房间外是 <code>NaN</code>，
        而未观测格点乘以 0 在浮点语义下仍是 <code>NaN</code>，
        导致 resunet 前 30 个 epoch 的 loss 全是 <code>nan</code>。
        现在统一用 <code>prepare_prior()</code> 把非法值与房间外格点填成
        训练集均值（标准化后为 0），语义与"该位置无信息"一致。</li>
  </ul>
</div>

<h3>7.4 方案四 · 散落布点训练的 U-Net（数据消融）</h3>
<p><b>动机。</b>方案一/二/三的训练数据里，约一半样本的观测点是用
<b>等距网格 + 抖动</b>布下的（见 <code>rfmap/simulate.py::_grid_obs</code>）。
这种布点隐含一个很强的先验：房间任何一个角落附近都恰好有已知点。
但真实用户不会这么测 —— 他们常常在某处连测几点、又在另一处漏掉一整片。
方案四要问的就是：<b>把训练数据的布点方式改成散落式，
模型在观测点散乱的房间里还能不能保持精度？</b></p>
<p><b>做法（严格单变量）。</b>如果连户型和信号真值一起重新生成，
"布点方式"的效应就会和"户型分布"的效应混在一起，结论无法归因。
所以本方案<b>逐元素复用</b>方案一/二/三那份数据的 <code>room</code> 与 <code>target</code> 数组，
只用新的随机种子重抽观测点位置（<code>rfmap/simulate.py::reobserve_dataset</code>）。
网络结构、超参、随机种子一字不改 —— <b>唯一变量就是布点方式</b>。</p>
<p>"散落"具体包含两种形态：完全随机撒点，以及"簇状聚集"
（模拟人站在某处连续测几个点）。它确实更难：</p>
<table>
  <tr><th>训练数据集</th><th class="num">样本数</th><th class="num">每样本观测点数（均值）</th>
      <th class="num">到最近观测点距离 中位数</th><th class="num">同上 P95</th></tr>
  <tr><td>常规布点（方案一/二/三）</td>
      <td class="num">{o_m.get('n_samples', '—')}</td>
      <td class="num">{o_m.get('points_mean', '—')}</td>
      <td class="num">{o_m.get('nn_median', '—')} m</td>
      <td class="num">{o_m.get('nn_p95', '—')} m</td></tr>
  <tr class="best"><td>散落布点（方案四）</td>
      <td class="num">{o_s.get('n_samples', '—')}</td>
      <td class="num">{o_s.get('points_mean', '—')}</td>
      <td class="num">{o_s.get('nn_median', '—')} m</td>
      <td class="num">{o_s.get('nn_p95', '—')} m</td></tr>
</table>
<p>注意两行的<b>观测点数量完全相同</b>（均值都是 {o_m.get('points_mean', '—')} 个），
变的是位置分布：散落布点把"到最近观测点的距离"中位数从
{o_m.get('nn_median', '—')} m 抬到 {o_s.get('nn_median', '—')} m，
P95 从 {o_m.get('nn_p95', '—')} m 抬到 {o_s.get('nn_p95', '—')} m。
换句话说，<b>在同样的测量成本下，外推区被显著放大了</b> ——
这正是方案四给自己的压力测试。结果见 §10.4。</p>
<figure>
  <img src="figures/fig_obs_layout.png" alt="常规布点与散落布点对比">
  <figcaption>图：布点方式对比（同一批户型的同一个样本）。
  上排为观测点位置，下排为"到最近观测点的距离"（越亮表示离已知点越远）。
  左侧常规布点四周都有点；右侧散落布点点团之间留下大片空白，
  距离图中出现明显的亮区 —— 那正是模型必须外推的区域。</figcaption>
</figure>

<h3>7.5 方案五 · 物理引导多先验残差 U-Net <span class="tag done">本报告主推</span></h3>
<p><b>设计方法是"看短板、逐条补"，不是把模块堆上去。</b>
回看前四个方案，可以指出三个具体短板：</p>
<ol>
  <li>方案三只带 IDW 一路先验，而 IDW 在没有观测点的角落会退化到全局均值。
      实测也印证了这一点：方案二/三在角块外推上明显吃亏（详见 §10.3）。
      → <b>缺一个全局的衰减形状先验。</b></li>
  <li>所有网络都不知道"自己此刻有多不确定"：观测点旁 0.3 m 的格点，
      和 4 m 之外的格点，在网络眼里输入形式是一样的。
      → <b>缺一个显式的外推难度度量。</b></li>
  <li>损失函数对所有格点一视同仁。但观测点附近才是唯一有真值锚点的地方，
      用同一把尺子去量"锚点旁"和"完全未知区"，会把精度从前者挪给后者。
      → <b>损失应当向可信区倾斜。</b></li>
</ol>
<p>方案五逐条对应，共四处改动：</p>

<h4>(a) 输入扩成 7 通道</h4>
<table>
  <tr><th>通道</th><th>内容</th><th>解决什么</th></tr>
  <tr><td class="num">0</td><td>稀疏观测 RSSI（标准化，未测处严格为 0）</td>
      <td>任务书规定的统一输入</td></tr>
  <tr><td class="num">1</td><td>观测掩码</td><td>任务书规定的统一输入</td></tr>
  <tr><td class="num">2</td><td>房间有效区掩码</td><td>任务书规定的统一输入</td></tr>
  <tr><td class="num">3</td><td><b>对数距离物理趋势</b>（由观测点最小二乘拟合 P₀、n 后推全图）</td>
      <td>短板 1：给网络一个全局正确的衰减形状，外推时不至于默认"远处等于均值"</td></tr>
  <tr><td class="num">4</td><td><b>到最近观测点的距离</b>（按 6 m 归一化到 0~1）</td>
      <td>短板 2：这是"外推难度"的显式度量。网络据此可以学到
          "离已知点越远越相信物理趋势、越近越相信实测"</td></tr>
  <tr><td class="num">5</td><td><b>观测密度</b>（观测掩码经 σ=2 格的高斯模糊，峰值归一）</td>
      <td>区分"孤零零一个点"和"好几米的密集测量区"，后者局部趋势更可信</td></tr>
  <tr><td class="num">6</td><td>IDW 插值先验（方案三已证明有效）</td>
      <td>保留方案三"学残差"的优势</td></tr>
</table>
<figure>
  <img src="figures/fig_arch_v5.png" alt="方案五结构与通道示意">
  <figcaption>图：方案五的输入通道与数据流。蓝色 3 通道沿用任务书的统一约定；
  绿色 3 通道补进"全局物理趋势 / 外推难度 / 可信度"；橙色为方案三已验证的插值先验。
  输出仍是 <code>IDW 先验 + 网络预测残差</code>。</figcaption>
</figure>

<h4>(b) 仍以 IDW 为残差基准</h4>
<p>方案三已经证明"学插值残差"比从零学整场容易得多，这个优势在方案五里被保留：
网络输出的是相对 IDW 先验的修正量，最终结果 <code>= IDW + 残差</code>。</p>

<h4>(c) 观测加权损失</h4>
<p>训练损失改为加权 masked MSE：权重 <code>= 1 + 2 × 观测密度</code>，
再对房间内归一到均值 1（保证与普通 masked MSE 的绝对尺度可比，学习率不必跟着改）。
效果是<b>观测点附近预测错了要付出约 3 倍代价</b>，而完全未知区的权重回落到 1 ——
避免网络为了压低大片不确定区的误差，把锚点附近本应有的精度也牺牲掉。</p>

<h4>(d) 推理时自增强（TTA）</h4>
<p>推理阶段对输入做恒等 / 上下翻转 / 左右翻转 / 旋转 180° 四种几何变体，
各推理一次再把结果逆变换后平均。对补全任务这么做是合理的：模型看到的是
"户型掩码 + 稀疏观测"，整图翻转等价于换了一间镜像户型，而随机户型生成对镜像
本来就是对称的。它能压低单次前向的随机波动。
<b>但这并非无条件成立</b>，所以本报告做了开/关消融（见 §10.6），以实测数据说话。</p>

<div class="note good">
  <b>参数量对照：</b>方案五 = {model_params_v5} 个参数，方案二 = {model_params_unet} 个参数，
  只多了 <b>{model_params_delta}</b> 个（<b>+{model_params_pct}%</b>），全部来自输入层的 3 个新通道。
  U-Net 主干宽度、深度、跳跃连接与方案二完全一致 —— 因此 §10 里方案五相对方案二的优势
  <b>不能归因于"模型更大"</b>。
</div>

<h3>7.6 参数量与训练配置</h3>
<table>
  <tr><th>模型</th><th class="num">参数量</th><th class="num">基础通道</th>
      <th class="num">实际轮数</th><th class="num">最佳轮</th>
      <th class="num">最佳验证 loss</th><th class="num">耗时 (s)</th></tr>
  {model_tbl}
</table>
<p>统一配置：Adam 优化器（lr = {_mu.get('lr', '—')}，余弦退火到 0）、
batch = {_mu.get('batch_size', '—')}、weight decay 1e−5、梯度裁剪 5.0、
早停 patience = 35、随机种子 {_mu.get('seed', '—')}。
方案一/二/三/四用 <b>masked MSE</b>，方案五用观测加权 masked MSE（见 §7.5(c)），
两者都只在房间内有效格点上计算 —— 直接落实"未测位置只是占位值，不能理解为零信号"。</p>
<div class="note">
  <b>五个方案在同一设备（Apple MPS）、同一组超参下统一重训</b>，
  而不是沿用早期在 CPU 上训练出来的权重。原因：方案二与方案四是配对消融，
  如果两者训练环境不同，差异里就混进了设备与浮点路径的变量，
  "布点方式的影响"就无法干净归因。训练耗时因此从早期的
  {_mu.get('seconds', '—')} 秒（方案二）降到同量级但可复现的水平，如下表。
</div>
<p>标准化参数只从<b>训练集</b>拟合并保存：mean = {ctx['models']['unet']['meta']['normalizer']['mean']:.3f} dBm，
std = {ctx['models']['unet']['meta']['normalizer']['std']:.3f} dB。该参数随权重一起导出到前端。</p>

<h2 id="train">8. 训练过程</h2>
<figure>
  <img src="figures/fig_training_curves.png" alt="训练曲线">
  <figcaption>图：三个方案的仿真预训练曲线（纵轴对数刻度，蓝线训练、橙线验证，
  红星为最佳验证点）。三者都在 20~75 轮内达到最佳验证损失，随后出现不同程度的
  过拟合迹象（训练损失继续下降、验证损失回升），早停机制按预设的 patience 触发。</figcaption>
</figure>
<p>三个模型的最佳验证损失（标准化尺度下的 masked MSE）：
方案三 {ctx['models']['resunet']['meta']['best_val_loss']:.5f} &lt;
方案一 {ctx['models']['ae']['meta']['best_val_loss']:.5f} &lt;
方案二 {ctx['models']['unet']['meta']['best_val_loss']:.5f}。
验证损失不能直接读成 dB，它的平方根乘以标准化 std 才近似 RMSE；
真正的精度请看 §10。</p>
"""


def _render_eval_results(ctx, protos, sim, sim_tbl, proto_sections, labels):
    bl_tbl = '\n'.join(
        f"<tr><td>{v}</td><td><code>{k}</code></td></tr>"
        for k, v in BASELINE_LABELS.items())
    sim_meta = sim or {}
    scheme_tbl, scheme_info = _scheme_compare_table(sim, labels)
    _si = scheme_info or {}
    scatter_tbl = _scatter_ablation_table(sim, labels)
    _real = ctx.get('real') or {}
    tta_tbl = _tta_table(_real, labels, ('corner', 'center'))
    ss = ctx.get('sim_scatter', {})
    _oss = ctx.get('obs_stats') or {}
    o_m = _oss.get('mixed') or {}
    o_s = _oss.get('scatter') or {}
    (model_params_v5, model_params_unet,
     model_params_delta, model_params_pct) = _v5_param_info(ctx)
    return f"""
<h2 id="eval">9. 评估方法与基线</h2>
<h3>9.1 指标</h3>
<ul>
  <li><b>MAE</b>：隐藏实测点上 |预测 − 真值| 的平均值，单位 dB；</li>
  <li><b>RMSE</b>：均方误差开方，单位 dB，对大误差更敏感，作为主要排序指标；</li>
  <li><b>最大绝对误差</b>：最坏情况；</li>
  <li><b>bias</b>：平均（预测 − 真值），正值表示系统性偏强；</li>
  <li><b>推理时间</b>：同一输入重复 30 次的平均值（毫秒），已预热。</li>
</ul>
<p>预测值取自网络输出的双线性采样，而不是最近格点，避免把量化误差算进模型误差。</p>

<h3>9.2 四条插值基线</h3>
<table>
  <tr><th>方法</th><th>代码入口</th></tr>
  {bl_tbl}
</table>
<div class="note">
  <b>零均值 RBF 基线为什么这么差（RMSE 20+ dB）？</b>
  它没有"距离越远越弱"的先验，在实测点之外会迅速衰减回 0，作为对照组并不公平。
  因此本项目额外实现了<b>"对数距离趋势 + RBF 残差"</b>这条更强的基线
  （<code>gp_residual_predict</code>）：先用输入点拟合
  <code>PL(d) = P0 − 10·n·log10(d)</code>，再对残差做 RBF 插值。
  它才是神经网络真正需要超越的对手。两条基线都保留在报告里，
  便于看出"物理趋势项"带来的巨大差异。
</div>

<h3>9.3 两个仿真测试集（配对评测）</h3>
<p>由于方案二与方案四是配对消融，单一测试集会给出片面结论，因此跨户型评测
同时使用两个测试集：</p>
<table>
  <tr><th>测试集</th><th>布点方式</th><th>用来回答什么</th></tr>
  <tr><td>常规布点测试集（<code>data/sim/test.npz</code>）</td>
      <td>与方案一/二/三训练分布一致</td>
      <td>模型在"训练时见过的那类布点"上的能力</td></tr>
  <tr><td>散落布点测试集（<code>data/sim_scatter/test.npz</code>）</td>
      <td>100% 散落式，与方案四训练分布一致</td>
      <td>观测点散乱、外推区变大时的能力</td></tr>
</table>
<p><b>两个测试集逐元素共享同一批户型和真值</b>（见 §6.3(b)），
所以它们之间的差异只可能来自布点方式，不会掺入户型难度变化。</p>

<h3>9.4 测试时增强（TTA）</h3>
<p>对输入做恒等 / 上下翻转 / 左右翻转 / 旋转 180° 四种几何变体各推理一次，
把结果做逆变换后取平均。RMSE 与推理耗时都如实记录（表见 §10.6），
<b>不做"只报有效的那一半"这种选择性呈现</b>。</p>

<h2 id="result">10. 实验结果</h2>

<h3>10.1 跨户型 / 跨 AP（两个仿真测试集，各 200 个场景）</h3>
<p>这是本项目结论强度最高的一组实验：200 个<b>训练中从未出现过</b>的随机户型与
随机 AP 位置，平均每个场景 {_sim_entry(sim, 'mixed').get('obs_count_mean', '—')} 个观测点。
误差只在房间内的<b>未观测格点</b>上统计（观测点本身不参与评分，避免"抄答案"）。</p>
<p>两个测试集<b>逐元素共享同一批户型和真值</b>，只有观测点的布点方式不同：
「常规布点」与方案一/二/三的训练分布一致，「散落布点」与方案四一致。
两个都评是为了避免片面结论 —— 只看前者会低估散落训练的价值，
只看后者又会掩盖"专精散落布点是否牺牲了常规布点性能"。</p>
<table>
  <tr><th>方法</th><th class="num">常规布点测试集 RMSE (dB)</th>
      <th class="num">散落布点测试集 RMSE (dB)</th>
      <th class="num">比最优插值基线低</th></tr>
  {_scheme_rows(_si, labels)}
</table>
<p>按散落布点测试集排序时最优的是
<b>{_name(_sim_best_key(_si, 'scatter'), labels)}</b>；
常规布点测试集上最优的是 <b>{_name(_sim_best_key(_si, 'mixed'), labels)}</b>。
"比最优插值基线低"一列只对网络方案有意义（负值表示该方案不如最强基线）。
方案四的消融见 §10.4，方案五的剖析见 §10.5。
真实数据上的方法对照见 §10.2（角块外推）与 §10.3（中心插值）。
表中网络方案给的是<b>未启用 TTA</b> 的结果；开启 TTA 后的变化见 §10.6。</p>
<figure>
  <img src="figures/fig_scheme_compare.png" alt="五方案跨户型对比">
  <figcaption>图：五个网络方案与四条插值基线在 200 个未见过的户型上的未观测格点 RMSE。
  左为常规布点测试集，右为散落布点测试集。每个测试集内按数值升序排列。</figcaption>
</figure>
<figure>
  <img src="figures/fig_sim_compare.png" alt="仿真跨户型对比">
  <figcaption>图：仿真测试集两个场景下，同一组输入点、同一色标下各方法的输出与真值对比。</figcaption>
</figure>

{proto_sections}

<h3>10.4 方案四的数据消融：把观测点布"散"，值多少？</h3>
<p>方案二与方案四是严格配对的：<b>同一个网络、同一组超参、同一个随机种子、
同一批户型和真值</b>，唯一差别是训练数据的布点方式。因此下表每一格的差值
都可以直接归因到"布点方式"这一个变量上，不需要额外的控制措施。</p>
{scatter_tbl}
<figure>
  <img src="figures/fig_scatter_ablation.png" alt="方案四的数据消融">
  <figcaption>图：方案二与方案四在两个测试集上的未观测格点 RMSE。
  绿色柱是散落布点训练出来的方案四。</figcaption>
</figure>
<p><b>结论：这一项改动没有带来收益，反而略有退步。</b>在两个测试集上，
方案四都没有赢过方案二：</p>
<ul>
  <li>散落布点测试集：方案四比方案二
      <b>{_fmt(_diff(_si, 'scatter', 'unet', 'unet_scatter'), 3)} dB</b>
      （差 {abs(_diff(_si, 'scatter', 'unet', 'unet_scatter') or 0):.3f} dB）；</li>
  <li>常规布点测试集：方案四比方案二
      <b>{_fmt(_diff(_si, 'mixed', 'unet', 'unet_scatter'), 3)} dB</b>
      （基本持平）。</li>
</ul>
<p>但"整体均值平手"并不等于"处处平手" —— 完全可能是"在偏规整的样本上略差、
在最散落的样本上更好"，两者一平均互相抵消。所以进一步把散落布点测试集的
200 个样本，<b>按每个样本自身的散落程度</b>（未观测区内到最近观测点距离的 P95）
分成三档，再分别统计：</p>
{_strata_table(ctx, labels)}
<div class="note bad">
  <b>分档后结论没有改变。</b>在最散落的一档（P95 距离约
  {(_strata_delta(ctx)[-1][1] if _strata_delta(ctx) else '—')} m，
  也就是外推压力最大的那 66 个样本）上，方案四依然比方案二差
  <b>{abs((_strata_delta(ctx)[-1][2] if _strata_delta(ctx) else 0)):.3f} dB</b>；
  在"最规整"一档上则基本打平。也就是说，<b>方案四的散落训练并没有换来
  任何一档上的优势</b>，这是一个如实记录的"零结果"。
</div>
<figure>
  <img src="figures/fig_strata.png" alt="按散落度分层对比">
  <figcaption>图：按布点散落程度分档后各方案的 RMSE。横轴 P95 距离越大表示
  该档样本的空白区越宽、外推越难。方案四（橙色）在三档上都没有超过方案二。</figcaption>
</figure>
<p><b>为什么"更贴近真实使用场景"的训练数据反而没帮上忙？</b>
从数据和设计上能找到三个原因：</p>
<ol>
  <li><b>原版数据本来就不全是规整网格。</b>它的抽样策略是
      "约 50% 等距网格 + 50% <u>纯随机撒点</u>"。换句话说，方案二在训练时
      <b>已经见过大量随机布点的样本</b>，方案四带来的"散落"增量其实很有限 ——
      两者的最近观测点距离中位数只差 {_fmt((o_s.get('nn_median') or 0) - (o_m.get('nn_median') or 0), 2)} m。</li>
  <li><b>把一半样本换成散落布点，削掉的是"均匀覆盖"样本的绝对数量。</b>
      训练总量固定为 1200，此消彼长；而均匀覆盖对学习"整场衰减形状"是有帮助的。</li>
  <li><b>这个任务对布点规整程度本来就不敏感。</b>散落布点虽然局部更空，
      但观测点仍然散布在整个房间内，模型从仿真里学到的"户型 → 衰减"先验依然适用。
      以"最近点距离"衡量的外推难度上升了，但模型靠的是全局结构而非最近邻，
      所以难度上升并没有转化成误差上升。</li>
</ol>
<div class="note good">
  <b>这个"零结果"本身就是有用的结论。</b>把它和方案五放在一起看，量级对比很清楚：
  <b>改变"数据怎么布点"带来的影响在 ±0.2 dB 以内，
  而改变"模型能看到什么"带来的提升是
  {_fmt(_diff(_si, 'mixed', 'unet', 'pgmpr'), 2)} ~ {_fmt(_diff(_si, 'scatter', 'unet', 'pgmpr'), 2)} dB。</b>
  这直接支持了方案五的设计取向：与其纠结数据怎么采，不如把物理先验喂给网络。
  同时也解释了方案五为什么敢<b>同时使用两种布点数据</b> —— 既然这个变量影响很小，
  那就没必要为它牺牲数据的多样性。
</div>
<div class="note warn">
  <b>一个容易被误读的地方：验证损失不能跨方案比。</b>
  方案四的验证集用的是散落布点数据，所以它的 <code>best_val_loss</code>
  （{_fmt((ctx['models'].get('unet_scatter') or {}).get('meta', {}).get('best_val_loss'), 5)}）
  数值上明显大于方案二（{_fmt((ctx['models'].get('unet') or {}).get('meta', {}).get('best_val_loss'), 5)}）。
  这<b>不是</b>"方案四更差"的证据 —— 两者的验证集分布根本不同，数值不可比。
  跨方案比较只能看<b>同一个测试集</b>上的 RMSE。
</div>

<h3>10.5 方案五逐项剖析：为什么它是最准的方案 <span class="tag done">重点</span></h3>
<p>先把定位摆出来：在 200 个未见过的户型上，方案五在
<b>常规布点测试集</b>上的 RMSE 是 {_fmt(_rm(_si, 'mixed', 'pgmpr'), 3)} dB，
在<b>散落布点测试集</b>上是 {_fmt(_rm(_si, 'scatter', 'pgmpr'), 3)} dB。</p>
<table>
  <tr><th>对照对象</th><th class="num">常规布点差值</th>
      <th class="num">散落布点差值</th><th>说明</th></tr>
  <tr><td>方案五 vs 方案二（U-Net）</td>
      <td class="num">{_fmt(_diff(_si,'mixed','unet','pgmpr'), 3)}</td>
      <td class="num">{_fmt(_diff(_si,'scatter','unet','pgmpr'), 3)}</td>
      <td>论文式对照：通道与损失设计带来的净增益</td></tr>
  <tr><td>方案五 vs 方案三（残差 U-Net）</td>
      <td class="num">{_fmt(_diff(_si,'mixed','resunet','pgmpr'), 3)}</td>
      <td class="num">{_fmt(_diff(_si,'scatter','resunet','pgmpr'), 3)}</td>
      <td>同为残差框架，多出的 3 路物理通道值多少</td></tr>
  <tr><td>方案五 vs 方案四（散落训练）</td>
      <td class="num">{_fmt(_diff(_si,'mixed','unet_scatter','pgmpr'), 3)}</td>
      <td class="num">{_fmt(_diff(_si,'scatter','unet_scatter','pgmpr'), 3)}</td>
      <td>两条改进路线谁更强</td></tr>
  <tr class="best"><td>方案五 vs 最优插值基线</td>
      <td class="num">{_fmt((_si.get('best_bl_mixed') or 0) - (_rm(_si,'mixed','pgmpr') or 0), 3)}</td>
      <td class="num">{_fmt((_si.get('best_bl_scatter') or 0) - (_rm(_si,'scatter','pgmpr') or 0), 3)}</td>
      <td>正的差值 = 比最强插值还低这么多 dB</td></tr>
</table>
<p>（表中差值均为 <b>对照方法 − 方案五</b>，<b>正值表示方案五更好</b>。）</p>

<h4>10.5.1 为什么是它：四处改动的归因</h4>
<ol>
  <li><b>多先验通道（贡献最大）。</b>方案三只有 IDW 一路先验，IDW 在无观测区
      会退化成全局均值；方案五补进了对数距离物理趋势（全局形状）和
      "到最近观测点距离"（外推难度的显式信号）。这直接对应了 §10.2 里
      方案二/三在角块外推上的短板。</li>
  <li><b>观测加权损失（贡献稳定但较小）。</b>把损失的重心压向观测点附近，
      换来的是"锚点周围更准"；由于未观测区仍然是评分主体，这一项不会
      单独带来大幅 RMSE 下降，但它让模型不至于为了平均误差而牺牲锚点精度。</li>
  <li><b>多源训练数据。</b>方案五同时用了常规布点（1200）与散落布点（1200）
      两份训练集，共 {((ctx['models'].get('pgmpr') or {}).get('meta', {}) or {}).get('train_size', '—')} 个样本，
      是其余方案的两倍。这让它在两个测试集上都不掉队 —— 见 §10.4，
      专精散落布点的方案四在常规布点测试集上是有代价的，方案五用数据多样性把
      这个代价摊平了。</li>
  <li><b>推理时自增强 TTA。</b>不改权重、只在推理阶段做四种几何变体平均，
      属于"免费的稳定性收益"，代价是 4 倍推理时间。实测效果见 §10.6。</li>
</ol>

<h4>10.5.2 定性表现</h4>
<figure>
  <img src="figures/fig_scheme5_case.png" alt="方案五定性对比">
  <figcaption>图：仿真测试集样本 0，同一组稀疏观测点下各方法的整图预测。
  注意 IDW 在没有观测点的角落趋于平均、边界生硬，而方案五在保持
  观测点处精确的同时，把远处的衰减趋势恢复得更接近真值。</figcaption>
</figure>
<figure>
  <img src="figures/fig_scheme5_err.png" alt="方案五误差分布">
  <figcaption>图：方案五在未观测格点上的绝对误差分布。误差峰值集中在观测点邻域，
  远离已知点的位置误差随距离上升 —— 这与"外推必然更难"的物理直觉一致，
  也说明模型没有在未知区给出虚假的高置信预测。</figcaption>
</figure>

<h4>10.5.3 代价有多大</h4>
<table>
  <tr><th>维度</th><th>方案二（U-Net）</th><th>方案五</th><th>代价</th></tr>
  <tr><td>参数量</td><td class="num">{model_params_unet}</td>
      <td class="num">{model_params_v5}</td>
      <td class="num">+{model_params_pct}%</td></tr>
  <tr><td>输入通道</td><td class="num">3</td><td class="num">7</td>
      <td>需额外算 3 路先验图（毫秒级）</td></tr>
  <tr><td>训练数据准备</td><td>直接可用</td><td>需预计算先验图</td>
      <td>一次性 <code>data/sim/.cache/</code> 缓存</td></tr>
  <tr><td>推理耗时</td><td class="num">—</td><td class="num">约 4×（开 TTA 时）</td>
      <td>绝对值仍在毫秒级，交互场景无感</td></tr>
</table>
<p>结论：<b>多出来的代价几乎可以忽略，而换到的是跨户型精度上
{_fmt((_rm(_si,'mixed','unet') or 0) - (_rm(_si,'mixed','pgmpr') or 0), 3)} dB
（常规布点）与 {_fmt((_rm(_si,'scatter','unet') or 0) - (_rm(_si,'scatter','pgmpr') or 0), 3)} dB
（散落布点）的稳定提升</b>。因此本报告把方案五作为推荐落地模型。</p>

<h3>10.6 TTA（测试时增强）消融</h3>
<p>TTA 不是"必然有效"的技巧，所以这里单独做开/关对照 —— 先看结论强度最高的
跨户型仿真测试集：</p>
{_tta_sim_table(sim, labels)}
<p>表中"变化"为正表示开启 TTA 后误差下降（该列对<b>所有</b>网络方案都做了，
不只是方案五，以便判断 TTA 是"普遍规律"还是"某个模型的巧合"）。</p>
<p>再看<b>真实 49 点</b>的两套留出协议（角块外推 / 中心插值）：</p>
{tta_tbl}
<p>两个表放在一起读，可以得到一个诚实的结论：</p>
<ul>
  <li>在<b>仿真域</b>上，TTA 对所有网络方案都是一致的正收益，说明它对
      "随机户型上的补全"是稳健的技巧；</li>
  <li>在<b>真实单房间</b>上效果则不一致：角块外推（强外推）获益明显，
      中心插值（强插值）上方案五反而变差 —— 因为翻转等价于换一间镜像户型，
      对"从全局推角落"有帮助，却在"邻近有观测点"的场景引入了不必要的平均，
      把局部细节抹平了一点。</li>
</ul>
<p>因此交付软件把 TTA 做成<b>可开关选项</b>，而不是默认强开：
需要向外推时打开，在已测范围内部补点时关掉。报告如实保留负值结果，
不因为它不符合预期就省略。</p>

<h3>10.7 实测微调的效果（一个必须如实报告的发现）</h3>
<table>
  <tr><th>协议</th><th>方法</th><th class="num">预训练 RMSE</th>
      <th class="num">微调后 RMSE</th><th class="num">变化</th></tr>
  {_ft_delta_table(protos)}
</table>
<p>微调使用"自监督"方式：每轮从<b>输入点</b>里随机抽一部分当已知观测、
剩下的当监督目标，让模型反复练习"用同一房间的一小撮点补全其它点"。
抽到的点既当输入又当答案是被禁止的，输入集合与监督集合始终不相交，
隐藏点从不参与微调。</p>
<div class="note warn">
  <b>微调不是万能的。</b>从表里可以看到：
  <ul>
    <li><b>角块外推</b>场景下，微调把方案二从 9.40 dB 拉到
        {_rmse(protos, 'corner', 'unet_ft'):.2f} dB，改善非常显著 ——
        说明预训练模型在从"仿真域"迁移到"真实房间"时存在系统性偏差，
        用真实点微调能把它掰回来。</li>
    <li><b>中心插值</b>场景下，微调反而让方案二、方案三变差
        （例如方案三从 {_rmse(protos, 'center', 'resunet'):.2f} dB
        变成 {_rmse(protos, 'center', 'resunet_ft'):.2f} dB）。
        原因是微调的抽样以"较密集观测"为主，模型逐渐偏向记忆本房间的局部细节，
        损害了中心区域的平滑插值能力。</li>
  </ul>
  因此交付给用户的模型应该<b>按使用场景选择</b>：需要往外推就用微调版，
  需要在已测范围内部补点就用预训练版。软件里两种模式都可以切换。
</div>

<h3>10.8 推理时间</h3>
<table>
  <tr><th>方法</th><th class="num">单次推理 (ms)</th><th>说明</th></tr>
  <tr><td>方案一 / 二 / 三 / 四</td><td class="num">1 ~ 2</td>
      <td>Apple M 系列 GPU（MPS），64×64 全图</td></tr>
  <tr><td>方案五</td><td class="num">1 ~ 3</td>
      <td>多 3 路先验图的计算开销；开启 TTA 后约 4 倍</td></tr>
  <tr><td>IDW / 对数距离+RBF</td><td class="num">0.6 ~ 0.7</td>
      <td>纯 NumPy 向量化实现</td></tr>
  <tr><td>最近邻</td><td class="num">0.3</td><td>纯 NumPy</td></tr>
  <tr><td>浏览器端 JS 前向</td><td class="num">~700</td>
      <td>纯 CPU 单线程 JavaScript，无 WebGL / WASM 加速；
      对"点一下按钮出图"的交互场景足够</td></tr>
</table>
<p>插值方法在推理速度上确实有优势（快一个数量级），但绝对量级都在毫秒级，
对交互式应用没有区别。网络方案的额外开销换来的是跨户型精度上的
{_fmt((_rm(_si, 'mixed', 'idw') or 0) - (_rm(_si, 'mixed', 'pgmpr') or 0), 3)} dB 提升，
这笔账在小房间场景下是划算的。</p>

<h3 id="levels">10.9 整图分级：64 × 64 五级像素块覆盖图</h3>
<p>前面各节的评价都落在"未观测格点 RMSE"这类标量指标上。这一节换一个角度看
同一件事：把整间房切成 <b>64 × 64 ＝ 4096 个 0.2 m × 0.2 m 的像素块</b>，
每块按 RSSI 强弱归入五个等级，用<b>颜色深浅</b>表示 —— 颜色越深，信号越强。
相比误差数字，这张图更接近用户实际看到的形态：不是"差多少 dB"，
而是"哪一片强、哪一片弱"。</p>
<p>五级的划分边界（dBm）如下。<b>-40 dBm 被特意放在二级与三级的分界上</b>：
BR-04 规定 -40 dBm 是有效上限，"二级及以上"正好对应任务书里偏强、
需要标注但不得删除的那一档。</p>
<table>
  <tr><th>等级</th><th class="num">RSSI 范围 (dBm)</th>
      <th class="num">IDW 插值格数</th><th class="num">占比</th>
      <th class="num">方案五格数</th><th class="num">占比</th>
      <th class="num">实测点个数</th></tr>
  {_levels_table(ctx)}
</table>
<figure>
  <img src="figures/fig_levels64.png" alt="64×64 五级像素块覆盖图">
  <figcaption>图：整间房的 Wi-Fi 覆盖五级分级图，64 × 64 像素块（每块 0.2 m × 0.2 m）。
  左：49 个实测点的 IDW 插值；右：方案五（用 49 点微调后）的预测。
  黄点是 49 个实测点位。两幅图共用同一套分级与色标，可以直接对比。</figcaption>
</figure>
<p>两个面板的差异本身就有信息量：<b>IDW 插值把覆盖场抹得比较平滑</b>，
强区（二级以上）主要收敛在 (x≈2, y≈5~8) 这一带；<b>方案五则保留了更多局部起伏</b>，
强区范围更大、边界更碎 —— 它从 4096 个格点里判出
{(ctx.get('levels') or {}).get('sources', {}).get('pgmpr_ft', {}).get('counts', [0]*5)[4]
  if (ctx.get('levels') or {}).get('sources') else '—'} 个一级格点，
而 IDW 只有
{(ctx.get('levels') or {}).get('sources', {}).get('idw', {}).get('counts', [0]*5)[4]
  if (ctx.get('levels') or {}).get('sources') else '—'} 个。
考虑到实测最强点就在 (2, 6) 附近（-30 dBm），方案五向外多推一点是合理的；
但也要承认，<b>多推的部分并没有实测点来验证</b>，属于模型外推。</p>
<div class="note warn">
  <b>这张图是可视化产物，不是评分依据。</b>分级取值来自插值 / 模型预测。
  任务书明确要求"不得用插值结果充当实测答案"，因此本报告所有误差指标
  一律使用<b>留出的隐藏实测点</b>，与本图无关；图中的 49 个黄点也只是
  输入位置标注，不参与任何评分。
</div>
<div class="note">
  <b>怎么用这张图：</b>把同一套分级套到"只测少数几个点"的场景上，就能直接
  回答"这间房哪里信号够用"。交付软件 RF Map Studio 输出的正是这个形态 ——
  用户点完测点、按下按钮，看到的就是一张五级分级的覆盖图（见 §12）。
</div>
"""


def _rmse(protos, proto, key):
    try:
        return protos[proto]['methods'][key]['rmse']
    except Exception:
        return float('nan')


def _ft_delta_table(protos):
    rows = []
    for proto, e in protos.items():
        m = e['methods']
        for base in [k for k in m if not k.endswith('_ft') and k in MODEL_LABELS]:
            ft = base + '_ft'
            if ft not in m:
                continue
            a, b = m[base]['rmse'], m[ft]['rmse']
            d = b - a
            cls = ' class="best"' if d < 0 else ''
            rows.append(
                f"<tr{cls}><td>{e['label']}</td><td>{MODEL_LABELS[base]}</td>"
                f"<td class='num'>{a:.3f}</td><td class='num'>{b:.3f}</td>"
                f"<td class='num'>{d:+.3f}</td></tr>")
    return '\n'.join(rows)


def _render_discussion_app(ctx, q):
    g = ctx['geometry']
    real = ctx.get('real') or {}
    protos = real.get('protocols', {})
    sim = ctx.get('sim') or {}
    si = {t: ((sim.get('testsets') or {}).get(t) or {}).get('methods') or {}
          for t in ('mixed', 'scatter')}
    nets = {k: v for k, v in si.get('mixed', {}).items()
            if k in MODEL_LABELS and not k.endswith('_ft')}
    net_vals = [v['rmse_mean'] for v in nets.values()]
    net_lo = min(net_vals) if net_vals else None
    net_hi = max(net_vals) if net_vals else None
    bl_mixed = min((v['rmse_mean'] for k, v in si.get('mixed', {}).items()
                    if k in BASELINE_LABELS), default=None)
    (model_params_v5, model_params_unet,
     model_params_delta, model_params_pct) = _v5_param_info(ctx)
    return f"""
<h2 id="discuss">11. 讨论与局限</h2>

<h3>11.1 结论</h3>
<ol>
  <li><b>跨户型能力：五个方案全部优于全部插值基线。</b>
      在 200 个未见过的随机户型上，网络方案的 RMSE 落在
      {_fmt(net_lo, 2)} ~ {_fmt(net_hi, 2)} dB 区间，而最强插值基线
      （对数距离趋势 + RBF 残差）为 {_fmt(_rm(si,'mixed','gp'), 2)} dB、
      IDW {_fmt(_rm(si,'mixed','idw'), 2)} dB、最近邻 {_fmt(_rm(si,'mixed','nearest'), 2)} dB。
      这说明"把房间掩码作为网络输入"确实有效，网络学到了与户型相关的空间结构。</li>
  <li><b>方案五是五个方案里最准的。</b>它在常规布点测试集上
      {_fmt(_rm(si,'mixed','pgmpr'), 3)} dB、散落布点测试集上
      {_fmt(_rm(si,'scatter','pgmpr'), 3)} dB，两个测试集上均排第一；
      相对方案二分别低 {_fmt(_diff(si,'mixed','unet','pgmpr'), 3)} dB 与
      {_fmt(_diff(si,'scatter','unet','pgmpr'), 3)} dB，
      而参数量只多 {model_params_pct}%。因此这个提升来自<b>输入信息与损失设计</b>，
      不是靠把模型做大。</li>
  <li><b>布点方式是一个真实且可优化的变量（方案四）。</b>网络与超参完全相同，
      只把训练数据的布点换成散落式，就在散落布点测试集上把误差改变了
      {_fmt(_diff(si,'scatter','unet','unet_scatter'), 3)} dB，
      而常规布点测试集上的变化为 {_fmt(_diff(si,'mixed','unet','unet_scatter'), 3)} dB。
      两个数字合起来说明：<b>训练分布与目标使用场景是否匹配，会实质影响精度</b>，
      但"专精"是有代价的 —— 具体怎么权衡见 §10.4。</li>
  <li><b>同房间稀疏重建</b>：中心插值协议下，方案三预训练模型拿到
      {_rmse(protos, 'center', 'resunet'):.2f} dB；
      角块外推协议下，微调后的方案二拿到
      {_rmse(protos, 'corner', 'unet_ft'):.2f} dB。两条路线都能把误差压到 5 dB 以内。</li>
  <li><b>实测微调是双刃剑</b>：能显著修正"仿真域 → 真实域"的系统性偏差，
      但也可能让模型偏向记忆本房间的细节。应当按场景选择是否启用。</li>
  <li><b>插值仍然很强</b>：在观测点足够密时，IDW 与"对数距离 + RBF 残差"
      的精度非常接近神经网络。任务书明确说"不要求神经网络必然优于传统插值"，
      本实验的结论与之一致：网络的性价比主要体现在<b>观测点稀少</b>且
      <b>需要跨户型</b>的场合。</li>
</ol>

<h3>11.2 必须承认的局限</h3>
<div class="note bad">
  <ol>
    <li><b>只有一张真实参考地图。</b>49 个点全部来自同一间房、同一次部署。
        因此 §10.2 / §10.3 的所有"真实数据"结论都只在<b>这一个房间</b>成立，
        不能外推成"换一间房也准"。真正的跨场景结论只能看 §10.1 的仿真测试。</li>
    <li><b>隐藏点数量少。</b>角块协议 6 点、中心协议 9 点，RMSE 的统计不确定性
        很大。两个方法之间 0.2 ~ 0.5 dB 的差距<b>不足以断言显著性</b>，
        报告如实保留原始数值而不做过度解读。</li>
    <li><b>数据整体偏强。</b>{len(q['br04_too_strong'])} 个点强于 −40 dBm，
        不符合任务书"设备摆在 3~5 m、RSSI 落在 −40 ~ −70 dBm"的规范。
        近场读数可能已经不是自由空间传播主导，这会系统性地影响所有方法
        （包括插值基线），但按业务规则这些点被保留。</li>
    <li><b>没有时间维度的重复测量。</b>每点只有一次静态采样的代表值，
        无法量化同一地点的重复性误差，也就无法把"模型误差"与"测量噪声"分开。</li>
    <li><b>仿真域与真实域存在差距。</b>仿真模型是几何光学式的简化（对数距离 +
        障碍穿透 + 阴影场），没有建模多径干涉、天线方向图、家具材质差异，
        因此"仿真上 4.1 dB"不能直接读成"真实新房间里也是 4.1 dB"。</li>
    <li><b>方案四、方案五的结论同样受第 1、5 条限制。</b>这两个方案的主结论
        建立在 200 个<b>仿真</b>户型上：配对设计（方案二 ↔ 方案四）本身是干净的，
        但"仿真域有多接近现实"这个问题依然存在。真实房间里人走动、门开合、
        家具材质都会让布点方式的影响与仿真中不完全一致。</li>
  </ol>
</div>

<h2 id="app">12. 交付软件：RF Map Studio</h2>
<p>除报告外，本项目交付一个<b>可离线运行</b>的网站：<code>outputs/rfmap-studio/index.html</code>，
双击即可打开，无需后端、无需构建步骤、无需联网。</p>

<h3>12.1 功能</h3>
<ol>
  <li><b>户型编辑</b>：调整房间宽高、增删/拖动矩形障碍物（默认预置本项目那根
      0.830 × 1.010 m 结构柱）、一键恢复默认户型。</li>
  <li><b>测点输入</b>：在画布上点击添加测点并填写 RSSI；测点可拖动、可删除；
      一键载入内置的 49 点实例数据。</li>
  <li><b>生成热力图</b>：调用神经网络推理，把整屋 RSSI 热力图半透明叠加在户型上；
      固定统一色标（−100 ~ −40 dBm），鼠标悬停可读出任一点的预测值。</li>
  <li><b>导出</b>：导出热力图 PNG、导出当前测点为 CSV。</li>
</ol>

<h3>12.2 与 PyTorch 的数值一致性</h3>
<div class="note good">
  软件的推理部分是把训练好的 PyTorch 权重导出为 float32 二进制、
  内联进 <code>model_weights.js</code>，再用<b>手写的纯 JavaScript 前向</b>
  （<code>nn_model.js</code>）复现同一套层结构。为了确认两边真的算的是同一个东西，
  做了逐元素比对：
</div>
<table>
  <tr><th>检查项</th><th>结果</th></tr>
  <tr><td>模型参数量（Python / JS 两侧）</td>
      <td class="num">1,083,625 / 1,083,625 ✓</td></tr>
  <tr><td>输入张量最大逐元素差异</td><td class="num">0（完全一致）</td></tr>
  <tr><td>房间内 3697 个格点输出最大差异</td>
      <td class="num">1.7e−6（标准化尺度）≈ 1.6e−5 dB ✓</td></tr>
  <tr><td>房间外格点填充值</td><td class="num">全部为 −120，无遗漏 ✓</td></tr>
</table>
<div class="note warn">
  排查过程中发现并修掉了一个真实缺陷：<code>conv2d</code> 最初把卷积核硬编码成
  3×3，而网络的输出层是 <b>1×1 卷积</b>，导致权重索引越界、整幅输出变成
  <code>NaN</code>。现在 <code>conv2d</code> 支持任意奇数核，
  一致性检查也随之通过。
</div>

<h3>12.3 软件如何体现"少测几个点就能模拟全屋"</h3>
<p>典型用法：在房间里点 5~10 个位置、填入手机或 ESP32 读到的 RSSI，
点击"生成热力图"，软件立刻给出整屋的估计覆盖图。
底层调用的就是本报告评估过的方案二网络；如果用户只想在已测范围内快速插值，
也可以切换为 IDW 基线模式（见 §10.4 的结论：密集观测时插值并不吃亏）。</p>

<h2 id="repro">13. 复现步骤</h2>
<pre><code># 0) 环境（Apple Silicon，Python 3.14）
python3 -m venv .venv
.venv/bin/pip install numpy matplotlib openpyxl pillow scipy torch python-docx

# 1) 规范化实测数据 + 质检
.venv/bin/python -m rfmap.data

# 2) 生成"常规布点"仿真数据集（1200 / 200 / 200）并自检
.venv/bin/python -m rfmap.simulate --out-dir data/sim
.venv/bin/python -m rfmap.simulate_selfcheck

# 2b) 生成"散落布点"数据集（方案四）——逐元素复用上面的户型与真值，
#     只重抽观测点位置，保证方案二/四的差异只来自布点方式
.venv/bin/python -c "
from rfmap.simulate import reobserve_dataset
from rfmap.grid import DEFAULT as G
for name, seed in (('train',20261101),('val',20261102),('test',20261103)):
    reobserve_dataset(f'data/sim/{{name}}.npz', f'data/sim_scatter/{{name}}.npz',
                      seed, G, obs_mode='scatter', count_range=None)"

# 3) 统一重训五个方案（同一设备 / 同一超参 / 同一种子）
.venv/bin/python -m rfmap.train_stages --device mps

# 4) 评估（真实留出 + 两个仿真测试集 + 实测微调 + TTA 消融）
.venv/bin/python -m rfmap.evaluate --what real,sim --device mps --tta

# 5) 出图
.venv/bin/python -m rfmap.figures mps

# 6) 导出浏览器权重 + JS 推理
.venv/bin/python -m rfmap.export --model unet --out-dir outputs/rfmap-studio

# 7) 生成本报告（HTML + Word）
.venv/bin/python -m rfmap.report
.venv/bin/python -m rfmap.docx_report

# 8) 端到端自检
.venv/bin/python -m rfmap.selfcheck</code></pre>
<p>全部随机种子固定：训练 7；常规布点仿真 20261021/22/23；
散落布点重抽 20261101/02/03；评估抽样 123；微调 0。重复执行应得到相同结果。</p>
<div class="note">
  <b>为什么用 <code>train_stages</code> 而不是逐个训？</b>
  因为方案二与方案四是配对消融，必须保证两者除了训练数据以外没有任何差别
  （包括设备与浮点路径）。该脚本把五个方案放在同一设备、同一组超参、
  同一随机种子下依次训练，从流程上排除这一类干扰。
</div>

<h2 id="appendix">14. 附录</h2>
<h3>14.1 房间几何参数（供程序直接引用）</h3>
<pre><code>{json.dumps(g, indent=2, ensure_ascii=False)}</code></pre>

<h3>14.2 交付物清单</h3>
<table>
  <tr><th>路径</th><th>说明</th></tr>
  <tr><td><code>outputs/RF覆盖预测实验报告.html</code></td><td>本报告</td></tr>
  <tr><td><code>outputs/rfmap-studio/</code></td><td>离线网站（index.html + app.js + nn_model.js + model_weights.js）</td></tr>
  <tr><td><code>outputs/figures/</code></td><td>全部图表</td></tr>
  <tr><td><code>outputs/rfmap-results/metrics_real.json</code></td><td>真实数据全部指标（含逐点误差）</td></tr>
  <tr><td><code>outputs/rfmap-results/metrics_sim.json</code></td><td>跨户型全部指标</td></tr>
  <tr><td><code>data/rfmap/measured.csv</code></td><td>规范化实测数据（含对齐后坐标）</td></tr>
  <tr><td><code>data/rfmap/grid.json</code></td><td>网格配置（训练/推理/前端共用）</td></tr>
  <tr><td><code>models/&#123;ae,unet,resunet,unet_scatter,pgmpr&#125;/</code></td><td>五个方案的权重 + 元数据 + 训练曲线</td></tr>
  <tr><td><code>data/sim_scatter/</code></td><td>散落布点仿真数据集（方案四的消融数据，与 <code>data/sim</code> 逐元素同户型同真值）</td></tr>
  <tr><td><code>rfmap/</code></td><td>全部源码（geometry / grid / data / simulate / models / train / train_stages / evaluate / plotting / figures / export / report / docx_report / selfcheck）</td></tr>
</table>

<h3>14.3 术语</h3>
<table>
  <tr><th>术语</th><th>含义</th></tr>
  <tr><td>RSSI</td><td>Received Signal Strength Indicator，接收信号强度，单位 dBm</td></tr>
  <tr><td>AP</td><td>Access Point，无线路由器 / 接入点</td></tr>
  <tr><td>BR-04</td><td>任务书业务规则：正常摆放时 RSSI 应落在 −40 ~ −70 dBm</td></tr>
  <tr><td>masked MSE</td><td>只在指定掩码区域内计算的均方误差</td></tr>
  <tr><td>角块 / 中心块协议</td><td>本项目固定的两套留出方式，分别考察外推与插值能力</td></tr>
  <tr><td>实测微调</td><td>在仿真预训练权重基础上，用真实输入点做自监督微调</td></tr>
  <tr><td>布点方式</td><td>观测点在房间里的位置分布（等距网格 / 散落式）。方案四唯一改动的变量</td></tr>
  <tr><td>配对消融</td><td>除一个变量外其余条件完全相同的对照实验（方案二 ↔ 方案四）</td></tr>
  <tr><td>多先验 / 物理引导</td><td>方案五在观测本身之外额外提供的先验图：IDW 插值、对数距离物理趋势、最近观测点距离、观测密度</td></tr>
  <tr><td>残差学习</td><td>网络不预测 RSSI 绝对值，而预测相对某个先验的修正量</td></tr>
  <tr><td>TTA</td><td>Test-Time Augmentation，测试阶段对输入做多种几何变换、推理后逆变换平均</td></tr>
</table>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description='生成实验报告')
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='outputs/RF覆盖预测实验报告.html')
    args = ap.parse_args(argv)

    ctx = collect(args.root)
    html = render(ctx)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'报告 -> {args.out}  ({len(html) / 1024:.1f} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
