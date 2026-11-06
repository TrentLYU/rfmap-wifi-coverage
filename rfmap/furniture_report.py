"""由真实训练产物生成图表、HTML/Markdown/Word 报告和 Excel 汇总。"""

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import os
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle

from . import geometry as G
from . import plotting as P
from .furniture_models import MODEL_ORDER, MODEL_LABELS, BASELINE_LABELS
from .grid import DEFAULT as DEFAULT_GRID
from .models import idw_predict

REPORT_BASE = '放家具环境_RF覆盖预测实验报告'


def _json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _setup_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS', 'Heiti TC', 'Songti SC'):
        if name in available:
            plt.rcParams['font.family'] = name
            break
    plt.rcParams['axes.unicode_minus'] = False


def _read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def _method_label(key):
    if key.endswith('_ft_tta'):
        return MODEL_LABELS.get(key[:-7], key[:-7]) + '（微调 + TTA）'
    if key.endswith('_tta'):
        return MODEL_LABELS.get(key[:-4], key[:-4]) + '（TTA）'
    if key.endswith('_ft'):
        return MODEL_LABELS.get(key[:-3], key[:-3]) + '（实测微调）'
    return MODEL_LABELS.get(key, BASELINE_LABELS.get(key, key))


def _room_axis(ax):
    ax.add_patch(Rectangle((0, 0), G.INNER_W, G.INNER_H, fill=False,
                           edgecolor='#333333', linewidth=1.2))
    ax.add_patch(Rectangle((G.COL_X0, G.COL_Y0), G.COL_W, G.COL_H,
                           facecolor='#444444', edgecolor='#222222'))
    ax.set_xlim(-0.35, 13.0)
    ax.set_ylim(-0.35, 12.15)
    ax.set_aspect('equal')
    ax.set_xlabel('x（m，向东）')
    ax.set_ylabel('y（m，向北）')


def _draw_furniture(ax, furniture, labels=False, alpha=0.8):
    for f in furniture:
        color = '#d9a441' if f['kind'] == 'table' else '#5b9bd5'
        ax.add_patch(Circle((f['x'], f['y']), f['radius'], facecolor=color,
                            edgecolor='#444444', alpha=alpha, linewidth=0.7, zorder=5))
        if labels:
            ax.text(f['x'], f['y'], f['name'], ha='center', va='center', fontsize=6, zorder=6)


def make_figures(out_root):
    _setup_font()
    figdir = os.path.join(out_root, 'figures')
    os.makedirs(figdir, exist_ok=True)
    quality = _json(os.path.join(out_root, 'data/data_quality.json'))
    furniture = _json(os.path.join(out_root, 'data/furniture.json'))
    measured = _read_csv(os.path.join(out_root, 'data/measured.csv'))
    x = np.array([float(r['x_room']) for r in measured])
    y = np.array([float(r['y_room']) for r in measured])
    rssi = np.array([float(r['rssi_dbm']) for r in measured])

    # 数据、家具与信号画像
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    ax = axes[0]
    _room_axis(ax)
    sc = ax.scatter(x, y, c=rssi, cmap='viridis', vmin=-70, vmax=-15, s=28,
                    edgecolors='white', linewidths=0.3, zorder=4)
    _draw_furniture(ax, furniture, labels=True, alpha=0.68)
    ax.set_title(f'实测点与家具（{len(rssi)} 点）')
    fig.colorbar(sc, ax=ax, label='RSSI（dBm）', shrink=0.78)
    ax = axes[1]
    ax.axvspan(-70, -40, color='#b7dfc5', alpha=0.55, label='BR-04 正常范围')
    ax.hist(rssi, bins=np.arange(-55, -14, 2), color='#377eb8', edgecolor='white')
    ax.axvline(-40, color='#a15c00', linestyle='--', linewidth=1)
    ax.set_xlabel('RSSI（dBm）')
    ax.set_ylabel('测点数')
    ax.set_title('RSSI 分布（符号统一后）')
    ax.legend(fontsize=8)
    ax = axes[2]
    strongest = int(np.argmax(rssi))
    dist = np.hypot(x - x[strongest], y - y[strongest])
    near = np.array([any(np.hypot(px-f['x'], py-f['y']) <= 1.0 for f in furniture)
                     for px, py in zip(x, y)])
    ax.scatter(dist[~near], rssi[~near], s=22, alpha=0.75, label='距家具中心 >1 m')
    ax.scatter(dist[near], rssi[near], s=30, alpha=0.85, color='#d95f02', label='距家具中心 ≤1 m')
    ax.scatter(dist[strongest], rssi[strongest], marker='*', s=180, color='#d7301f',
               label='最强测点（AP 代理）', zorder=5)
    ax.set_xlabel('到最强测点的距离（m）')
    ax.set_ylabel('RSSI（dBm）')
    ax.set_title('距离衰减与家具附近测点')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'data_and_furniture.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    # 训练曲线
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for ax, name in zip(axes.flat, MODEL_ORDER):
        h = _json(os.path.join(out_root, 'models', name, 'history.json'))
        meta = _json(os.path.join(out_root, 'models', name, 'meta.json'))
        if h:
            ax.plot(h['epoch'], h['train'], label='训练损失', linewidth=1.4)
            ax.plot(h['epoch'], h['val'], label='验证损失', linewidth=1.4)
            b = int(meta['best_epoch'])
            ax.scatter([b], [meta['best_val_loss']], marker='*', s=90, color='#d7301f', zorder=5)
            ax.set_yscale('log')
        ax.set_title(MODEL_LABELS[name].replace(' · ', '\n'), fontsize=10)
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    fig.suptitle('五种模型的仿真预训练曲线（星号为最佳验证点）')
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'training_curves.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    sim = _json(os.path.join(out_root, 'results/metrics_sim.json'), {})
    if sim:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
        names = MODEL_ORDER + list(BASELINE_LABELS)
        colors = ['#377eb8'] * 5 + ['#999999'] * 4
        for ax, tag, title in zip(axes, ('mixed', 'scatter'), ('常规布点测试集', '散落布点测试集')):
            methods = sim['testsets'][tag]['methods']
            values = [methods[n]['rmse_mean'] for n in names]
            yy = np.arange(len(names))
            ax.barh(yy, values, color=colors)
            ax.set_yticks(yy, [MODEL_LABELS.get(n, BASELINE_LABELS.get(n)).replace(' · ', '\n')
                               for n in names], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('RMSE（dB，越低越好）')
            ax.set_title(title)
            ax.grid(axis='x', alpha=0.2)
            for yi, v in zip(yy, values):
                ax.text(v, yi, f' {v:.2f}', va='center', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, 'sim_comparison.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

    real = _json(os.path.join(out_root, 'results/metrics_real.json'), {})
    if real:
        protocols = list(real['protocols'])
        fig, axes = plt.subplots(1, len(protocols), figsize=(17, 5.5))
        names = [n + '_ft' for n in MODEL_ORDER] + list(BASELINE_LABELS)
        for ax, proto in zip(axes, protocols):
            entry = real['protocols'][proto]
            methods = entry['methods']
            values = [methods[n]['rmse'] for n in names]
            yy = np.arange(len(names))
            ax.barh(yy, values, color=['#377eb8'] * 5 + ['#999999'] * 4)
            ax.set_yticks(yy, [_method_label(n).replace(' · ', '\n') for n in names], fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel('隐藏点 RMSE（dB）')
            ax.set_title(f"{entry['label']}\n输入 {entry['n_input']} / 隐藏 {entry['n_hidden']}", fontsize=10)
            ax.grid(axis='x', alpha=0.2)
            for yi, v in zip(yy, values):
                ax.text(v, yi, f' {v:.2f}', va='center', fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, 'real_protocols.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name in MODEL_ORDER + ['idw', 'gp']:
            ks = sorted(int(k) for k in real['sweep'])
            means = [real['sweep'][str(k)][name]['rmse_mean'] for k in ks]
            stds = [real['sweep'][str(k)][name]['rmse_std'] for k in ks]
            ax.plot(ks, means, marker='o', label=_method_label(name))
            ax.fill_between(ks, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.08)
        ax.set_xlabel('随机输入测点数 K')
        ax.set_ylabel('隐藏点 RMSE（dB）')
        ax.set_title('输入点数量对预测误差的影响（预训练模型，阴影为 ±1 标准差）')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, 'point_sweep.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, len(protocols), figsize=(15, 4.7))
        for ax, proto in zip(axes, protocols):
            methods = real['protocols'][proto]['methods']
            for name in MODEL_ORDER:
                rows = methods[name + '_ft']['per_point']
                truth = [r[2] for r in rows]
                pred = [r[3] for r in rows]
                ax.scatter(truth, pred, s=25, alpha=0.75, label=MODEL_LABELS[name].split(' · ')[0])
            lo = min(-60, min(r[2] for r in rows) - 2)
            hi = max(-10, max(r[2] for r in rows) + 2)
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect('equal')
            ax.set_xlabel('隐藏点实测 RSSI（dBm）')
            ax.set_ylabel('预测 RSSI（dBm）')
            ax.set_title(real['protocols'][proto]['label'], fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, 'error_scatter.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

    # 全部实测点微调后的五张部署整图
    if os.path.exists(os.path.join(out_root, 'predictions/pgmpr_full_map.npz')):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, name in zip(axes.flat, MODEL_ORDER):
            d = np.load(os.path.join(out_root, 'predictions', name + '_full_map.npz'))
            im = ax.imshow(d['rssi'], origin='lower', extent=[0, 12.8, 0, 12.8],
                           cmap='viridis', vmin=-70, vmax=-15)
            _room_axis(ax)
            _draw_furniture(ax, furniture, labels=False, alpha=0.65)
            ax.scatter(x, y, s=5, c='white', edgecolors='black', linewidths=0.2, alpha=0.65)
            ax.set_title(MODEL_LABELS[name].replace(' · ', '\n'), fontsize=10)
        axes.flat[-1].axis('off')
        fig.colorbar(im, ax=list(axes.flat), label='预测 RSSI（dBm）', shrink=0.75)
        fig.suptitle('五种模型使用全部 140 个实测点微调后的覆盖图（部署展示，非独立测试）')
        fig.savefig(os.path.join(figdir, 'final_heatmaps.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)

        # 与 1.0 相同阈值的五级像素块图；增加真实家具轮廓和 140 点 IDW 对照。
        final = np.load(os.path.join(out_root, 'predictions/pgmpr_full_map.npz'))
        room = final['room']
        X, Y = DEFAULT_GRID.centers()
        idw = idw_predict(np.column_stack([x, y]), rssi, X, Y, room > 0.5)
        pred = np.where(room > 0.5, final['rssi'], np.nan)
        idw = np.where(room > 0.5, idw, np.nan)
        level_points = [SimpleNamespace(x=float(r['x_room']), y=float(r['y_room']))
                        for r in measured]
        P.fig_levels64(
            [('(a) 家具环境实测 140 点 IDW 插值', idw),
             ('(b) 方案五 PG-MPR-UNet（140 点微调后预测）', pred)],
            os.path.join(figdir, 'furniture_levels64.png'), DEFAULT_GRID,
            points=level_points, furniture=furniture, reverse_x=False,
            suptitle='带家具环境 Wi-Fi RSSI 覆盖五级分级图：64 × 64 像素块'
                     '（每块 0.2 m × 0.2 m，共 4096 块），颜色越深信号越强',
            note='黄点为 140 个实测点；橙色圆桌、蓝色圆椅。x 向东、y 向北；'
                 '分级值来自插值/模型预测，仅用于可视化，评分使用留出的隐藏实测点。')
        level_stat = {
            'edges': P.LEVEL_EDGES, 'names': P.LEVEL_NAMES,
            'ranges': P.LEVEL_RANGES, 'grid': {'nx': 64, 'ny': 64, 'cell': 0.2},
            'sources': {},
            'notes': ['有效格点不含房间外与结构柱', '最终 PG-MPR-UNet 整图不是独立测试结果'],
        }
        for key, field in (('idw', idw), ('pgmpr_ft', pred)):
            counts = P.level_counts(field)
            total = int(sum(counts))
            level_stat['sources'][key] = {
                'counts': counts, 'total': total,
                'share': [round(c / total, 4) if total else 0.0 for c in counts],
            }
        with open(os.path.join(out_root, 'results/furniture_levels64.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(level_stat, f, ensure_ascii=False, indent=2)


def _fmt(v, n=3):
    if v is None:
        return '—'
    if isinstance(v, (float, np.floating)):
        return f'{v:.{n}f}'
    return str(v)


def _best(methods, keys):
    valid = [(k, methods[k]) for k in keys if k in methods and methods[k].get('rmse') is not None]
    return min(valid, key=lambda kv: kv[1]['rmse']) if valid else ('—', {'rmse': None})


def build_blocks(out_root):
    q = _json(os.path.join(out_root, 'data/data_quality.json'))
    furniture = _json(os.path.join(out_root, 'data/furniture.json'))
    train = _json(os.path.join(out_root, 'training_summary.json'))
    sim = _json(os.path.join(out_root, 'results/metrics_sim.json'))
    real = _json(os.path.join(out_root, 'results/metrics_real.json'))
    final = _json(os.path.join(out_root, 'results/final_finetune.json'), {})
    b = []
    add = b.append
    add(('h1', '放家具环境下室内 Wi-Fi 覆盖预测（RF Map）实验报告'))
    add(('p', '基于 140 个新环境实测点、2 张桌子与 6 把椅子的实测中心位置，对上一版五种神经网络方案进行家具感知重训、实测微调与统一评估。'))
    add(('p', f'报告生成日期：{dt.date.today().isoformat()}；计算网格：64×64，格距 0.2 m；房间净尺寸：{G.INNER_W}×{G.INNER_H} m。'))
    add(('note', '本报告全部由训练权重、训练历史、预测文件和指标 JSON 自动生成。最终整图使用全部实测点微调，只用于部署展示；独立误差只来自协议内隐藏实测点或独立仿真测试场景。'))

    net_keys = [n + '_ft' for n in MODEL_ORDER]
    base_keys = list(BASELINE_LABELS)
    summary_rows = []
    for proto, entry in real['protocols'].items():
        nk, nv = _best(entry['methods'], net_keys)
        all_net_keys = [k for k in entry['methods'] if k not in base_keys]
        ok, ov = _best(entry['methods'], all_net_keys)
        bk, bv = _best(entry['methods'], base_keys)
        summary_rows.append([entry['label'], entry['n_input'], entry['n_hidden'],
                             _method_label(nk), _fmt(nv['rmse']),
                             _method_label(ok), _fmt(ov['rmse']),
                             _method_label(bk), _fmt(bv['rmse'])])
    add(('h2', '摘要与关键结果'))
    add(('table', ['真实评估协议', '输入点', '隐藏点', '最优微调网络', 'RMSE/dB',
                   '全部网络最优（含预训练/TTA）', 'RMSE/dB', '最优插值基线', 'RMSE/dB'], summary_rows))
    add(('note', '实测微调并非普遍提升：它在部分协议中改善外推，也可能因单一真实地图上的过拟合而变差。摘要同时列出预训练、微调和 TTA，实际部署应使用输入点内部的验证集选择是否微调，不能用外层隐藏点替部署模型调参。'))
    mixed = sim['testsets']['mixed']['methods']
    scatter = sim['testsets']['scatter']['methods']
    best_mixed = min(MODEL_ORDER, key=lambda n: mixed[n]['rmse_mean'])
    best_scatter = min(MODEL_ORDER, key=lambda n: scatter[n]['rmse_mean'])
    add(('p', f"独立仿真测试上，常规布点最优网络为{_method_label(best_mixed)}（RMSE={mixed[best_mixed]['rmse_mean']:.3f} dB），散落布点最优网络为{_method_label(best_scatter)}（RMSE={scatter[best_scatter]['rmse_mean']:.3f} dB）。这些是本轮家具仿真分布内的结果，不等价于真实跨房间泛化精度。"))
    add(('figure', 'real_protocols.png', '三种真实留出协议：五种实测微调网络与四条插值基线的隐藏点 RMSE。'))

    add(('h2', '1. 实验目标与上一版的差异'))
    add(('bullets', [
        '沿用五种核心方法：补全自编码器、小型 U-Net、IDW 引导残差 U-Net、散落布点训练 U-Net、物理引导多先验残差 U-Net。',
        '本轮新增家具占据通道，五种网络均可利用桌椅位置；家具不作为“房间外”掩码，家具处仍可预测 RSSI。',
        '预训练仿真新增两组“1 桌+3 椅”的随机布局和简化射线遮挡衰减，真实微调使用新环境 140 点。',
        '除角块、中心块和随机点数扫描外，新增家具附近留出协议，并分别统计近/远家具隐藏点误差。',
        '上一版模型、报告与结果均不覆盖，本轮全部写入 outputs/rfmap-furniture/。']))

    add(('h2', '2. 实测数据与质量控制'))
    add(('table', ['检查项', '结果'], [
        ['有效测点', f"{q['n_points']} / {q['nominal_cells']}；缺失 {q['n_missing']} 个，缺失位置保留为空"],
        ['RSSI 范围', f"{q['rssi_min']:.0f}～{q['rssi_max']:.0f} dBm；均值 {q['rssi_mean']}，中位数 {q['rssi_median']}"],
        ['符号修正', f"{q['raw_positive_count']} 个正值、{q['raw_negative_count']} 个负值；统一为 −|原值|，原值保留在 measured.csv"],
        ['BR-04 偏强', f"{len(q['br04_too_strong'])} 点强于 −40 dBm；全部标注、未删除"],
        ['BR-04 偏弱', f"{len(q['br04_too_weak'])} 点弱于 −70 dBm"],
        ['房间外测点', str(q['outside_room_idx'])],
        ['最小测点间距', f"{q['min_neighbor_spacing_m']} m"],
        ['家具附近测点', f"距家具中心 ≤1 m 共 {len(q['near_furniture_1m_idx'])} 点"],
    ]))
    add(('note', '正值 RSSI 在本实验中无物理意义，因此采用 −|原值| 是明确的数据清洗假设，而不是对源文件的静默改写。若正值列另有仪器定义，需要修改读取规则后重跑。'))
    add(('figure', 'data_and_furniture.png', '实测点、家具位置、RSSI 分布与距离衰减画像。'))

    add(('h2', '3. 坐标对齐与家具几何'))
    add(('p', f"原点为室内西南角，x 向东、y 向北。x 不缩放；y 乘以 {q['y_scale']:.9f}=11.731/12。该方向按家具图约定解释，是一项已记录的坐标假设。"))
    frows = []
    for f in furniture:
        src = '，'.join(f'{k}={v}' for k, v in f['source'].items())
        frows.append([f['name'], '桌子' if f['kind'] == 'table' else '椅子',
                      f"{f['x']:.3f}", f"{f['y']:.3f}", f"{f['radius']:.2f}", src])
    add(('table', ['家具', '类型', 'x/m', 'y/m', '半径/m', '原始测距'], frows))
    add(('p', '家具圆盘仅用于几何和通道栅格化。示意图曾为防重叠把个别椅子外移到 0.88 m 桌椅中心距；模型不使用这些展示位置，而使用上表实测中心点。家具材质、高度和精确外形未提供，因此不做材质级电磁仿真。'))

    add(('h2', '4. 五种模型与输入通道'))
    add(('table', ['方案', '网络变化', '基础通道', '含 IDW 后总通道', '训练数据'], [
        ['方案一', '无跳跃连接补全自编码器', '稀疏 RSSI、观测掩码、房间掩码、家具掩码', 4, '常规布点'],
        ['方案二', '带跳跃连接的小型 U-Net', '同方案一', 4, '常规布点'],
        ['方案三', 'IDW 先验 + 残差学习', '同方案一', 5, '常规布点'],
        ['方案四', '结构同方案二，唯一训练变量为散落布点', '同方案一', 4, '散落布点'],
        ['方案五', '物理趋势、最近观测距离、观测密度、IDW 残差与密度加权损失', '方案一 4 通道 + 3 路先验', 8, '常规+散落'],
    ]))
    add(('p', '未测格点的值通道严格为 0，并由观测掩码区分“未测”与“0 dBm”。损失只在房间有效区计算；家具区域内仍保留真值与监督。'))

    add(('h2', '5. 仿真预训练与实测微调'))
    add(('p', '每张仿真地图随机生成房间、结构障碍、AP、两组桌椅、对数距离路损、遮挡衰减、空间相关阴影和独立噪声。桌椅遮挡为桌 1～4 dB、椅 0.4～2 dB 的简化射线交叠损耗；RSSI 截断为 −95～−15 dBm。方案四复用同一户型、家具和真值，只重抽观测位置。'))
    trows = []
    for name in MODEL_ORDER:
        m = train[name]
        trows.append([MODEL_LABELS[name], f"{m['params']:,}", m['n_input_channels'], m['train_size'],
                      f"{m.get('epochs_max', 120)}/{m.get('patience', 35)}",
                      m['epochs_run'], m['best_epoch'], _fmt(m['best_val_loss'], 5), m['device'], m['seconds']])
    add(('table', ['模型', '参数量', '输入通道', '训练样本', '轮数上限/patience', '实际轮数', '最佳轮', '最佳验证损失', '设备', '耗时/s'], trows))
    add(('p', '统一超参：width=24、Adam、lr=0.002、batch=16、weight decay=1e−5、梯度裁剪 5、训练种子 7。方案二至五使用 120/35 的轮数上限/patience；方案一初次在初始平台期提前停止，因此按预设质量门改为 160/80 后重训，最佳轮与最终配置见表。方案五训练合并两类布点，其余按上表；所有模型使用同一常规布点验证集早停。实测自监督微调 lr=3e−4、种子 123；每轮输入点与监督点不相交。固定种子保证数据划分可复现，但 MPS 浮点路径不承诺跨进程逐位一致。'))
    add(('p', '方案一与方案五早期出现过有限数值的损失尖峰，随后恢复并正常收敛；训练曲线保留这些尖峰，没有裁剪。最终权重均取最佳验证轮，而不是最后一轮，因此早期瞬态和后期过拟合不会替代最佳权重。'))
    add(('figure', 'training_curves.png', '五种模型的训练/验证损失与最佳验证轮。'))

    add(('h2', '6. 评估协议与公平性'))
    prows = []
    for entry in real['protocols'].values():
        prows.append([entry['label'], entry['n_input'], entry['n_hidden'],
                      '仅输入点参与微调；隐藏点只用于最终评分'])
    add(('table', ['协议', '输入点', '隐藏点', '防泄漏规则'], prows))
    add(('bullets', [
        '角块：隐藏 x∈{0.9,1.09}、y∈{9.81,10.9,12} 的 6 点，考察外推。',
        '中心块：隐藏 x∈{4.09,5.09,6.09}、y∈{4.36,5.45,6.54} 的 9 点，考察插值。',
        '家具附近：隐藏距家具表面 ≤0.6 m 的测点，考察家具周边压力场景。',
        '点数扫描：K=10/20/40/60/80/100，各固定随机抽样 20 次，使用预训练模型，报告 RMSE 均值和标准差。',
        '所有方法在同一协议中使用完全相同的输入点；误差只在隐藏实测点上计算，MAE/RMSE 单位为 dB。',
    ]))

    add(('h2', '7. 独立仿真测试结果'))
    srows = []
    for name in MODEL_ORDER + list(BASELINE_LABELS):
        srows.append([_method_label(name), _fmt(mixed[name]['rmse_mean']), _fmt(mixed[name]['mae_mean']),
                      _fmt(scatter[name]['rmse_mean']), _fmt(scatter[name]['mae_mean'])])
    add(('table', ['方法', '常规 RMSE', '常规 MAE', '散落 RMSE', '散落 MAE'], srows))
    add(('figure', 'sim_comparison.png', '两个独立仿真测试集的五模型与基线 RMSE。'))

    add(('h2', '8. 新环境真实隐藏点结果'))
    for proto, entry in real['protocols'].items():
        add(('h3', '8.' + str(list(real['protocols']).index(proto) + 1) + ' ' + entry['label']))
        rows = []
        keys = MODEL_ORDER + [n + '_ft' for n in MODEL_ORDER] + list(BASELINE_LABELS)
        if 'pgmpr_tta' in entry['methods']:
            keys += ['pgmpr_tta', 'pgmpr_ft_tta']
        for key in sorted(keys, key=lambda k: entry['methods'][k]['rmse']):
            m = entry['methods'][key]
            rows.append([_method_label(key), _fmt(m['mae']), _fmt(m['rmse']), _fmt(m['max_abs']),
                         _fmt(m['bias']), _fmt(m.get('infer_ms'), 2), m['n']])
        add(('table', ['方法', 'MAE/dB', 'RMSE/dB', '最大绝对误差/dB', 'bias/dB', '推理/ms', '隐藏点'], rows))
    add(('figure', 'error_scatter.png', '三种协议中五种微调网络的预测值与隐藏实测值。'))
    add(('figure', 'point_sweep.png', '不同随机输入点数下的 RMSE 均值与标准差。'))

    add(('h2', '9. 家具附近误差与最终覆盖图'))
    nrows = []
    for proto, entry in real['protocols'].items():
        for name in MODEL_ORDER:
            m = entry['methods'][name + '_ft']
            nrows.append([entry['label'], MODEL_LABELS[name],
                          _fmt(m['near_furniture'].get('rmse')), m['near_furniture']['n'],
                          _fmt(m['far_furniture'].get('rmse')), m['far_furniture']['n']])
    add(('table', ['协议', '模型', '近家具 RMSE/dB', '近点n', '远家具 RMSE/dB', '远点n'], nrows))
    add(('note', '近/远家具误差是分层描述，不足以证明家具造成了多少 dB 的变化；没有同点位“撤掉家具”的对照测量，不能据此做因果结论。某些协议的近家具隐藏点可能很少，需结合 n 阅读。'))
    add(('figure', 'furniture_levels64.png', '带家具环境的 64×64 五级 RSSI 覆盖图：140 点 IDW 与方案五微调预测对照。'))
    add(('figure', 'final_heatmaps.png', '五种模型使用全部实测点微调后的最终覆盖预测。'))
    add(('p', '最终整图文件同时提供 CSV 和 NPZ，并保留房间/家具掩码。整图在观测附近更可信；远离测点的区域属于模型外推，不能当作新增实测值。'))

    add(('h2', '10. 讨论、限制与结论'))
    add(('bullets', [
        '本轮只有 1 张真实部署地图。多种遮挡协议能评估同房间稀疏重建，但不能证明对任意新房间、新 AP 都有同样精度。',
        '120 个正读数按负值解释；这是数据可用性的关键假设，原始值和修正标记均已交付。',
        '部分读数远强于 −40 dBm，可能包含近 AP 或饱和区；未删除这些点，误差指标包含其影响。',
        '家具采用直径 0.8 m 圆盘和简化遮挡，不含真实材质、高度、多径和天线方向图；仿真优势不自动等于现实优势。',
        '最终模型选用应结合隐藏点 RMSE、推理成本和目标布点方式，不应只凭热力图视觉效果。',
        '建议下一轮在相同测点进行无家具/有家具配对采集，并增加至少一个不同家具布局的独立地图，以分离家具效应与场景记忆。',
    ]))
    add(('p', '本轮已经完成五种模型的实际重训、独立仿真评估、三种真实留出评估、随机点数扫描和全量微调整图输出。具体“哪一方案最好”按协议分别给出，不把单一测试集排名包装成普遍结论。'))

    add(('h2', '11. 复现与交付物'))
    add(('code', '.venv/bin/python -m rfmap.furniture_experiment --stage all --device mps'))
    add(('p', '训练种子 7、评估种子 123、仿真种子 20261121/22/23、散落重抽种子 20261131/32/33。输入文件 SHA-256、软件版本与完整运行参数见 run_manifest.json。'))
    add(('table', ['路径', '内容'], [
        ['放家具环境_RF覆盖预测实验报告.html / .docx / .md', '三种格式实验报告'],
        ['实验结果汇总.xlsx', '实测、家具、训练、仿真、真实协议与扫描汇总'],
        ['models/<方案>/', '预训练权重、全量微调权重、训练历史与元数据'],
        ['results/', '完整指标 JSON，含逐点预测和近/远家具分组；validation.json 为交付自检证据'],
        ['predictions/', '五模型最终整图 CSV + NPZ'],
        ['figures/', '报告全部图表'],
        ['data/', '规范化实测数据、原始 RSSI、家具坐标与质检'],
        ['复现与模型使用说明.md', '运行命令、输入通道、模型加载及注意事项'],
    ]))
    return b


def _write_markdown(blocks, path):
    lines = []
    for block in blocks:
        kind = block[0]
        if kind == 'h1':
            lines += ['# ' + block[1], '']
        elif kind == 'h2':
            lines += ['## ' + block[1], '']
        elif kind == 'h3':
            lines += ['### ' + block[1], '']
        elif kind in ('p', 'note'):
            lines += [('> ' if kind == 'note' else '') + block[1], '']
        elif kind == 'bullets':
            lines += ['- ' + v for v in block[1]] + ['']
        elif kind == 'table':
            headers, rows = block[1], block[2]
            lines += ['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |']
            lines += ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows] + ['']
        elif kind == 'figure':
            lines += [f"![{block[2]}](figures/{block[1]})", f"*{block[2]}*", '']
        elif kind == 'code':
            lines += ['```bash', block[1], '```', '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _write_html(blocks, path):
    css = '''body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;max-width:1120px;margin:auto;padding:28px;color:#18212b;line-height:1.7}h1{background:#173b68;color:white;padding:28px;border-radius:10px}h2{border-bottom:2px solid #dde5ee;padding-bottom:7px;margin-top:38px}table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}th,td{border:1px solid #dce2e8;padding:7px;text-align:left;vertical-align:top}th{background:#f2f6fa}tr:nth-child(even){background:#fafbfd}img{max-width:100%;border:1px solid #dde3e8;border-radius:7px}.note{background:#fff7e8;border-left:4px solid #d88b18;padding:12px;margin:14px 0}.caption{text-align:center;color:#5d6874;font-size:13px}pre{background:#111c2c;color:#e7eef7;padding:15px;border-radius:7px;overflow:auto}li{margin:5px 0}'''
    out = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>放家具环境 RF 覆盖预测实验报告</title><style>' + css + '</style></head><body>']
    for block in blocks:
        kind = block[0]
        if kind in ('h1', 'h2', 'h3'):
            out.append(f'<{kind}>{html.escape(block[1])}</{kind}>')
        elif kind == 'p':
            out.append('<p>' + html.escape(block[1]) + '</p>')
        elif kind == 'note':
            out.append('<div class="note">' + html.escape(block[1]) + '</div>')
        elif kind == 'bullets':
            out.append('<ul>' + ''.join('<li>' + html.escape(v) + '</li>' for v in block[1]) + '</ul>')
        elif kind == 'table':
            out.append('<table><thead><tr>' + ''.join('<th>' + html.escape(str(v)) + '</th>' for v in block[1]) + '</tr></thead><tbody>')
            for row in block[2]:
                out.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>')
            out.append('</tbody></table>')
        elif kind == 'figure':
            out.append(f'<figure><img src="figures/{html.escape(block[1])}" alt="{html.escape(block[2])}"><figcaption class="caption">{html.escape(block[2])}</figcaption></figure>')
        elif kind == 'code':
            out.append('<pre><code>' + html.escape(block[1]) + '</code></pre>')
    out.append('</body></html>')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))


def _write_docx(blocks, out_root, path):
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    doc = Document()
    normal = doc.styles['Normal']
    normal.font.name = 'PingFang SC'
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'), 'PingFang SC')
    for style_name in ('Heading 1', 'Heading 2', 'Heading 3'):
        st = doc.styles[style_name]
        st.font.name = 'PingFang SC'
        st.element.rPr.rFonts.set(qn('w:eastAsia'), 'PingFang SC')
    for block in blocks:
        kind = block[0]
        if kind in ('h1', 'h2', 'h3'):
            doc.add_heading(block[1], level=int(kind[-1]))
        elif kind in ('p', 'note'):
            p = doc.add_paragraph(block[1])
            if kind == 'note':
                p.runs[0].bold = True
        elif kind == 'bullets':
            for item in block[1]:
                doc.add_paragraph(item, style='List Bullet')
        elif kind == 'table':
            headers, rows = block[1], block[2]
            t = doc.add_table(rows=1, cols=len(headers))
            t.style = 'Table Grid'
            for i, v in enumerate(headers):
                t.rows[0].cells[i].text = str(v)
                for run in t.rows[0].cells[i].paragraphs[0].runs:
                    run.bold = True
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = str(v)
            for row in t.rows:
                for cell in row.cells:
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.font.size = Pt(7.5)
            doc.add_paragraph()
        elif kind == 'figure':
            p = os.path.join(out_root, 'figures', block[1])
            if os.path.exists(p):
                doc.add_picture(p, width=Inches(6.4))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp = doc.add_paragraph(block[2])
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp.runs[0].italic = True
        elif kind == 'code':
            p = doc.add_paragraph(block[1])
            p.runs[0].font.name = 'Menlo'
    doc.save(path)


def _write_xlsx(out_root):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    wb.remove(wb.active)

    def sheet(name, headers, rows):
        ws = wb.create_sheet(name)
        ws.append(list(headers))
        for row in rows:
            ws.append([json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in row])
        ws.freeze_panes = 'A2'
        for cell in ws[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='1F4E78')
        for i, h in enumerate(headers, 1):
            values = [str(h)] + [str(r[i - 1]) if i - 1 < len(r) else '' for r in rows[:200]]
            ws.column_dimensions[get_column_letter(i)].width = min(max(len(v) for v in values) + 2, 42)
        return ws

    measured = _read_csv(os.path.join(out_root, 'data/measured.csv'))
    sheet('实测数据', measured[0].keys(), [[r[k] for k in measured[0]] for r in measured])
    furniture = _json(os.path.join(out_root, 'data/furniture.json'))
    sheet('家具位置', ['name', 'kind', 'x', 'y', 'radius', 'source'],
          [[f['name'], f['kind'], f['x'], f['y'], f['radius'], f['source']] for f in furniture])
    q = _json(os.path.join(out_root, 'data/data_quality.json'))
    sheet('数据质检', ['检查项', '值'], list(q.items()))
    train = _json(os.path.join(out_root, 'training_summary.json'))
    keys = ['model', 'label', 'params', 'n_input_channels', 'train_size', 'val_size',
            'epochs_run', 'best_epoch', 'best_val_loss', 'seconds', 'device', 'normalizer']
    sheet('训练摘要', keys, [[m.get(k) for k in keys] for m in train.values()])
    sim = _json(os.path.join(out_root, 'results/metrics_sim.json'))
    rows = []
    for tag, entry in sim['testsets'].items():
        for name, m in entry['methods'].items():
            rows.append([tag, name, _method_label(name)] + [m[k] for k in
                        ('rmse_mean', 'rmse_median', 'rmse_p90', 'mae_mean', 'runs')])
    sheet('仿真指标', ['testset', 'method', 'label', 'rmse_mean', 'rmse_median', 'rmse_p90', 'mae_mean', 'runs'], rows)
    real = _json(os.path.join(out_root, 'results/metrics_real.json'))
    rows = []
    for proto, entry in real['protocols'].items():
        for name, m in entry['methods'].items():
            rows.append([proto, name, _method_label(name), entry['n_input'], entry['n_hidden'],
                         m['mae'], m['rmse'], m['max_abs'], m['bias'], m.get('infer_ms'),
                         m['near_furniture'].get('rmse'), m['far_furniture'].get('rmse')])
    sheet('真实协议指标', ['protocol', 'method', 'label', 'n_input', 'n_hidden', 'mae', 'rmse',
                        'max_abs', 'bias', 'infer_ms', 'near_furniture_rmse', 'far_furniture_rmse'], rows)
    rows = []
    for k, methods in real['sweep'].items():
        for name, m in methods.items():
            rows.append([int(k), name, _method_label(name), m['rmse_mean'], m['rmse_std'], m['runs']])
    sheet('点数扫描', ['K', 'method', 'label', 'rmse_mean', 'rmse_std', 'runs'], rows)
    wb.save(os.path.join(out_root, '实验结果汇总.xlsx'))


def _write_usage(out_root):
    q = _json(os.path.join(out_root, 'data/data_quality.json'))
    text = f'''# 复现与模型使用说明

## 1. 一键复现

在本项目根目录运行：

```bash
.venv/bin/python -m rfmap.furniture_experiment --stage all --device mps
.venv/bin/python -m rfmap.furniture_selfcheck
```

默认依次执行：Excel 规范化 → 家具物理仿真 → 五模型预训练 → 独立仿真/真实留出评估 → 全部实测点微调 → 图表与 HTML/Markdown/Word/Excel 报告。

## 2. 固定口径

- 原始输入：`放家具数据.xlsx`，SHA-256 见 `run_manifest.json`。
- RSSI：`{q['rssi_rule']}`；原始值和修正标记保存在 `data/measured.csv`。
- 坐标：{q['coordinate_convention']}；y 缩放系数 {q['y_scale']:.9f}，x 不缩放。
- 网格：64×64，0.2 m；房间有效区不包含结构柱，家具不是无效区。
- 训练种子 7，评估/微调种子 123；详细仿真种子见 `run_manifest.json`。

## 3. 模型文件

`models/ae`、`unet`、`resunet`、`unet_scatter`、`pgmpr` 均包含：

- `model.pt`：仿真预训练最佳验证权重；
- `model_finetuned.pt`：全部 140 个真实点自监督微调后的部署权重；
- `meta.json`：结构、通道、标准化参数与训练配置；
- `history.json`：每轮训练/验证损失；
- `finetune.json`：全量微调配置。

## 4. 输入通道

| 模型 | 基础输入 | IDW 拼接后 |
|---|---|---:|
| ae / unet / unet_scatter | 稀疏 RSSI、观测掩码、房间掩码、家具掩码 | 4 |
| resunet | 同上 | 5 |
| pgmpr | 同上 + 对数距离趋势 + 最近观测距离 + 观测密度 | 8 |

家具通道索引为 3。方案三/五的 IDW 不在基础数组里重复拼接，而由模型 forward 拼接。

## 5. 加载示例

```python
from rfmap.furniture_models import load_trained, predict_grid
from rfmap.furniture_data import furniture_mask
from rfmap.grid import DEFAULT as grid

model, meta, norm = load_trained(
    'outputs/rfmap-furniture/models/pgmpr', device='mps', finetuned=True)
# obs_v / obs_m 是由允许使用的实测点构造的 64×64 数组。
# room = grid.room_mask().astype('float32')
# fm = furniture_mask(furniture, grid, room)
# pred = predict_grid(model, meta, norm, obs_v, obs_m, room, fm, grid, 'mps')
```

## 6. 五级图与离线网页

- 五级对照图：`figures/furniture_levels64.png`，阈值为 −48/−44/−40/−36 dBm。
- 等级统计：`results/furniture_levels64.json`；IDW 与 PG-MPR-UNet 均为 3697 个有效格点。
- 离线网页：`outputs/rfmap-studio/index.html`，默认使用方案二的四通道**预训练**权重，支持 12 点示例、家具拖放和无家具双推理对照。
- 重新导出网页权重：

```bash
.venv/bin/python -m rfmap.export --furniture --model unet --out-dir outputs/rfmap-studio
```

只有明确需要当前房间 140 点微调版本时才追加 `--finetuned`。网页重新训练不是必需步骤，也不应把当前房间微调权重默认当成跨房间模型。

## 7. 结果使用边界

- `results/metrics_real.json` 的隐藏点才用于真实误差评分；隐藏点未进入对应输入或微调。
- `predictions/` 是使用全部实测点后的部署整图，不能将其当成独立测试证据。
- 家具误差分层不是家具因果效应；需要同点位撤/放家具配对采集才能估计因果效应。
- 只有一张真实地图，不能把同房间留出结果外推为任意户型泛化能力。
'''
    with open(os.path.join(out_root, '复现与模型使用说明.md'), 'w', encoding='utf-8') as f:
        f.write(text)


def build_all(out_root):
    make_figures(out_root)
    blocks = build_blocks(out_root)
    _write_markdown(blocks, os.path.join(out_root, REPORT_BASE + '.md'))
    _write_html(blocks, os.path.join(out_root, REPORT_BASE + '.html'))
    _write_docx(blocks, out_root, os.path.join(out_root, REPORT_BASE + '.docx'))
    _write_xlsx(out_root)
    _write_usage(out_root)
    print(f'报告与汇总已生成：{out_root}', flush=True)
