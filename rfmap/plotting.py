"""出图：户型几何、热力图对比、误差图、训练曲线、输入点数扫描。

坐标与朝向
----------
所有图都遵循现场方位：**x 轴反向**（房间 x 越大越靠左），原点在右下角。
这一点从 :func:`rfmap.geometry` 的约定一路贯穿到 Canvas 前端，避免"图和现场
左右镜像"这类低级错误。

色标一致性
----------
任务书要求"展示输入点、预测热力图和隐藏点误差时使用一致的颜色标尺"。
真实数据的每一组对比图内部使用**同一组 vmin/vmax**（由 :data:`REAL_VMIN` /
:data:`REAL_VMAX` 固定），仿真图另用一组，并在图注中标明范围。
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from . import geometry as G
from .grid import DEFAULT as DEFAULT_GRID

# 真实数据集的统一色标（数据本身在 -30 ~ -51 dBm）
REAL_VMIN, REAL_VMAX = -55.0, -27.0
# 仿真数据集的统一色标（真值域 -95 ~ -25 dBm）
SIM_VMIN, SIM_VMAX = -90.0, -30.0

_FONT_DONE = False


def setup_fonts():
    """让 matplotlib 能显示中文。"""
    global _FONT_DONE
    if _FONT_DONE:
        return
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS',
                                       'Heiti TC', 'Hiragino Sans GB',
                                       'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 130
    plt.rcParams['savefig.bbox'] = 'tight'
    _FONT_DONE = True


def _extent(grid):
    return [grid.x0, grid.x_max, grid.y0, grid.y_max]


def _apply_orientation(ax, grid):
    """x 轴反向 + 等比例，保证房间不被拉扁。"""
    ax.set_xlim(grid.x_max, grid.x0)
    ax.set_ylim(grid.y0, grid.y_max)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m) —— 数值增大方向指向户型图左侧')
    ax.set_ylabel('y (m)')


def draw_room(ax, grid, color='#222', lw=1.6, fill=None, fill_alpha=0.12):
    """把房间多边形、结构柱、门、窗画到给定坐标系上。"""
    poly = G.room_polygon()
    if fill:
        ax.add_patch(Polygon(poly, closed=True, facecolor=fill,
                             edgecolor='none', alpha=fill_alpha, zorder=0))
    ax.add_patch(Polygon(poly, closed=True, facecolor='none',
                         edgecolor=color, linewidth=lw, zorder=4))
    # 结构柱
    ax.add_patch(Rectangle((G.COL_X0, G.COL_Y0), G.COL_W, G.COL_H,
                           facecolor='#444', edgecolor='none', zorder=5))
    # 门（底边开口）
    ax.plot([G.DOOR_X0, G.DOOR_X1], [0, 0], color='#1f6feb', lw=3.4,
            solid_capstyle='butt', zorder=6)
    # 顶部窗
    ax.plot([G.TOP_WINDOW_X0, G.TOP_WINDOW_X1], [G.INNER_H, G.INNER_H],
            color='#137a4b', lw=3.0, solid_capstyle='butt', zorder=6)


def _scatter_points(ax, pts, size=26, edge='#111'):
    """画实测点；按 BR-04 把超出 -40~-70 dBm 的点单独标注。"""
    x = [p.x for p in pts]
    y = [p.y for p in pts]
    strong = [p for p in pts if p.rssi > -40.0]
    ok = [p for p in pts if -70.0 <= p.rssi <= -40.0]
    if ok:
        ax.scatter([p.x for p in ok], [p.y for p in ok], s=size, c='#ffd166',
                   edgecolors=edge, linewidths=0.8, zorder=8, label='RSSI 在 -40~-70 dBm')
    if strong:
        ax.scatter([p.x for p in strong], [p.y for p in strong], s=size,
                   c='#ef476f', edgecolors=edge, linewidths=0.8, zorder=9,
                   label='BR-04 偏强 (> -40 dBm)')
    return x, y


# ==========================================================================
# 1. 户型几何 + 实测点
# ==========================================================================
def fig_geometry(pts, out_path, grid=None):
    setup_fonts()
    grid = grid or DEFAULT_GRID
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    draw_room(ax, grid, fill='#cfd8e3')
    _scatter_points(ax, pts)
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.92)
    ax.set_title('实测点分布与房间几何（49 点，颜色按 BR-04 分档）')
    _apply_orientation(ax, grid)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 2. 热力图（可多子图对比，统一色标）
# ==========================================================================
def fig_heatmaps(panels, out_path, vmin, vmax, grid=None, ncols=3,
                 suptitle=None, points=None, cmap='turbo',
                 show_points_in=None):
    """``panels`` = [(title, field2d), ...] 或 [(title, field2d, pts), ...]。

    ``field`` 房间外为 NaN；``pts`` 为该子图要叠加的测点列表（可选）。
    """
    setup_fonts()
    grid = grid or DEFAULT_GRID
    norm_panels = []
    for p in panels:
        if len(p) == 2:
            norm_panels.append((p[0], p[1], points))
        else:
            norm_panels.append(p)
    nrows = int(np.ceil(len(norm_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, (title, field, pts) in zip(axes, norm_panels):
        im = ax.imshow(field, origin='lower', extent=_extent(grid),
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear',
                       zorder=1)
        draw_room(ax, grid)
        if pts:
            for p in pts:
                mark = getattr(p, 'mark', 'white')
                ax.scatter([p.x], [p.y], s=30, c=mark, edgecolors='black',
                           linewidths=0.7, zorder=9)
        ax.set_title(title, fontsize=11)
        _apply_orientation(ax, grid)
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in axes[len(norm_panels):]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.0)
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.82, label='RSSI (dBm)')
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 3. 误差图（隐藏点误差）
# ==========================================================================
def fig_error_points(panels, out_path, grid=None, ncols=3, suptitle=None):
    """``panels`` = [(title, [(x, y, err), ...]), ...]，散点大小/颜色 = |误差|。"""
    setup_fonts()
    grid = grid or DEFAULT_GRID
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    sc = None
    for ax, (title, errs) in zip(axes, panels):
        draw_room(ax, grid, fill='#f2f5f8')
        if errs:
            xs = [e[0] for e in errs]
            ys = [e[1] for e in errs]
            vs = [abs(e[2]) for e in errs]
            sc = ax.scatter(xs, ys, c=vs, s=130, cmap='magma_r',
                            vmin=0, vmax=12, edgecolors='k', linewidths=0.6,
                            zorder=6)
            for e in errs:
                ax.annotate(f'{e[2]:+.1f}', (e[0], e[1]), fontsize=7.5,
                            ha='center', va='center', zorder=7)
        ax.set_title(title, fontsize=11)
        _apply_orientation(ax, grid)
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in axes[len(panels):]:
        ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.0)
    if sc is not None:
        fig.colorbar(sc, ax=axes.tolist(), shrink=0.82, label='|误差| (dB)')
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 4. 训练曲线
# ==========================================================================
def fig_training_curves(histories, out_path, labels=None):
    """``histories`` = {name: {'epoch':[], 'train':[], 'val':[]}}"""
    setup_fonts()
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.7))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h['epoch'], h['train'], label='训练 loss', lw=1.6)
        ax.plot(h['epoch'], h['val'], label='验证 loss', lw=1.6)
        if 'val' in h and h['val']:
            bi = int(np.argmin(h['val']))
            ax.scatter([h['epoch'][bi]], [h['val'][bi]], marker='*', s=90,
                       color='#b42318', zorder=5,
                       label=f'最佳 ep{ h["epoch"][bi] }')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel('masked MSE（标准化尺度）')
        ax.set_title((labels or {}).get(name, name), fontsize=10.5)
        ax.grid(alpha=0.28)
        ax.legend(fontsize=8)
    fig.suptitle('仿真预训练曲线（对数纵轴）', fontsize=12.5, y=1.03)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 5. 输入点数扫描
# ==========================================================================
def fig_sweep(sweep: dict, out_path, labels: dict | None = None,
              only=None, title=None):
    """``sweep`` = {'5': {'unet': {'rmse_mean':..,'rmse_std':..}, ...}, ...}

    ``only`` 用于只保留关键方法：方案变多以后全画会挤成一团、反而看不清对比。
    """
    setup_fonts()
    ks = sorted(int(k) for k in sweep)
    methods = sorted({m for v in sweep.values() for m in v})
    if only is not None:
        methods = [m for m in methods if m in only]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for m in methods:
        ys, es = [], []
        for k in ks:
            e = sweep[str(k)].get(m)
            ys.append(e['rmse_mean'] if e else np.nan)
            es.append(e['rmse_std'] if e else 0.0)
        ax.errorbar(ks, ys, yerr=es, marker='o', capsize=3, lw=1.7,
                    label=(labels or {}).get(m, m))
    ax.set_xlabel('输入实测点数量')
    ax.set_ylabel('隐藏点 RMSE (dB)')
    ax.set_title(title or '输入点数对预测精度的影响（真实数据，每档 20 次随机抽样）')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.8)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 6. 数据画像
# ==========================================================================
def fig_data_profile(pts, out_path):
    """RSSI-距离关系与直方图，用来说明数据的物理合理性与 BR-04 违规点。"""
    setup_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
    v = np.array([p.rssi for p in pts])
    axes[0].hist(v, bins=12, color='#4c78a8', edgecolor='white')
    axes[0].axvspan(-70, -40, color='#137a4b', alpha=0.10,
                    label='BR-04 合规区 -70~-40')
    axes[0].set_xlabel('RSSI (dBm)')
    axes[0].set_ylabel('测点数')
    axes[0].set_title('RSSI 分布')
    axes[0].legend(fontsize=8)

    # 到最强点的距离
    imax = int(np.argmax(v))
    cx, cy = pts[imax].x, pts[imax].y
    d = np.array([np.hypot(p.x - cx, p.y - cy) for p in pts])
    axes[1].scatter(d, v, s=32, c='#e45756', edgecolors='k', linewidths=0.4)
    axes[1].set_xlabel(f'距最强点的距离 (m)（最强点 x={cx:.2f}, y={cy:.2f}）')
    axes[1].set_ylabel('RSSI (dBm)')
    axes[1].set_title('RSSI - 距离关系')
    axes[1].grid(alpha=0.3)

    xs = [p.x_data for p in pts]
    axes[2].scatter(xs, [p.y_data for p in pts], c=v, cmap='turbo_r', s=60,
                    edgecolors='k', linewidths=0.5)
    axes[2].set_xlim(max(xs) + 1, min(xs) - 1)
    axes[2].set_aspect('equal')
    axes[2].set_xlabel('x（现场标称，m）')
    axes[2].set_ylabel('y（现场标称，m）')
    axes[2].set_title('测点位置与强度')
    axes[2].grid(alpha=0.3)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 7. 仿真样本示例
# ==========================================================================
def fig_sim_samples(npz_path, out_path, grid=None, k=4):
    setup_fonts()
    grid = grid or DEFAULT_GRID
    d = np.load(npz_path)
    fig, axes = plt.subplots(3, k, figsize=(3.6 * k, 10.2))
    for c in range(k):
        i = c * max(1, d['target'].shape[0] // k)
        i = min(i, d['target'].shape[0] - 1)
        room = d['room'][i].astype(bool)
        tgt = np.where(room, d['target'][i], np.nan)
        obs = np.where(d['obs_mask'][i] > 0.5, d['obs_value'][i], np.nan)
        roomv = np.where(room, 1.0, np.nan)
        for r, (field, title) in enumerate([(tgt, '真值'), (obs, '观测点'), (roomv, '户型掩码')]):
            ax = axes[r, c]
            if r < 2:
                ax.imshow(field, origin='lower', extent=_extent(grid),
                          cmap='turbo', vmin=SIM_VMIN, vmax=SIM_VMAX,
                          interpolation='nearest')
            else:
                ax.imshow(field, origin='lower', extent=_extent(grid),
                          cmap='Greys', vmin=0, vmax=1.3, interpolation='nearest')
            ax.set_title(f'{title} · 样本{i}', fontsize=9.5)
            _apply_orientation(ax, grid)
            ax.set_xlabel('')
            ax.set_ylabel('')
    fig.suptitle('仿真训练数据示例（随机户型 / 随机 AP）', fontsize=12.5, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 8. 布点方式对比（方案四的数据消融图）
# ==========================================================================
def fig_obs_layout(pairs, out_path, grid=None, k=0, suptitle=None):
    """对比两种布点方式：观测点位置 + "到最近观测点的距离"。

    ``pairs`` = [(标题, npz 路径), ...]，至少两组。

    下方那张距离图是方案四最关键的证据：散落布点会让"离最近观测点有多远"
    整体右移，也就是**外推区变大**。方案四要考察的正是"模型在这种更难补全的
    布点下还能不能保持精度"。
    """
    setup_fonts()
    grid = grid or DEFAULT_GRID
    from scipy.ndimage import distance_transform_edt

    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 9.2), squeeze=False)
    X, Y = grid.centers()

    panels = []
    dmax = 0.0
    for title, path in pairs:
        d = np.load(path)
        om = d['obs_mask'][k] > 0.5
        room = d['room'][k] > 0.5
        dist = np.where(room, distance_transform_edt(~om) * grid.cell, np.nan)
        panels.append((title, om, room, dist))
        if np.isfinite(dist).any():
            dmax = max(dmax, float(np.nanmax(dist)))

    im = None
    for col, (title, om, room, dist) in enumerate(panels):
        ax = axes[0][col]
        ax.imshow(np.where(room, 1.0, np.nan), origin='lower',
                  extent=_extent(grid), cmap='Greys', vmin=0, vmax=1.45,
                  interpolation='nearest', zorder=0)
        draw_room(ax, grid)
        jj, ii = np.nonzero(om)
        ax.scatter(X[jj, ii], Y[jj, ii], s=24, c='#d62828',
                   edgecolors='black', linewidths=0.55, zorder=9)
        md = float(np.nanmedian(dist[room]))
        p95 = float(np.nanpercentile(dist[room], 95))
        ax.set_title(f'{title}\n观测点 {int(om.sum())} 个 ｜ 最近点距离 中位数 '
                     f'{md:.2f} m / P95 {p95:.2f} m', fontsize=10.5)
        _apply_orientation(ax, grid)
        ax.set_xlabel('')
        ax.set_ylabel('')

        ax = axes[1][col]
        im = ax.imshow(dist, origin='lower', extent=_extent(grid), cmap='magma_r',
                       vmin=0, vmax=dmax, interpolation='bilinear', zorder=1)
        draw_room(ax, grid)
        ax.set_title('到最近观测点的距离（越亮 = 离已知点越远）', fontsize=10.5)
        _apply_orientation(ax, grid)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if col == n - 1 and im is not None:
            fig.colorbar(im, ax=ax, shrink=0.86, label='距离 (m)')

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 9. 方案五的结构示意图
# ==========================================================================
def fig_arch_v5(out_path):
    """方案五（物理引导多先验残差 U-Net）的输入通道与数据流示意图。"""
    setup_fonts()
    from matplotlib.patches import FancyBboxPatch, Rectangle as MplRect

    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.6)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec='#33404f', fs=9.6, weight='normal', tc='#16181d'):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.045,rounding_size=0.09',
                                    facecolor=fc, edgecolor=ec, linewidth=1.2,
                                    zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, fontweight=weight, color=tc, zorder=4, linespacing=1.45)

    def arrow(x0, y0, x1, y1, color='#33404f', style='-|>', lw=1.5, rad=0.0):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0), zorder=2,
                    arrowprops=dict(arrowstyle=style, color=color, linewidth=lw,
                                    connectionstyle=f'arc3,rad={rad}',
                                    shrinkA=0, shrinkB=0))

    # --- 输入通道 ---
    chans = [
        ('通道 0  稀疏观测 RSSI（未测处严格为 0）', '#e8f0fe'),
        ('通道 1  观测掩码', '#e8f0fe'),
        ('通道 2  房间有效区掩码', '#e8f0fe'),
        ('通道 3  对数距离物理趋势（全局衰减形状）', '#e6f7ee'),
        ('通道 4  到最近观测点的距离（外推难度）', '#e6f7ee'),
        ('通道 5  观测密度（高斯模糊后）', '#e6f7ee'),
        ('通道 6  IDW 插值先验（方案三已有）', '#fdf0e6'),
    ]
    x0, w = 0.15, 5.0
    h, gap = 0.55, 0.115
    total = len(chans) * h + (len(chans) - 1) * gap
    y_top = 5.25
    y_start = y_top - total
    ch_y = {}
    for i, (txt, fc) in enumerate(chans):
        y = y_top - (i + 1) * h - i * gap
        box(x0, y, w, h, txt, fc, fs=9.2)
        ch_y[i] = y + h / 2

    # --- U-Net ---
    ux, uw = 6.05, 2.5
    uy, uh = 1.55, 2.6
    box(ux, uy, uw, uh, '小型 U-Net\n（与方案二同宽）\n\n编码 → 瓶颈 → 解码\n+ 跳跃连接',
        '#dbeafe', ec='#1f6feb', fs=10.4, weight='bold')

    for i in ch_y:
        arrow(x0 + w, ch_y[i], ux, uy + uh / 2 + (0.95 - i * 0.14),
              color='#8aa0b8', lw=1.0)

    # --- 残差与相加 ---
    box(9.35, 2.85, 1.75, 0.95, '残差 Δ（标准化后）', '#fff4d6',
        ec='#a15c00', fs=9.6)
    arrow(ux + uw, uy + uh / 2, 9.35, 3.32)

    ax.text(11.55, 3.32, '＋', fontsize=25, ha='center', va='center', color='#0f7a4a')
    # prior 直接旁路到加法器
    arrow(x0 + w, ch_y[6], 11.55, 3.05, color='#c2641c', lw=1.8, rad=-0.28)
    ax.text(8.0, 1.05, 'IDW 先验同时旁路直连（残差相加的基准）',
            fontsize=9.2, color='#c2641c', ha='center')

    box(11.95, 2.75, 1.15, 1.15, '整图\nRSSI', '#d9f2e4', ec='#0f7a4a',
        fs=10.4, weight='bold')
    arrow(11.82, 3.32, 11.95, 3.32, color='#0f7a4a', lw=1.8)

    ax.text(0.15, 5.42, '输入：6 个基础通道 + 1 个插值先验通道（共 7 通道）',
            fontsize=10.4, fontweight='bold', color='#1b2733')
    ax.text(0.15, 0.30,
            '设计要点：蓝色 3 通道沿用任务书的统一输入约定；绿色 3 通道补进"全局物理趋势 / 外推难度 / 观测可信度"；'
            '橙色为方案三已验证有效的插值先验。',
            fontsize=9.4, color='#5c6470')
    ax.text(0.15, 0.03,
            '输出 = IDW 先验 + 网络预测的残差：网络不必从零学整场，只需学"插值错在哪"，'
            '这是方案三的结论，方案五予以保留。',
            fontsize=9.4, color='#5c6470')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 10. 通用对比条形图（不依赖具体实验，输入即画）
# ==========================================================================
def fig_compare_bars(groups, out_path, ylabel='RMSE (dB)', suptitle=None,
                     colors=None, figsize=None, annotate=True, log_x=False):
    """分组条形对比图。

    ``groups`` = [(组名, [(方法名, 数值, 是否高亮), ...]), ...]

    每组一个子图，组内按数值升序；高亮项（通常是本组最优或本项目主推方案）
    用绿色标出，方便一眼看出"谁最好"。

    ``log_x`` 用于数值跨度极大的场合：零均值 RBF 这种"对照基线"会高达
    20~35 dB，线性轴会把它之外的差异全部压扁，改对数轴才能看清主要对比。
    """
    setup_fonts()
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=figsize or (6.2 * n, 4.6), squeeze=False)
    axes = axes.ravel()
    for ax, (gname, items) in zip(axes, groups):
        items = sorted(items, key=lambda t: t[1])
        names = [t[0] for t in items]
        vals = [t[1] for t in items]
        hi = [t[2] if len(t) > 2 else False for t in items]
        cols = []
        for k, flag in enumerate(hi):
            if flag:
                cols.append('#0f7a4a')
            elif colors and k < len(colors):
                cols.append(colors[k])
            else:
                cols.append('#7f9cc0')
        ypos = np.arange(len(names))[::-1]
        ax.barh(ypos, vals, color=cols, edgecolor='#33404f', linewidth=0.7,
                height=0.68)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=9.6)
        ax.set_xlabel(ylabel)
        ax.set_title(gname, fontsize=11.5)
        ax.grid(axis='x', alpha=0.25, linestyle=':')
        ax.set_axisbelow(True)
        if annotate:
            for y, v in zip(ypos, vals):
                ax.text(v * 1.02, y, f'{v:.2f}', va='center', fontsize=9.0,
                        color='#33404f')
        if log_x:
            ax.set_xscale('log')
            lo = min(vals) * 0.75
            hi = max(vals) * 1.35
            ax.set_xlim(lo, hi)
            from matplotlib.ticker import FixedLocator, FuncFormatter
            ticks = [t for t in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 60)
                     if lo <= t <= hi]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.grid(axis='x', alpha=0.25, linestyle=':', which='both')
        else:
            ax.set_xlim(0, max(vals) * 1.18)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 11. 按散落度分层的分组柱状图
# ==========================================================================
def fig_strata(strat, out_path, labels=None, suptitle=None):
    """``strat`` 来自 :func:`rfmap.evaluate.eval_stratified`。

    横轴是"样本自身的散落程度"（用未观测区内到最近观测点距离的 P95 度量），
    纵轴是该档内的平均 RMSE。用来回答"某个方案是不是只在最难的样本上更强" ——
    只看整体均值会把这种局部差异抹平。
    """
    setup_fonts()
    testsets = strat.get('testsets') or {}
    n = max(len(testsets), 1)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 4.7), squeeze=False)
    for ax, (tag, e) in zip(axes.ravel(), testsets.items()):
        bins = e['bins']
        if not bins:
            continue
        methods = list(bins[0]['methods'].keys())
        x = np.arange(len(bins))
        w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            vals = [b['methods'][m] for b in bins]
            pos = x + i * w - 0.4 + w / 2
            ax.bar(pos, vals, w * 0.92, label=(labels or {}).get(m, m),
                   edgecolor='#33404f', linewidth=0.6)
            for xx, v in zip(pos, vals):
                ax.text(xx, v * 1.012, f'{v:.2f}', ha='center', fontsize=7.8,
                        color='#33404f')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b['label']}\nP95 距离 {b['spread_mean']} m\nn = {b['n']}"
                            for b in bins], fontsize=9)
        ax.set_xlabel('样本按布点散落程度分档')
        ax.set_ylabel('未观测格点 RMSE (dB)')
        ax.grid(axis='y', alpha=0.25, linestyle=':')
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(b['methods'][m] for b in bins for m in methods) * 1.22)
        ax.legend(fontsize=8.7)
        ax.set_title(f'{tag} 测试集', fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ==========================================================================
# 12. 五级分级的 64 x 64 像素块图
# ==========================================================================
# 分级边界（dBm）。把观测范围切成 5 档，颜色越深 = 信号越强。
# 特意让 -40 dBm 落在 L2 / L3 的分界上：BR-04 规定 -40 dBm 是"有效上限"，
# 这样"二级及以上"就正好对应任务书里偏强、需要标注的那一档。
LEVEL_EDGES = [-70.0, -48.0, -44.0, -40.0, -36.0, -20.0]
# 索引 0 = 最弱（颜色最浅），索引 4 = 最强（颜色最深）
LEVEL_NAMES = ['五级 · 极弱', '四级 · 弱', '三级 · 中', '二级 · 强', '一级 · 极强']
LEVEL_RANGES = ['< -48', '-48 ~ -44', '-44 ~ -40', '-40 ~ -36', '>= -36']


def level_colors():
    """从 Blues 单色阶取 5 个由浅到深的颜色（越深 = 信号越强）。"""
    cmap = plt.get_cmap('Blues')
    return [cmap(t) for t in np.linspace(0.20, 0.95, 5)]


def level_index(field, edges=None):
    """把连续 RSSI 场映射到 0..4 的等级索引（0 最弱），无效格点记 -1。"""
    edges = list(edges or LEVEL_EDGES)
    f = np.asarray(field, dtype=np.float64)
    idx = np.full(f.shape, -1, dtype=np.int32)
    ok = np.isfinite(f)
    li = np.digitize(f[ok], edges[1:-1], right=False)
    idx[ok] = np.clip(li, 0, len(edges) - 2)
    return idx


def level_counts(field, edges=None):
    """统计一个场在各等级上的有效格点数，返回长度 5 的列表（索引 0 最弱）。"""
    idx = level_index(field, edges)
    return [int((idx == k).sum()) for k in range(5)]


def fig_levels64(panels, out_path, grid=None, edges=None, suptitle=None,
                 points=None, figsize=None, cell_line=0.35, note=None):
    """把 RSSI 场按信号强弱分成 5 级，画成 64 x 64 的像素块图。

    ``panels`` = [(标题, 场), ...]，每个场是 ny x nx 的 RSSI（dBm），房间外为 NaN。

    刻意用 ``pcolormesh`` 逐格上色、而不是平滑的 ``imshow``：任务书要求"分为
    像素块"，逐格描白边能让人看清每个 0.2 m x 0.2 m 网格各自的等级，而双线性
    插值会把它们抹成渐变色带。

    注意：本图的取值来自插值 / 模型预测，仅用于**可视化**；评分一律使用留出的
    隐藏实测点，绝不用这里的插值结果充当实测答案（任务书硬性要求）。
    """
    setup_fonts()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    grid = grid or DEFAULT_GRID
    edges = list(edges or LEVEL_EDGES)
    cols = level_colors()
    cmap = ListedColormap(cols)
    norm = BoundaryNorm(edges, cmap.N)

    n = max(len(panels), 1)
    fig, axes = plt.subplots(1, n, figsize=figsize or (7.2 * n, 7.0),
                             squeeze=False)
    axes = axes.ravel()
    X = grid.x_coords()
    Y = grid.y_coords()
    pcm = None
    for ax, (title, field) in zip(axes, panels):
        f = np.asarray(field, dtype=np.float64)
        ax.set_facecolor('#eef1f4')
        pcm = ax.pcolormesh(X, Y, np.ma.masked_invalid(f), cmap=cmap, norm=norm,
                            shading='nearest', edgecolors=(1.0, 1.0, 1.0, 0.55),
                            linewidth=cell_line, antialiased=True, zorder=1)
        draw_room(ax, grid)
        if points:
            for p in points:
                ax.scatter([p.x], [p.y], s=24, c='#ffd166', edgecolors='black',
                           linewidths=0.7, zorder=8)
        ax.set_title(title, fontsize=11.5)
        _apply_orientation(ax, grid)
        ax.set_xlabel('')
        ax.set_ylabel('')
    for ax in axes[len(panels):]:
        ax.axis('off')

    # 手动排版：色标与图例都推到画布边缘，避免压住房间内部
    fig.subplots_adjust(left=0.075, right=0.875, bottom=0.145,
                        top=0.90 if suptitle else 0.965, wspace=0.10)
    if pcm is not None:
        cax = fig.add_axes([0.895, 0.30, 0.030, 0.42])
        cbar = fig.colorbar(pcm, cax=cax, spacing='uniform')
        cbar.set_ticks([])
        for i in range(5):
            cbar.ax.text(0.5, (i + 0.5) / 5.0, str(5 - i),
                         transform=cbar.ax.transAxes, ha='center', va='center',
                         fontsize=11.5, fontweight='bold',
                         color=('#ffffff' if i >= 2 else '#12395e'))
        cbar.ax.set_title('等级\n(1 最强)', fontsize=9.0, pad=6)

    marks = ['⑤', '④', '③', '②', '①']
    handles = [Patch(facecolor=cols[i], edgecolor='#5a6b7d',
                     label=f'{marks[i]}{LEVEL_NAMES[i].split(" · ")[-1]} '
                           f'{LEVEL_RANGES[i]} dBm')
               for i in range(5)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.475, 0.036),
               ncol=5, fontsize=8.8, framealpha=0.95, borderpad=0.55,
               title='信号等级（颜色越深 = 信号越强）', title_fontsize=9.2,
               columnspacing=1.0, handlelength=1.4)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.982, linespacing=1.45)
    if note:
        fig.text(0.475, 0.006, note, ha='center', fontsize=8.2, color='#4a5560')
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == '__main__':
    from . import data as D
    os.makedirs('outputs/figures', exist_ok=True)
    pts = D.load(root='.')
    print(fig_geometry(pts, 'outputs/figures/geometry.png'))
    print(fig_data_profile(pts, 'outputs/figures/data_profile.png'))
