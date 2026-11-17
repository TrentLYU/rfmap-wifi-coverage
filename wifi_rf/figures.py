"""生成交付用图表（中文标注）。"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle

from . import config as C
from . import models as M
from .data import level_of
from .deploy import deploy_field
from .evaluate import METHODS, baseline_idw, metrics
from .physics import cell_centers, room_mask
from .realdata import load_real, real_inputs, sample_bilinear, predict_field

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

LCOL = ["#b23a48", "#e07a5f", "#f2cc8f", "#8ab17d", "#2f6f4e"]
LEVEL_NAMES = ["L1 最弱", "L2 较弱", "L3 一般", "L4 良好", "L5 最强"]
CMAP_LV = ListedColormap(LCOL)
CMAP_DB = LinearSegmentedColormap.from_list("rf", LCOL)
FIGS = os.path.join(C.OUT_DIR, "figures")


def _setup(ax, title=None, grid=False):
    ax.set_xlim(0, C.DOMAIN)
    ax.set_ylim(0, C.DOMAIN)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 13.001, 2))
    ax.set_yticks(np.arange(0, 12.001, 2))
    ax.tick_params(labelsize=7, colors="#666")
    for s in ax.spines.values():
        s.set_color("#ccc")
    if title:
        ax.set_title(title, fontsize=9.5, pad=5, color="#222")


def _room(ax, under=None):
    """画房间轮廓与浅色底盘。"""
    ax.add_patch(Rectangle((0, 0), C.ROOM_W, C.ROOM_H, facecolor="#f6f6f4",
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0, 0), C.ROOM_W, C.ROOM_H, facecolor="none",
                           edgecolor="#8a8a8a", lw=1.2, zorder=6))


def _ap(ax, ap, label=True, z=8):
    ax.plot([ap[0]], [ap[1]], marker="*", ms=15, mfc="#f6b400", mec="#7a5200",
            mew=0.8, zorder=z, clip_on=False)
    if label:
        ax.annotate("AP", (ap[0], ap[1]), textcoords="offset points", xytext=(8, -12),
                    fontsize=8, color="#7a5200", zorder=z)


def _pts(ax, xy, split, thr, val=None, level=True, ms=34, ring=False):
    for k, (mask, mk, col) in enumerate([
            (split == 0, "o", "#1f6fd0"), (split == 1, "s", "#d64545"),
            (split == 2, "X", "#333333")]):
        if mask.sum() == 0:
            continue
        c = col
        if level and val is not None and k < 2:
            c = [LCOL[int(v) - 1] for v in level_of(val[mask], thr)]
        ax.scatter(xy[mask, 0], xy[mask, 1], s=ms, marker=mk,
                   facecolor=c if not ring else "none", edgecolor="#222" if k == 2 else "white",
                   linewidths=0.9, zorder=7 if not ring else 9)


def _field_img(ax, field, mask):
    m = np.where(mask > 0, field, np.nan)
    im = ax.imshow(m, origin="lower", extent=(0, C.DOMAIN, 0, C.DOMAIN),
                   cmap=CMAP_DB, vmin=-48, vmax=-32, interpolation="nearest", zorder=1)
    return im


def _level_img(ax, lv, mask):
    rgba = np.zeros(lv.shape + (4,))
    for i in range(5):
        rgba[lv == i + 1] = matplotlib.colors.to_rgba(LCOL[i])
    rgba[mask <= 0, :] = (1, 1, 1, 0)
    ax.imshow(rgba, origin="lower", extent=(0, C.DOMAIN, 0, C.DOMAIN),
              interpolation="nearest", zorder=1)


def _level_contour(ax, field, mask, thr):
    """按阈值画等高线分区（比格心色块更接近真实空间分布）。"""
    X, Y = cell_centers()
    f = np.where(mask > 0, field, np.nan)
    levels = [-70.0] + [float(t) for t in thr] + [-10.0]
    ax.contourf(X, Y, f, levels=levels, colors=LCOL, zorder=1, antialiased=True)


def fig_dataset(real):
    xy, rssi, ap, thr = real["xy"], real["rssi"], real["ap"], real["thresholds"]
    tr, te, valid = real["train_idx"], real["test_idx"], real["valid"]
    split = np.full(len(xy), 0)
    split[te] = 1
    split[~valid] = 2

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    for ax in axes:
        _room(ax)

    ax = axes[0]
    for x in np.arange(0.9, 17, 2):
        if x <= C.ROOM_W:
            ax.plot([x, x], [0, C.ROOM_H], color="#dddddd", lw=0.7, ls=":", zorder=2)
    for y in np.arange(0, 12.1, 2):
        ax.plot([0, C.ROOM_W], [y, y], color="#dddddd", lw=0.7, ls=":", zorder=2)
    _pts(ax, xy, split, thr, level=False)
    _ap(ax, ap)
    ax.scatter([], [], s=30, marker="o", facecolor="#1f6fd0", edgecolor="white", label=f"训练点 {len(tr)} 个 (70%)")
    ax.scatter([], [], s=30, marker="s", facecolor="#d64545", edgecolor="white", label=f"测试点 {len(te)} 个 (30%)")
    ax.scatter([], [], s=30, marker="X", facecolor="#333", edgecolor="#222", label="异常点 ×1（已剔除）")
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9, borderpad=0.5)
    _setup(ax, "(a) 42 个实测点 · 按强度分层 70/30 划分")

    ax = axes[1]
    obs = np.where(valid)[0]
    f = deploy_field(xy[obs], rssi[obs], ap)
    im = _field_img(ax, f, room_mask())
    _pts(ax, xy, split, thr, val=rssi, level=True)
    _ap(ax, ap)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("RSSI (dBm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _setup(ax, "(b) 实测信号强度分布（点颜色 = 真实五级）")
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=LCOL[i], mec="none", label=LEVEL_NAMES[i])
               for i in range(4, -1, -1)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.03))

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_dataset.png"), bbox_inches="tight")
    plt.close(fig)


def fig_methods(real, res, mean, std):
    xy, rssi, ap, thr = real["xy"], real["rssi"], real["ap"], real["thresholds"]
    tr, te, valid = real["train_idx"], real["test_idx"], real["valid"]
    split = np.full(len(xy), 0)
    split[te] = 1
    split[~valid] = 2
    pack = real_inputs(tr, mean, std)

    panels = []
    panels.append(("基线 · IDW 插值", baseline_idw(pack, _grid_xy()).reshape(C.GRID_N, C.GRID_N)))
    for key, label in METHODS:
        model = M.build(key)
        import torch
        model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, key, "weights.pt"), map_location="cpu"))
        model.eval()
        panels.append((label, predict_field(model, pack["x"], mean, std, key)))

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0))
    for ax, (label, field) in zip(axes.ravel(), panels):
        _room(ax)
        im = _field_img(ax, field, room_mask())
        _pts(ax, xy, split, thr, val=rssi, level=True, ms=26, ring=True)
        _ap(ax, ap, label=False)
        m = metrics(rssi[te], sample_bilinear(field, xy[te]), level_of(rssi[te], thr),
                    level_of(sample_bilinear(field, xy[te]), thr))
        _setup(ax, f"{label}\n测试 RMSE {m['rmse']:.2f} dB · 分级命中 {m['level_acc']*100:.0f}%")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.015)
    cb.set_label("预测 RSSI (dBm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("六个模型在同一组 29 个输入点上的全屋预测（圆环 = 12 个隐藏测试点）",
                 fontsize=11, y=0.99)
    fig.savefig(os.path.join(FIGS, "fig2_methods.png"), bbox_inches="tight")
    plt.close(fig)


def _grid_xy():
    X, Y = cell_centers()
    return np.stack([X.ravel(), Y.ravel()], -1)


def fig_levels(real, res):
    xy, rssi, ap, thr = real["xy"], real["rssi"], real["ap"], real["thresholds"]
    tr, te, valid = real["train_idx"], real["test_idx"], real["valid"]
    split = np.full(len(xy), 0)
    split[te] = 1
    split[~valid] = 2
    obs = np.where(valid)[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    # 左：真实等级（全部有效点插值）
    f = deploy_field(xy[obs], rssi[obs], ap)
    ax = axes[0]
    _room(ax)
    _level_contour(ax, f, room_mask(), thr)
    _pts(ax, xy, split, thr, val=rssi, level=True, ms=30)
    _ap(ax, ap)
    _setup(ax, "(a) 真实五级（41 个有效点插值）")

    # 右：部署模型（29 训练点）
    m = M.build("pgmpr")
    import torch
    m.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, "pgmpr", "weights.pt"), map_location="cpu"))
    m.eval()
    pack = real_inputs(tr, -56.98366, 9.98995)
    f2 = predict_field(m, pack["x"], -56.98366, 9.98995, "pgmpr", tta=True)
    ax = axes[1]
    _room(ax)
    _level_contour(ax, f2, room_mask(), thr)
    _pts(ax, xy, split, thr, val=rssi, level=True, ms=30, ring=True)
    _ap(ax, ap)
    _setup(ax, "(b) 方案五 + TTA 预测五级（只用 29 个训练点）")
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=LCOL[i], mec="none", label=LEVEL_NAMES[i])
               for i in range(4, -1, -1)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_levels.png"), bbox_inches="tight")
    plt.close(fig)


def fig_parity(real, res):
    xy, rssi, thr = real["xy"], real["rssi"], real["thresholds"]
    te = real["test_idx"]
    rows = [r for r in res["results"] if not r["key"].endswith("_tta")]
    keep = ["idw", "gp", "ae", "unet", "resunet", "pgmpr"]
    rows = [r for r in rows if r["key"] in keep]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.0))
    lo, hi = -49, -31
    for ax, r in zip(axes.ravel(), rows):
        ax.plot([lo, hi], [lo, hi], color="#bbb", lw=1, ls="--")
        ax.scatter(rssi[te], r["pred"], s=36, c="#1f6fd0", edgecolor="white", lw=0.8, zorder=3)
        e = np.array(r["pred"]) - rssi[te]
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_xlabel("实测 RSSI (dBm)", fontsize=8)
        ax.set_ylabel("预测 RSSI (dBm)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{r['label']}\nRMSE {r['rmse']:.2f} dB · 偏差 {e.mean():+.2f} dB",
                     fontsize=9)
    fig.suptitle("12 个隐藏测试点上的预测-实测一致性", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_parity.png"), bbox_inches="tight")
    plt.close(fig)


def fig_bar(real, res):
    rows = [r for r in res["results"]]
    order = ["idw", "deploy", "nearest", "gp", "ae", "unet", "resunet", "unet_scatter", "pgmpr", "pgmpr_tta"]
    rows = sorted([r for r in rows if r["key"] in order], key=lambda r: -r["rmse"])
    labels = [r["label"].replace("（TTA）", " +TTA").replace("方案", "S").replace(" · ", "·") for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))
    ax = axes[0]
    colors = ["#c1442e" if r["key"] == "deploy" else
              ("#9a9a9a" if r["kind"] == "baseline" else "#2f6f4e") for r in rows]
    b = ax.barh(labels, [r["rmse"] for r in rows], color=colors, height=0.62)
    for rect, r in zip(b, rows):
        ax.text(rect.get_width() + 0.05, rect.get_y() + rect.get_height() / 2,
                f"{r['rmse']:.2f}", va="center", fontsize=8, color="#333")
    ax.axvline(min(r["rmse"] for r in rows), color="#d64545", lw=1, ls=":")
    ax.set_xlabel("测试 RMSE (dB)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title("(a) 12 个测试点的 RMSE（越小越好）", fontsize=10)
    ax.grid(axis="x", alpha=0.25, lw=0.6)

    ax = axes[1]
    keys = ["idw", "ae", "unet", "resunet", "unet_scatter", "pgmpr", "pgmpr_tta"]
    rows2 = [next(r for r in res["results"] if r["key"] == k) for k in keys]
    short = ["IDW", "S1 AE", "S2 U-Net", "S3 ResUNet", "S4 Scatter", "S5 PGMPR", "S5 +TTA"]
    x = np.arange(len(rows2))
    w = 0.26
    ax.bar(x - w, [r["level_acc"] * 100 for r in rows2], w, label="五级完全命中", color="#2f6f4e")
    ax.bar(x, [r["level_acc_pm1"] * 100 for r in rows2], w, label="允许±1 级", color="#8ab17d")
    ax.bar(x + w, [r["binary_acc"] * 100 for r in rows2], w, label="好/差 二分类", color="#e0a33c")
    ax.axhline(100 / 5, color="#d64545", lw=1, ls=":")
    ax.text(len(rows2) - 0.5, 100 / 5 + 1.5, "五级随机猜测 20%", fontsize=7, color="#d64545", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=7.5, rotation=18)
    ax.set_ylim(0, 105)
    ax.set_ylabel("准确率 (%)", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("(b) 五级分级与二分类准确率", fontsize=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_metrics.png"), bbox_inches="tight")
    plt.close(fig)


def fig_scan(res):
    scan = res["scan"]
    ks = sorted({s["k"] for s in scan})
    name_map = {
        "idw": ("IDW 基线", "#9a9a9a", "--", None),
        "gp": ("趋势+RBF", "#555555", "--", None),
        "ae": ("S1 AE", "#1f6fd0", "-", "o"),
        "unet": ("S2 U-Net", "#e07a5f", "-", "o"),
        "resunet": ("S3 ResUNet", "#8ab17d", "-", "o"),
        "unet_scatter": ("S4 Scatter", "#e0a33c", "-", "o"),
        "pgmpr": ("S5 PGMPR", "#2f6f4e", "-", "o"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, (label, col, ls, mk) in name_map.items():
        ys = [next(s["rmse_mean"] for s in scan if s["k"] == k and s["key"] == key) for k in ks]
        er = [next(s["rmse_std"] for s in scan if s["k"] == k and s["key"] == key) for k in ks]
        ax.errorbar(ks, ys, yerr=er, label=label, color=col, ls=ls,
                    marker=mk, lw=1.5, ms=4, capsize=2.5)
    ax.set_xlabel("输入观测点数 K", fontsize=9)
    ax.set_ylabel("测试 RMSE (dB)", fontsize=9)
    ax.set_title("稀疏布点敏感性：K 个随机输入点（6 次重复，误差棒=标准差）", fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_scan.png"), bbox_inches="tight")
    plt.close(fig)


def _conf(ax, y_true, y_pred, title):
    cm = np.zeros((5, 5), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[int(a) - 1, int(b) - 1] += 1
    ax.imshow(cm, cmap="Greens", vmin=0, vmax=max(cm.max(), 1))
    for i in range(5):
        for j in range(5):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() * 0.55 else "#222")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f"L{i+1}" for i in range(5)], fontsize=8)
    ax.set_yticklabels([f"L{i+1}" for i in range(5)], fontsize=8)
    ax.set_xlabel("预测等级", fontsize=8)
    ax.set_ylabel("真实等级", fontsize=8)
    ax.set_title(title, fontsize=9)


def fig_confusion(real, res):
    te = real["test_idx"]
    rssi, thr = real["rssi"], real["thresholds"]
    best = next(r for r in res["results"] if r["key"] == "pgmpr_tta")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    _conf(axes[0], level_of(rssi[te], thr), level_of(np.array(best["pred"]), thr),
          f"方案五 + TTA：完全命中 {best['level_acc']*100:.0f}% / ±1 级 {best['level_acc_pm1']*100:.0f}%")
    dep = next(r for r in res["results"] if r["key"] == "deploy")
    _conf(axes[1], level_of(rssi[te], thr), level_of(np.array(dep["pred"]), thr),
          f"部署模型（物理+RBF/IDW 残差）：命中 {dep['level_acc']*100:.0f}% / ±1 级 {dep['level_acc_pm1']*100:.0f}%")
    fig.suptitle("五级混淆矩阵（12 个隐藏测试点）", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig7_confusion.png"), bbox_inches="tight")
    plt.close(fig)


def fig_train():
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    cols = {"ae": "#1f6fd0", "unet": "#e07a5f", "resunet": "#8ab17d",
            "unet_scatter": "#e0a33c", "pgmpr": "#2f6f4e"}
    names = {"ae": "S1 AE", "unet": "S2 U-Net", "resunet": "S3 ResUNet",
             "unet_scatter": "S4 Scatter", "pgmpr": "S5 PGMPR"}
    for key, col in cols.items():
        p = os.path.join(C.MODEL_DIR, key, "meta.json")
        if not os.path.exists(p):
            continue
        meta = json.load(open(p, encoding="utf-8"))
        h = meta["history"]
        ep = [x["epoch"] for x in h]
        axes[0].plot(ep, [x["train"] for x in h], color=col, lw=1.5, label=names[key])
        axes[1].plot(ep, [x["val"] for x in h], color=col, lw=1.5, label=names[key])
    for ax, t in zip(axes, ["(a) 仿真训练集损失", "(b) 仿真验证集损失"]):
        ax.set_yscale("log")
        ax.set_xlabel("epoch", fontsize=8.5)
        ax.set_ylabel("掩码 MSE（归一化）", fontsize=8.5)
        ax.set_title(t, fontsize=9.5)
        ax.grid(alpha=0.25, lw=0.6)
        ax.tick_params(labelsize=8)
    axes[1].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig8_training.png"), bbox_inches="tight")
    plt.close(fig)


def fig_deploy(real):
    xy, rssi, ap, thr = real["xy"], real["rssi"], real["ap"], real["thresholds"]
    valid = real["valid"]
    te = real["test_idx"]
    obs = np.where(valid)[0]
    f = deploy_field(xy[obs], rssi[obs], ap)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    _room(ax)
    _level_contour(ax, f, room_mask(), thr)
    _ap(ax, ap)
    ax.scatter(xy[obs, 0], xy[obs, 1], s=22, c="white", edgecolor="#333", lw=0.7, zorder=7)
    ax.scatter(xy[te, 0], xy[te, 1], s=110, facecolor="none", edgecolor="#ffffff", lw=1.8, zorder=8)
    ax.scatter(xy[te, 0], xy[te, 1], s=110, facecolor="none", edgecolor="#222", lw=0.8, zorder=8)
    _setup(ax, "部署模型全屋五级预测（全部 41 个有效测点）")
    handles = [plt.Line2D([], [], marker="s", ls="", mfc=LCOL[i], mec="none", label=LEVEL_NAMES[i])
               for i in range(4, -1, -1)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5,
              frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig9_deploy.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    real = load_real()
    res = json.load(open(os.path.join(C.OUT_DIR, "metrics.json"), encoding="utf-8"))
    mean, std = res["mean"], res["std"]

    # 补齐部署模型的评估行
    obs = real["train_idx"]
    f = deploy_field(real["xy"][obs], real["rssi"][obs], real["ap"])
    te = real["test_idx"]
    pred = sample_bilinear(f, real["xy"][te])
    thr = [float(t) for t in real["thresholds"]]
    m = metrics(real["rssi"][te], pred, level_of(real["rssi"][te], thr), level_of(pred, thr))
    if not any(r["key"] == "deploy" for r in res["results"]):
        res["results"].append({"key": "deploy", "label": "部署 · 物理趋势+实测残差",
                               "kind": "deploy", **m, "pred": pred.tolist()})
        with open(os.path.join(C.OUT_DIR, "metrics.json"), "w", encoding="utf-8") as fp:
            json.dump(res, fp, ensure_ascii=False, indent=2)

    fig_dataset(real)
    fig_methods(real, res, mean, std)
    fig_levels(real, res)
    fig_parity(real, res)
    fig_bar(real, res)
    fig_scan(res)
    fig_confusion(real, res)
    fig_train()
    fig_deploy(real)
    print("figures ->", FIGS)
    for fn in sorted(os.listdir(FIGS)):
        print("  ", fn, round(os.path.getsize(os.path.join(FIGS, fn)) / 1024), "KB")


if __name__ == "__main__":
    main()
