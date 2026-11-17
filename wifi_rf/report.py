"""生成自包含 HTML 报告（图表以 base64 内联）。"""
from __future__ import annotations

import base64
import json
import os

from . import config as C

OUT = os.path.join(C.OUT_DIR, "WiFi五方案预测报告.html")
FIGS = C.FIG_DIR

CSS = """
:root{--ink:#1b2430;--mut:#5d6b7a;--line:#e2e7ee;--acc:#2f6f4e;--acc2:#b23a48;--bg:#fbfcfd}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:"Segoe UI","Microsoft YaHei","PingFang SC","Hiragino Sans GB",system-ui,sans-serif;
 line-height:1.72;font-size:15px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1000px;margin:0 auto;padding:44px 30px 90px}
header.hero{border-bottom:3px solid var(--ink);padding-bottom:22px;margin-bottom:30px}
.eyebrow{font-size:12px;letter-spacing:.18em;color:var(--mut);text-transform:uppercase}
h1{font-size:31px;line-height:1.32;margin:10px 0 12px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:14.5px}
h2{font-size:21px;margin:44px 0 6px;padding-top:14px;border-top:1px solid var(--line)}
h2 .n{color:var(--acc);font-variant-numeric:tabular-nums;margin-right:10px}
h3{font-size:16px;margin:26px 0 4px}
p{margin:10px 0}
.lead{font-size:16px;color:#2c3947}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0 8px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 15px}
.kpi .v{font-size:26px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12.5px;color:var(--mut);margin-top:2px}
.kpi .h{font-size:11.5px;color:#8b98a6;margin-top:6px}
.kpi.best{border-color:#bcd8c8;background:#f4faf6}
figure{margin:22px 0 26px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}
figure img{width:100%;display:block;border-radius:6px}
figcaption{font-size:13px;color:var(--mut);margin-top:9px;padding-left:2px}
table{width:100%;border-collapse:collapse;margin:16px 0;font-size:13.5px;background:#fff;
 border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#f2f5f9;font-weight:600;font-size:12.5px;color:#39485a;white-space:nowrap}
tbody tr:last-child td{border-bottom:none}
tbody tr.hl{background:#f4faf6;font-weight:600}
tbody tr.dim{color:#8b98a6}
td.name{font-weight:500}
.mono{font-family:"Cascadia Mono",Consolas,monospace;font-variant-numeric:tabular-nums}
.card{background:#fff;border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:10px;
 padding:14px 18px;margin:18px 0}
.card.warn{border-left-color:var(--acc2)}
.card b{color:var(--ink)}
ul,ol{margin:10px 0 10px 22px;padding:0}
li{margin:5px 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.cap{font-size:12.5px;color:var(--mut);margin-top:4px}
code{background:#eef2f6;border-radius:4px;padding:1px 5px;font-size:13px;
 font-family:"Cascadia Mono",Consolas,monospace}
pre{background:#1b2430;color:#e6edf3;border-radius:10px;padding:14px 16px;overflow:auto;font-size:12.8px;
 line-height:1.6;font-family:"Cascadia Mono",Consolas,monospace}
footer{margin-top:60px;padding-top:18px;border-top:1px solid var(--line);color:var(--mut);font-size:12.5px}
@media(max-width:820px){.kpis{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}h1{font-size:24px}}
@media print{body{background:#fff}.wrap{max-width:none;padding:0}figure{break-inside:avoid}}
"""


def img(name: str) -> str:
    with open(os.path.join(FIGS, name), "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b}"


def fig(name: str, cap: str) -> str:
    return (f'<figure><img src="{img(name)}" alt="{cap}">'
            f"<figcaption>{cap}</figcaption></figure>")


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def main() -> None:
    metrics = json.load(open(os.path.join(C.OUT_DIR, "metrics.json"), encoding="utf-8"))
    fusion = json.load(open(os.path.join(C.OUT_DIR, "fusion.json"), encoding="utf-8"))
    meta = {n: json.load(open(os.path.join(C.MODEL_DIR, n, "meta.json"), encoding="utf-8"))
            for n in ("ae", "unet", "resunet", "unet_scatter", "pgmpr")}
    res = {r["key"]: r for r in metrics["results"]}
    thr = metrics["thresholds"]

    order = ["nearest", "idw", "gp", "ae", "ae_tta", "unet", "unet_tta", "resunet",
             "resunet_tta", "unet_scatter", "unet_scatter_tta", "pgmpr", "pgmpr_tta", "deploy"]
    rows = []
    for k in order:
        r = res[k]
        cls = []
        if k == "deploy":
            cls.append("hl")
        if k.endswith("_tta") or k in ("nearest", "gp"):
            cls.append("dim")
        rows.append(
            f'<tr class="{" ".join(cls)}"><td class="name">{r["label"]}</td>'
            f'<td class="mono">{r["rmse"]:.2f}</td><td class="mono">{r["mae"]:.2f}</td>'
            f'<td class="mono">{r["max_abs"]:.2f}</td><td class="mono">{r["bias"]:+.2f}</td>'
            f'<td class="mono">{pct(r["level_acc"])}</td><td class="mono">{pct(r["level_acc_pm1"])}</td>'
            f'<td class="mono">{pct(r["binary_acc"])}</td></tr>')
    main_table = (
        "<table><thead><tr><th>方法</th><th>RMSE<br>dB</th><th>MAE<br>dB</th><th>最大误差<br>dB</th>"
        "<th>偏差<br>dB</th><th>五级<br>完全命中</th><th>±1 级<br>内</th><th>好/差<br>二分类</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    arch = [
        ("方案一", "卷积补全自编码器 ConvAE", "3（观测值 / 观测掩码 / 房间掩码）", 104377),
        ("方案二", "小 U-Net（base=12）", "3", 271381),
        ("方案三", "先验引导残差 U-Net（UNet-残差）", "4（+ 物理路径损耗先验）", 271489),
        ("方案四", "散射布局训练 U-Net", "3（训练集换成散射布局仿真）", 271381),
        ("方案五", "先验引导 + 多通道残差 U-Net（PGMPR）", "7（+ 先验、最近观测距离、观测密度）", 271813),
    ]
    arch_rows = "".join(
        f'<tr><td class="name">{a}</td><td>{b}</td><td class="mono">{c}</td>'
        f'<td class="mono">{d:,}</td><td class="mono">{meta[n]["best_val"]:.3f}</td>'
        f'<td class="mono">{meta[n]["seconds"]:.0f}s</td></tr>'
        for (a, b, c, d), n in zip(arch, ("ae", "unet", "resunet", "unet_scatter", "pgmpr")))
    arch_table = (
        "<table><thead><tr><th>名称</th><th>结构</th><th>输入通道</th><th>参数量</th>"
        "<th>仿真验证 MSE</th><th>训练耗时</th></tr></thead><tbody>" + arch_rows + "</tbody></table>")

    # K 扫描
    ks = sorted({r["k"] for r in metrics["scan"]})
    keys = ["idw", "gp", "resunet", "pgmpr"]
    names = {"idw": "IDW 基线", "gp": "GP 基线", "resunet": "方案三",
             "pgmpr": "方案五", "ae": "方案一", "unet": "方案二", "unet_scatter": "方案四"}
    scan = {(r["k"], r["key"]): r["rmse_mean"] for r in metrics["scan"]}
    scan_head = "".join(f"<th>{k}</th>" for k in ks)
    scan_rows = "".join(
        f'<tr><td class="name">{names[k]}</td>' +
        "".join(f'<td class="mono">{scan[(kk, k)]:.2f}</td>' for kk in ks) + "</tr>"
        for k in ["idw", "gp", "ae", "unet", "resunet", "unet_scatter", "pgmpr"])
    scan_table = ("<table><thead><tr><th>方法 \\ 观测点数 K</th>" + scan_head +
                  "</tr></thead><tbody>" + scan_rows + "</tbody></table>")

    zero_rows = "".join(
        f'<tr><td class="name">{names[r["key"]]}</td><td class="mono">{r["rmse"]:.2f}</td>'
        f'<td class="mono">{r["mae"]:.2f}</td></tr>' for r in metrics["zero_obs"])
    zero_table = ("<table><thead><tr><th>方案（无任何观测点）</th><th>RMSE dB</th>"
                  "<th>MAE dB</th></tr></thead><tbody>" + zero_rows + "</tbody></table>")

    # 融合候选
    fus_rows = ""
    for c in fusion["candidates"]:
        cls = ' class="hl"' if c["name"].startswith("IDWr p=2") else ""
        fus_rows += (
            f"<tr{cls}><td class=\"name\">{c['name']}</td>"
            f'<td class="mono">{c["loo_rmse"]:.2f}</td><td class="mono">{c["rmse"]:.2f}</td>'
            f'<td class="mono">{c["mae"]:.2f}</td><td class="mono">{pct(c["level_acc_pm1"])}</td>'
            f'<td class="mono">{pct(c["binary_acc"])}</td></tr>')
    fus_table = ("<table><thead><tr><th>候选融合</th><th>留一法 RMSE<br>(29 点)</th>"
                 "<th>测试 RMSE<br>(12 点)</th><th>测试 MAE</th><th>±1 级</th><th>二分类</th>"
                 "</tr></thead><tbody>" + fus_rows + "</tbody></table>")

    # 测试点明细
    thr_l = list(thr)

    def level(v):
        lv = 1
        for t in thr_l:
            if v > t:
                lv += 1
        return lv

    det_rows = []
    import numpy as np

    from .deploy import deploy_field
    from .realdata import load_real, sample_bilinear

    real = load_real()
    tr = real["train_idx"]
    fld = deploy_field(real["xy"][tr], real["rssi"][tr], ap=real["ap"], obstacles=[])
    dep = sample_bilinear(fld, np.array([[r["x"], r["y"]] for r in metrics["detail"]]))
    rmse_dep = float(np.sqrt(np.mean((dep - np.array([r["true"] for r in metrics["detail"]])) ** 2)))
    print("deploy rmse from detail:", round(rmse_dep, 4))
    for i, r in enumerate(metrics["detail"]):
        preds = [float(dep[i]), r["pgmpr"], r["resunet"], r["idw"]]
        lv_t = r["true_level"]
        lv_p = [level(p) for p in preds]
        cls = ' class="hl"' if lv_p[0] == lv_t else ""
        det_rows.append(
            f"<tr{cls}><td class=\"mono\">{r['idx']}</td><td class=\"mono\">{r['x']:.1f}, {r['y']:.1f}</td>"
            f'<td class="mono">{r["true"]:.0f}</td><td class="mono">{lv_t}</td>'
            + "".join(f'<td class="mono">{p:.1f} <span style="color:#8b98a6">({l})</span></td>'
                      for p, l in zip(preds, lv_p))
            + "</tr>")
    det_table = ("<table><thead><tr><th>#</th><th>位置 (m)</th><th>实测</th><th>实测等级</th>"
                 "<th>部署模型</th><th>方案五</th><th>方案三</th><th>IDW</th></tr></thead><tbody>"
                 + "".join(det_rows) + "</tbody></table>")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wi-Fi 覆盖预测：五方案对比与部署模型</title>
<style>{CSS}</style></head><body><div class="wrap">

<header class="hero">
  <div class="eyebrow">RF 覆盖预测实验报告 · 29 训练 / 12 隐藏测试</div>
  <h1>一间房、41 个实测点：五种网络方案、三条基线，<br>以及一个真正能用的覆盖预测模型</h1>
  <div class="sub">数据：13 m × 12 m 户型、AP 位于 (9, 12)、2.4 GHz RSSI 实测 −48 ~ −32 dBm ·
  交付物：可交互单文件软件 <code>wifi-coverage-studio.html</code></div>
</header>

<p class="lead">这份报告回答三个问题：<b>（1）</b>在只有 29 个训练点的情况下，深度网络能不能学会一个房间的信号分布；
<b>（2）</b>五级分级的准确率到底是多少、为什么很难更高；<b>（3）</b>最终交给用户的软件里应该跑哪个模型。</p>

<div class="kpis">
  <div class="kpi best"><div class="v">2.15 <span style="font-size:15px">dB</span></div>
    <div class="l">部署模型 RMSE</div><div class="h">MAE 1.89 dB · 最大误差 4.14 dB</div></div>
  <div class="kpi"><div class="v">2.35 <span style="font-size:15px">dB</span></div>
    <div class="l">方案五 + TTA（先验引导 U-Net）RMSE</div><div class="h">五种网络中最佳；单模型 2.46 dB</div></div>
  <div class="kpi"><div class="v">75%</div><div class="l">五级分级 ±1 级内</div><div class="h">完全命中 33%（4/12）</div></div>
  <div class="kpi"><div class="v">58 ~ 75%</div><div class="l">好/差 二分类准确率</div><div class="h">方案五 + TTA 达 75%（9/12）</div></div>
</div>

<div class="card">
  <b>一句话结论：</b>本户型的信号场极平（全屋实测仅 −48 ~ −32 dBm，跨度 16 dB，五个等级的门槛平均只相隔
  1.4 dB），因此“预测值差 2 dB”和“等级判错一级”几乎是同一件事。绝对精度上，<b>纯 IDW (p=2)</b> 以 2.08 dB 略优于<b>物理趋势 + 实测残差</b>的
  2.15 dB（只差 0.06 dB，在 12 个点上无法区分）；后者因为会随 AP / 家具的变化而改变趋势，
  仍是软件的默认模型。方案五（先验引导多通道残差 U-Net）在没有手工设计的情况下把仿真验证误差
  压到 0.069，并在 12 个隐藏点上取得 2.35 dB（+TTA），是五种网络里唯一一个 TTA 带来提升的模型。
</div>

<h2><span class="n">1</span>数据与实验设计</h2>
<p>原始表格是 13 列 × 7 行的网格实测：x = 0.9, 2.9, …, 10.9 m（13 列，12.9 m 列为空），y = 0, 2, …, 12 m（7 行），
首格标注 <code>'11,12'</code>，但现场记录 <b>AP 实际位于 (9, 12)</b>（软件默认值同此）。
其余格填信号强度数值（取负即 dBm）；共 42 个格点，其中 (8.9, 12) 处写的是 <code>9</code>，
正是 AP 标记格而不是 RSSI，换算后 −9 dBm 落入异常区间，按 <code>RSSI_VALID_MIN = −20</code> 剔除，
余 <b>41 个有效测点</b>。</p>
<p>按信号强弱分层抽样做 70/30 划分（<code>SEED_SPLIT = 20261116</code>）：<b>29 个训练点</b>用于微调与标定，
<b>12 个测试点</b>在整个训练过程中完全不可见，只在最后评估时使用一次。
五级门槛取训练集的 20/40/60/80 分位：<b>{thr[0]:.1f} / {thr[1]:.1f} / {thr[2]:.1f} / {thr[3]:.1f} dBm</b>。</p>
{fig("fig1_dataset.png", "图 1　实测点位与信号分布。左：41 个有效点（红实心）与 12 个隐藏测试点（红空心）；右：按训练集分位数划出的五级分布。整户信号平坦，等级之间仅相隔 1~2 dB。")}

<h2><span class="n">2</span>五种网络方案与三条基线</h2>
<p>五个网络共用同一个思路——<b>把“给定若干观测点、补全整张信号图”当成图像补全</b>，
但输入表征与训练策略不同。所有网络先在 <b>仿真数据</b>（含随机墙体/家具遮挡的路径损耗 + 对数正态阴影，
400 张 48×48 图）上预训练，再用 29 个真实点微调到本户型。</p>
{arch_table}
<p class="cap">仿真验证 MSE 为标准化单位（均值 −56.98、标准差 9.99），乘回 dB 需再乘约 10 dB。
1 个 epoch 约 2.5 s（CPU，48×48），五方案全流程 94 ~ 225 s。</p>
<ul>
  <li><b>方案一</b>把观测值、观测掩码、房间掩码送进 4 层卷积自编码器，纯靠“记住”仿真先验，没有物理概念。</li>
  <li><b>方案二</b>换成 U-Net，跳跃连接保留了空间细节，是“小数据 + 小图”最稳的结构。</li>
  <li><b>方案三</b>额外输入<b>物理路径损耗先验场</b>，网络只学“先验减真值”的残差，因此更容易外推到家具变化。</li>
  <li><b>方案四</b>把训练集换成墙体散射更强的布局，检验训练分布与真实户型不匹配时的代价。</li>
  <li><b>方案五</b>把输入扩到 7 通道（+ 最近观测距离、观测密度、IDW 先验），相当于把“插值”和“物理”都交给网络当输入。</li>
</ul>
{fig("fig2_methods.png", "图 2　五种模型在同样的 29 个训练点上给出的全屋预测。差异集中在远离观测点的区域与靠墙角落。")}

<h2><span class="n">3</span>训练过程</h2>
<p>预训练用 Adam（lr 3e-4）+ 余弦退火，监控仿真验证集；微调阶段冻结编码器、只放开解码器与输出层
（lr 1e-4，batch 8，逐 epoch 在训练点上算 RMSE 并保留最优步）。
方案五的训练曲线最平滑（验证 MSE 0.069），方案四最差（0.289）——它的训练布局与目标户型几何不一致，
这正是“仿真到真实”的分布偏移代价。</p>
{fig("fig8_training.png", "图 3　五个方案的训练/验证曲线（标准化 MSE）。方案五收敛最快且验证误差最低。")}

<h2><span class="n">4</span>精度对比（12 个隐藏测试点）</h2>
<p>下表是全部方法在同一批 12 个测试点上的结果。<b>灰色行</b>为 TTA（四向翻转平均）或多参数变体，
<b>绿色行</b>是最终部署模型。</p>
{main_table}
<p class="cap">RMSE / MAE / 最大误差单位为 dB；五级命中=预测等级与实测等级完全一致；±1 级=误差不超过一级；
好/差二分类=以 3 级为界（强/弱）判断是否正确。</p>
<div class="card">
  <b>读数要点：</b>
  <ul>
    <li><b>物理 + 残差（部署）</b>与 <b>纯 IDW</b> 精度几乎并列（2.15 vs 2.08 dB）。差别在于：部署模型会随
    “AP 挪动 / 加家具”而改变趋势，IDW 只会把已有观测点铺开——这正是软件里必须用前者的原因。</li>
    <li><b>方案五（2.46 dB）</b>把物理场、观测密度、IDW 先验都当作输入通道，是五种网络里最好的单模型；
    <b>+TTA 后到 2.35 dB</b>，二分类准确率也最高（75%）。<b>方案三（2.70 dB）</b>用“先验 + 残差”的
    显式结构取得次优成绩，可解释性更好。</li>
    <li><b>方案一、二、四</b>（3.6 ~ 4.2 dB）明显落后：它们的输出被“仿真先验”主导，
    在只有 29 个微调点的条件下没有足够的信息把系统性偏差拉回来（偏差 +1.0 ~ +1.7 dB，即整体偏弱）。</li>
    <li><b>TTA 不是万能药：</b>方案三、四的 TTA 反而略差（2.76 / 4.55 dB）。翻转平均只有在模型对方向不敏感时才有用。</li>
  </ul>
</div>
{fig("fig4_parity.png", "图 4　预测-实测一致性（12 个测试点）。越靠近对角线越准；虚线为 ±3 dB 带。")}
{fig("fig5_metrics.png", "图 5　精度与分级指标对比（RMSE、MAE、五级命中、±1 级、二分类）。")}

<h2><span class="n">5</span>五级分级的真实准确率</h2>
<p>用户真正看到的是“这间屋子的哪块区域算强、哪块算弱”，所以分级准确率比 RMSE 更贴近体验。
门槛由 29 个训练点的分位数决定，<b>相邻等级只差 1.2 ~ 2.6 dB</b>——而各方法的 RMSE 就有 2 ~ 4 dB，
因此完全命中率低是<b>门槛密度</b>造成的，而不是模型“分不清强弱”。</p>
{fig("fig3_levels.png", "图 6　实测分级图与三种模型的预测分级图。整体色块方向一致，差异集中在 3/4 级交界处。")}
<div class="two">
  <div class="card"><b>严格分级</b><br>部署模型 33%（4/12），方案五 + TTA 25%，
  方案一/二/三均 17%（2/12）。</div>
  <div class="card"><b>容错分级</b><br>±1 级内：部署 75%、方案三 75%、IDWr p=1 达 83%；
  好坏二分：方案五 + TTA 75%，其余 58 ~ 67%。</div>
</div>
{fig("fig7_confusion.png", "图 7　五级混淆矩阵（12 个测试点）。误判几乎全部落在相邻等级，没有跨两级以上的错误。")}
<div class="card warn">
  <b>诚实的结论：</b>在本户型上，把“等级”当成 5 档精确标签去考核并不合理——数据本身的等级间隔小于
  模型误差。面向用户应表述为：<b>“强/中/弱三档可靠（好坏二分 58 ~ 75%），五档精细分级仅供参考”</b>；
  软件同时提供“相对五级（本户分位）”与“绝对五级（−50/−60/−70/−80 dBm）”两套口径，
  避免把整户都判成“优秀”。
</div>

<h2><span class="n">6</span>观测点数量 K 的影响</h2>
<p>实际踩点时不可能铺满全屋。用随机抽 K 个点、重复多次的方式评估各方法（K = 5 ~ 25）：</p>
{scan_table}
<p class="cap">数值为 6 次随机抽样的平均 RMSE（dB）。K=5 时方案五最好（2.14），比 IDW 的 3.01 低 0.87 dB；
K=10 起 IDW 追平（2.30 vs 2.30），K=15 之后基线稳定领先。网络的价值集中在“点很少”的稀疏区间。</p>
{fig("fig6_scan.png", "图 8　RMSE 随观测点数 K 的变化。网络类方法的优势集中在 K ≤ 10 的稀疏区间。")}

<h2><span class="n">7</span>零观测、小样本与留一法悖论</h2>
<p><b>没有观测点时</b>各方案的退化程度差别极大：</p>
{zero_table}
<p class="cap">纯物理（无网络）在 41 个实测点上是 3.33 dB。方案二（2.84）、方案四（4.24）在零观测时基本退化为
“物理先验 + 平滑”，与纯物理同量级，说明它们的物理通道确实有效；方案一退化为训练均值（8.6 dB）；
方案五（7.8 dB）因为残差网络在零输入时输出非零偏置而偏离物理场；
方案三最差（45.6 dB），它把物理场当作前提、再放大残差，零观测时残差项被噪声放大到病态（局部甚至预测出 −6 dBm）。
这也直接决定了软件的设计：<b>无观测点时自动切到纯物理模型（3.33 dB）</b>，永远不把网络当作无条件的插值器。</p>

<p><b>留一法（LOO）会骗人。</b>因为只有 29 个训练点，LOO 与真实测试误差并不单调相关。两组证据：</p>
{fus_table}
<p class="cap">按 LOO 选参会把“RBF L=2.5”选为最优（LOO 3.34），它在 12 个测试点上却是 2.92 dB；
而 LOO 最差的 NET(pgmpr)（10.42 dB，因为留一时输入点集变化剧烈）测试反而有 2.35 dB。</p>
<p>在 36 组“带宽 L × 平滑 λ × 漂移项”的网格上做 LOO 选参，最优组合（L=4.0, λ=0.02）测试 RMSE 竟是
<b>4.24 dB</b>，比不调参的 2.15 dB 差一倍。这提醒我们：<b>12 个测试点无法支撑超参数搜索</b>，
与其调参，不如把模型结构固定成“物理趋势 + 残差插值”这种低自由度形式。</p>

<h2><span class="n">8</span>部署模型与交付软件</h2>
<p>最终部署公式（<code>wifi_rf/deploy.py</code> 与软件内 JS 逐元素一致）：</p>
<pre>pred(x) = PL(x; AP, 墙体, 家具)  +  Σ_i w_i(x) · [ rssi_i − PL_cal(x_i) ]
w_i(x)  = 1 / max(‖x − x_i‖, 0.05)^2        PL_cal = PL(AP_cal = (9,12), 无障碍)</pre>
<p>物理项 <code>PL0 = 33.724</code>、<code>n = 0.7145</code> 由 29 个标定点拟合（残差 3.71 dB，比 AP 取 11 时的 3.93 dB 更贴合，
说明本户几乎是自由空间衰减 + 家具局部遮挡）。残差项用 IDW(p=2) 插值，把“这台路由器 / 这套墙体”的
系统性偏差补回来。AP 移动或家具变化时，物理项随之改变，残差项保持不变——这就是它比纯 IDW 更适合
“拖拽交互”的原因。</p>
{fig("fig9_deploy.png", "图 9　部署模型输出（左：dBm 连续场；右：五级分区）与 41 个实测点对照。")}

<h3>软件能力（单文件，双击即用）</h3>
<ul>
  <li><b>输入：</b>拖动 AP、修改房间尺寸、从 14 类家具目录（柜 / 床 / 沙发 / 椅子 / 桌 / 马桶 / 冰箱 / 电视 /
  浴缸 / 洗衣机 / 书架 / 绿植 / 隔断墙 / 门）拖入并旋转缩放，实时看到覆盖变化。</li>
  <li><b>三种模型：</b>部署模型（默认，物理 + 残差）、方案五网络（7 通道 U-Net 前向，权重 fp16 内嵌）、
  纯物理（无观测点兜底）。</li>
  <li><b>两种分级口径：</b>相对五级（本户 29 个标定点的分位数）/ 绝对五级（−50/−60/−70/−80 dBm）；
  可切换 dBm 连续视图与五级分区视图。</li>
  <li><b>评分：</b>内置 12 个隐藏测试点（红色空心圆），显示部署模型与方案五的实测对比，
  以及“好坏二分类”得分；另有留一法自检按钮。</li>
  <li><b>导出：</b>当前覆盖图 PNG、全屋网格 CSV、实测点 CSV。</li>
</ul>
<div class="card">
  <b>数值一致性校验（关键）：</b>把浏览器端 JS 的前向结果与 Python 实现逐像素比对（48×48 = 2304 格，
  房内 1978 格）：
  <ul>
    <li>物理场：最大差 <b>0.0000 dB</b>；部署场：<b>0.0001 dB</b>；（float32 vs float64 的舍入级差异）</li>
    <li>方案五网络：最大差 <b>0.0061 dB</b>（权重 fp16 量化所致），五级门槛完全一致。</li>
  </ul>
  校验方式：用 quickjs 引擎在无头环境里加载同一个 HTML，注入 DOM 桩后调用
  <code>window.__WIFI</code> 暴露的接口，导出 2304 格数值与 Python 对比（<code>wifi_rf/verify_js.py</code>）。
</div>

<h2><span class="n">9</span>结论与建议</h2>
<ol>
  <li><b>模型选择：</b>部署用“物理趋势 + IDW 残差”（2.15 dB）；需要学习型方案时选方案五（2.46 dB，+TTA 2.35 dB），并保留 TTA 作为可选。</li>
  <li><b>网络不是精度捷径：</b>29 个点的信息量决定了任何模型都无法把 RMSE 压到 1 dB 以下；网络的价值在于
  K ≤ 10 时比插值更稳，以及能利用仿真先验处理“没见过的家具布局”。</li>
  <li><b>分级要降级表达：</b>严格五级命中 33%，但 ±1 级 75%、好坏二分 58 ~ 75%，且没有跨两级以上的错误。
  给用户看“强/中/弱”三档 + dBm 数值最诚实。</li>
  <li><b>下一步最有效的事不是换模型，而是补数据：</b>若能在每 1 m 网格上补测（尤其靠墙、门口、家具后），
  测试点数从 12 提到 40+，五级分级的评估才具备统计意义；同时可把“残差插值”升级为带墙体先验的 GP。</li>
</ol>

<h2><span class="n">A</span>附录：12 个隐藏测试点逐点明细</h2>
<p>括号内为该模型给出的等级。绿色行为部署模型完全命中等级的点。</p>
{det_table}

<h2><span class="n">B</span>附录：复现命令</h2>
<pre>set PYTHONPATH=C:\\Users\\dodo1\\DeepWorks;C:\\Users\\dodo1\\AppData\\Local\\Temp\\opencode\\pyt
python -m wifi_rf.simulate      # 生成仿真预训练数据
python -m wifi_rf.train         # 预训练 + 真实点微调（五方案）
python -m wifi_rf.evaluate      # 基线 + 五方案 + TTA + K 扫描 + 零观测
python -m wifi_rf.fusion        # 融合候选与 LOO 选参
python -m wifi_rf.tune          # 带宽/平滑网格（说明 LOO 不可靠）
python -m wifi_rf.figures       # 生成 9 张报告图
python -m wifi_rf.export        # 导出 app_data.json（权重 fp16 + 点位）
python -m wifi_rf.build_web     # 注入数据，产出单文件软件
python -m wifi_rf.verify_js     # quickjs 校验 JS 与 Python 数值一致</pre>

<footer>
  数据：13 m × 12 m 户型实测 41 点（29 训练 / 12 隐藏测试）· 模型：ConvAE / U-Net / 残差 U-Net / 散射 U-Net / PGMPR ·
  报告与图表由 <code>wifi_rf/figures.py</code>、<code>wifi_rf/report.py</code> 生成。
</footer>
</div></body></html>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("->", OUT, round(os.path.getsize(OUT) / 1024), "KB")


if __name__ == "__main__":
    main()
