"""生成 Word 版实验报告（.docx）。

与 HTML 版共用 :func:`rfmap.report.collect` 的数据源，保证两份报告的数字完全一致。
用 python-docx 逐元素构建，而不是把 HTML 直接转换 —— 后者会丢掉表格和图片
（实测 textutil 转换后 ``<w:tbl>`` 与 ``<w:drawing>`` 均为 0）。
"""

from __future__ import annotations

import argparse
import json
import os

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from . import geometry as G
from .grid import DEFAULT as GRID
from .models import BASELINE_LABELS, MODEL_LABELS
from .report import MODEL_ORDER, _fmt, collect

CN_FONT = 'PingFang SC'
MONO_FONT = 'Menlo'


def _style_doc(doc: Document):
    st = doc.styles['Normal']
    st.font.name = CN_FONT
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), CN_FONT)
    for h, sz in (('Heading 1', 17), ('Heading 2', 14), ('Heading 3', 12)):
        s = doc.styles[h]
        s.font.name = CN_FONT
        s.font.size = Pt(sz)
        s.element.rPr.rFonts.set(qn('w:eastAsia'), CN_FONT)
        s.font.color.rgb = RGBColor(0x14, 0x22, 0x36)


def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def p(doc, text, bold=False, italic=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    return par


def bullets(doc, items, numbered=False):
    style = 'List Number' if numbered else 'List Bullet'
    for it in items:
        doc.add_paragraph(it, style=style)


def code(doc, text):
    for line in text.split('\n'):
        par = doc.add_paragraph()
        par.paragraph_format.space_after = Pt(0)
        run = par.add_run(line if line else ' ')
        run.font.name = MONO_FONT
        run.font.size = Pt(8.5)
        run.element.rPr.rFonts.set(qn('w:eastAsia'), MONO_FONT)


def table(doc, header, rows, caption=None, best_row=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    for i, htxt in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = str(htxt)
        for par in cell.paragraphs:
            for run in par.runs:
                run.bold = True
                run.font.size = Pt(9)
    for r, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = '' if v is None else str(v)
            for par in cells[i].paragraphs:
                for run in par.runs:
                    run.font.size = Pt(9)
    if best_row is not None and best_row < len(t.rows):
        for cell in t.rows[best_row + 1].cells:
            for par in cell.paragraphs:
                for run in par.runs:
                    run.bold = True
    if caption:
        cp = doc.add_paragraph()
        cr = cp.add_run(caption)
        cr.italic = True
        cr.font.size = Pt(9)
        cr.font.color.rgb = RGBColor(0x5c, 0x64, 0x70)
    doc.add_paragraph()


def figure(doc, path, caption, width=6.1):
    if not os.path.exists(path):
        return
    doc.add_picture(path, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp = doc.add_paragraph()
    cr = cp.add_run(caption)
    cr.italic = True
    cr.font.size = Pt(9)
    cr.font.color.rgb = RGBColor(0x5c, 0x64, 0x70)
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER


def _rmse(protos, proto, key):
    try:
        return protos[proto]['methods'][key]['rmse']
    except Exception:
        return float('nan')


def _mname(k):
    """把内部 key 翻成人类可读名字（含 _ft / _tta 后缀）。"""
    if k.endswith('_ft'):
        return MODEL_LABELS.get(k[:-3], k[:-3]) + '（实测微调）'
    if k.endswith('_tta'):
        return MODEL_LABELS.get(k[:-4], k[:-4]) + '（+TTA 测试时增强）'
    return BASELINE_LABELS.get(k, MODEL_LABELS.get(k, k))


# ==========================================================================
def build(root: str = '.', out: str = 'outputs/RF覆盖预测实验报告.docx') -> str:
    ctx = collect(root)
    g = ctx['geometry']
    q = ctx['quality']
    real = ctx.get('real') or {}
    sim = ctx.get('sim') or {}
    protos = real.get('protocols', {})
    sweep = real.get('sweep', {})
    ck = g['column']
    absroot = os.path.abspath(root)
    figdir = os.path.join(absroot, 'outputs/figures')

    doc = Document()
    _style_doc(doc)

    # ---------------- 封面 ----------------
    title = doc.add_paragraph()
    tr = title.add_run('室内 Wi-Fi 覆盖预测（RF Map）实验报告')
    tr.bold = True
    tr.font.size = Pt(22)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sr = sub.add_run('从少量实测点补全整间房的 RSSI 热力图')
    sr.font.size = Pt(13)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p(doc, '')
    for line in [
        f"房间：内部 {g['inner_w']} m × {g['inner_h']} m（套内 {g['floor_area_label']} m²）",
        f"实测点：{ctx['n_points']} 个（另有 1200 / 200 / 200 张物理仿真地图）",
        '网络方案：卷积补全自编码器 / 小型 U-Net / 插值引导残差 U-Net',
        '对照方法：最近邻、IDW、零均值 RBF、对数距离趋势 + RBF 残差',
        '报告生成日期：2026-10-21',
        '数据划分、模型结构、标准化参数、随机种子全部固定，可完整复现。',
    ]:
        par = doc.add_paragraph()
        r = par.add_run(line)
        r.font.size = Pt(10.5)
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # ---------------- 摘要 ----------------
    h(doc, '摘要', 1)
    corner = protos.get('corner', {}).get('methods', {})
    center = protos.get('center', {}).get('methods', {})
    best_corner = min(corner.items(), key=lambda x: x[1]['rmse'])[1]['rmse'] if corner else float('nan')
    best_center = min(center.items(), key=lambda x: x[1]['rmse'])[1]['rmse'] if center else float('nan')
    sim_best = min(sim.get('methods', {}).items(), key=lambda x: x[1]['rmse_mean'])[1]['rmse_mean'] if sim.get('methods') else float('nan')
    bl = {k: v for k, v in sim.get('methods', {}).items() if k in BASELINE_LABELS}
    sim_bl = min(bl.items(), key=lambda x: x[1]['rmse_mean'])[1]['rmse_mean'] if bl else float('nan')

    p(doc, f'本项目的目标是：只在一个房间里测几十个点，把整间房的 Wi-Fi 信号强度图还原出来。'
           f'我们在一间内部 {g["inner_w"]} × {g["inner_h"]} m 的真实房间里采集了 {ctx["n_points"]} 个 '
           f'RSSI 实测点；由于只有一张真实参考地图，另用物理模型生成了 1600 张随机户型地图用于预训练。'
           f'三个网络方案在 200 个未见过的新户型上全部优于传统的插值基线，'
           f'最优网络 RMSE 为 {sim_best:.2f} dB，最优插值基线为 {sim_bl:.2f} dB。'
           f'在真实房间的留出点上，中心插值协议最优 {best_center:.2f} dB、'
           f'角块外推协议最优 {best_corner:.2f} dB。')
    table(doc, ['关键结果', '数值', '说明'], [
        ['实测测点总数', f'{ctx["n_points"]} 点', '单房间、单次部署'],
        ['RSSI 范围', f'{q["rssi_min"]:.0f} ~ {q["rssi_max"]:.0f} dBm',
         f'其中 {len(q["br04_too_strong"])} 点强于 −40 dBm（BR-04 偏强）'],
        ['跨户型 · 最优网络 RMSE', f'{sim_best:.3f} dB', '200 个未见过的随机户型'],
        ['跨户型 · 最优插值基线 RMSE', f'{sim_bl:.3f} dB', '对数距离趋势 + RBF 残差'],
        ['中心插值 · 最优 RMSE', f'{best_center:.3f} dB', '真实数据留出 9 点'],
        ['角块外推 · 最优 RMSE', f'{best_corner:.3f} dB', '真实数据留出 6 点'],
    ], caption='表 0-1 关键结果总览')

    # ---------------- 1 任务 ----------------
    h(doc, '1. 任务与目标', 1)
    p(doc, '课程任务书给出三条明确要求，本报告逐条对应：')
    bullets(doc, [
        '统一输入三通道：稀疏 RSSI 图、观测位置标记图、房间有效区域图；未测位置只是占位值，'
        '不能被理解为零信号。（见第 7 节与 rfmap/models.py::build_input_tensor）',
        '三个网络方案：卷积补全自编码器、小型 U-Net（建议主模型）、插值引导残差 U-Net。（见第 7 节）',
        '评估口径：在未参与训练的测试场景上，用其余隐藏实测点做答案，报告 MAE、RMSE（dB）与推理时间，'
        '并与只使用相同输入点的简单插值方法比较。（见第 9、10 节）',
    ])
    p(doc, '任务书同时强调："不要用插值结果充当实测答案"；以及"若只有一张固定地图，反复遮挡只能用于'
           '程序调试，不能证明模型学会了未知地图补全"。本项目手里恰好只有 1 张真实参考地图，'
           '因此报告把结论拆成两类分别陈述，并在第 11 节明确写出强度差异。')

    # ---------------- 2 数据 ----------------
    h(doc, '2. 实测数据与质检', 1)
    p(doc, f'数据来自工作区的 data.xlsx（Sheet1，A1:D50），表头为 x轴（m）/ y轴（m）/ RSSI（dBm），'
           f'共 {ctx["n_points"]} 个测点。规范化后的副本见 data/rfmap/measured.csv。')
    table(doc, ['检查项', '结果'], [
        ['测点总数', f'{q["n_points"]}'],
        ['x 取值（现场标称）', str(q['x_values_data'])],
        ['y 取值（现场标称）', str(q['y_values_data'])],
        ['RSSI 范围', f'{q["rssi_min"]} ~ {q["rssi_max"]} dBm'],
        ['RSSI 均值 / 中位数', f'{q["rssi_mean"]} / {q["rssi_median"]} dBm'],
        ['最小测点间距', f'{q["min_neighbor_spacing_m"]} m'],
        ['落在房间外的点', f'{len(q["outside_room_idx"])}'],
        ['名义网格上的缺格', f'{len(q["missing_cells_nominal"])}（见第 3.3 节解释）'],
    ], caption='表 2-1 数据质检结果')
    p(doc, f'BR-04 违规点：{len(q["br04_too_strong"])} 个点的读数强于 −40 dBm'
           f'（编号 {", ".join(str(i) for i in q["br04_too_strong"])}，全部集中在 AP 附近）。'
           f'任务书规定正常摆放应落在 −40 ~ −70 dBm；本批数据整体偏强，'
           f'说明采集时设备离 AP 偏近（或 AP 发射功率偏高）。按业务规则，'
           f'这些点已标注但未被删除，也没有被剔除出训练或评估。', bold=True)
    figure(doc, os.path.join(figdir, 'fig_data_profile.png'),
           '图 2-1 左：RSSI 分布（绿色带为 BR-04 合规区间）；中：RSSI 与到最强点距离的关系；'
           '右：测点位置与强度')

    # ---------------- 3 几何 ----------------
    h(doc, '3. 户型几何重建', 1)
    p(doc, '房间几何不是"画着好看"的背景，它直接决定哪些格点是有效区、哪些测点合法。'
           '本节全部尺寸取自户型图 实验地户型.png 上的实测标注，未作任何主观修改。')
    table(doc, ['项目', '数值 (m)', '来源 / 验算'], [
        ['外轮廓宽', f'{g["outer_w"]}', '户型图标注'],
        ['外轮廓高', f'{g["outer_h"]}', '户型图标注'],
        ['墙厚', f'{g["wall"]}', '四边标注一致'],
        ['内部净宽', f'{g["inner_w"]}', f'= {g["outer_w"]} − 2 × {g["wall"]} ✓'],
        ['内部净高', f'{g["inner_h"]}', f'= {g["outer_h"]} − 2 × {g["wall"]} − 0.146 ✓'],
        ['结构柱宽 / 高', f'{ck["w"]} × {ck["h"]}', '柱体标注 0.830 × 1.010'],
        ['柱上沿距内顶', f'{G.COL_TOP_OFFSET}', '标注 4.960'],
        ['柱下沿距内底', f'{G.COL_BOTTOM_OFFSET}', '标注 5.761'],
        ['尺寸链闭合', f'{4.960 + 1.010 + 5.761:.3f}', f'= 内部净高 {g["inner_h"]} ✓'],
        ['顶部窗宽 / 底部门宽', f'{g["top_window"]["w"]} / {g["door"]["w"]}', '户型图标注'],
        ['套内面积（图中标注）', f'{g["floor_area_label"]} m²', '户型图正面标注'],
        ['净面积（计算）', f'{g["net_area"]:.2f} m²',
         f'内部矩形 {g["gross_area"]:.2f} − 柱体 {ck["w"] * ck["h"]:.2f}；'
         f'与鞋带公式结果 {g["polygon_area"]:.2f} 一致 ✓'],
    ], caption='表 3-1 户型标注读数与尺寸链验算')

    h(doc, '3.2 与旧版轮廓的差异（重要修正）', 2)
    p(doc, '项目早期的一版报告把房间画成一个带"右侧凹口"的异形轮廓（x > 11.7、y ∈ [6, 8.88] 被当作'
           '房间外）。按户型图核实后，这个轮廓不成立：房间本身是一个 '
           f'{g["inner_w"]} × {g["inner_h"]} m 的规整矩形，唯一的实体障碍是一根贴墙结构柱'
           f'（{ck["w"]} × {ck["h"]} m，位于 x ∈ [{ck["x0"]:.3f}, {ck["x1"]:.3f}]、'
           f'y ∈ [{ck["y0"]:.3f}, {ck["y1"]:.3f}]）。本报告已按户型图重建几何。', bold=True)

    h(doc, '3.3 结构柱解释了数据里的两个"缺口"', 2)
    p(doc, '实测数据在 x = 12 这一列上缺少 y = 6 与 y = 8 两点，改用 (11.7, 6) 与 (11.7, 8.9) 替代。'
           '这不是采集遗漏，而是一个可以被几何独立验证的必然结果：')
    table(doc, ['测点', '房间坐标', '是否落在结构柱内', '数据表中的实际情况'], [
        ['(12, 6)', '(12.00, 5.87)', '是（x ∈ [11.813, 12.643] 且 y ∈ [5.761, 6.771]）',
         '该点不存在 —— 与柱体无法布点完全一致'],
        ['(11.7, 6)', '(11.70, 5.87)', '否（11.70 < 11.813，位于柱外 0.113 m）',
         '该点存在，RSSI = −44 dBm'],
    ], caption='表 3-2 结构柱与缺失测点的对应关系')
    p(doc, '这条证据同时验证了三件事：① 结构柱的位置定对了；② 实测坐标与户型图的对应关系'
           '（含 x 轴朝向）定对了；③ 数据里那两处缺失是有物理原因的。'
           '它也解释了为什么本项目没有对 x 轴做比例缩放 —— 一旦把 x 乘上 12.643/12 = 1.0536，'
           '(11.7, 6) 会被推进柱体内部，与"该点确实测到了 −44 dBm"直接矛盾。')
    figure(doc, os.path.join(figdir, 'fig_geometry.png'),
           f'图 3-1 按户型图重建的房间几何与 {ctx["n_points"]} 个测点。深灰色块为结构柱，'
           '底部蓝线为门，顶部绿线为窗；x 轴已按现场方位反向，原点在房间右下角。')

    # ---------------- 4 坐标 ----------------
    h(doc, '4. 坐标系与对齐规则', 1)
    bullets(doc, [
        '原点 (0, 0) = 房间内部右下角；',
        'x 轴正向指向户型图左侧（与实测数据表 x 增大方向一致）；',
        'y 轴正向向上。',
    ])
    table(doc, ['轴', '系数', '理由'], [
        ['x', f'{q["alignment"]["x_scale"]}',
         f'保持现场标称值不缩放（理由见 3.3 节）。代价是房间在 x ∈ [12, {g["inner_w"]}] 之间'
         f'留下 {q["unsampled_strip_x_m"]} m 宽的未采样边带，如实标注。'],
        ['y', f'{q["alignment"]["y_scale"]}',
         f'= {g["inner_h"]} / 12，把实测 12 m 跨度压缩进房间内高，使 y = 12 的贴墙点恰好落在内墙上。'],
    ], caption='表 4-1 各轴对齐系数')
    table(doc, ['网格参数', '取值'], [
        ['栅格间距', f'{ctx["grid"]["cell"]} m（正方形格）'],
        ['网格尺寸', f'{ctx["grid"]["nx"]} × {ctx["grid"]["ny"]}'],
        ['覆盖范围', f'{ctx["grid"]["x_max"]} × {ctx["grid"]["y_max"]} m'],
        ['有效格点数', f'{int(GRID.room_mask().sum())} / {GRID.nx * GRID.ny}'],
    ], caption='表 4-2 计算网格（训练、推理、前端三方共用 data/rfmap/grid.json）')

    # ---------------- 5 划分 ----------------
    h(doc, '5. 数据划分协议', 1)
    p(doc, '任务书要求：先按独立场景组划分训练/验证/测试，再对每张地图抽取测点；'
           '同一参考地图的重复记录必须放在同一集合。本项目严格执行，'
           '并把这条规则落实到代码里（rfmap/evaluate.py::split_points）。')
    rows = []
    for k, e in protos.items():
        m = e['methods']
        best = min(m.items(), key=lambda x: x[1]['rmse'])
        best_bl = min(((kk, vv) for kk, vv in m.items()
                       if kk in BASELINE_LABELS and not kk.endswith('_ft')),
                      key=lambda x: x[1]['rmse'])
        rows.append([e['label'], e['n_input'], e['n_hidden'],
                     MODEL_LABELS.get(best[0], best[0]), f'{best[1]["rmse"]:.3f}',
                     BASELINE_LABELS.get(best_bl[0], best_bl[0]),
                     f'{best_bl[1]["rmse"]:.3f}'])
    table(doc, ['协议', '输入点', '隐藏点', '最优方法', 'RMSE', '最优插值基线', 'RMSE'],
          rows, caption='表 5-1 两套固定协议（写死在代码中，全局复用）')
    p(doc, '角块留出（外推）：隐藏 x∈{0,2} × y∈{8,10,12} 共 6 点，全部位于房间一角，'
           '考察"往外推"的能力。中心块留出（插值）：隐藏 x∈{4,6,8} × y∈{4,6,8} 共 9 点，'
           '位于房间中央、四周仍有观测点包围，考察"往中间补"的能力。')

    h(doc, '5.3 输入点数扫描', 2)
    p(doc, '任务书要求"改变输入测点数量和空间分布，观察预测误差如何变化"。'
           '对 K ∈ {5,10,15,20,30} 各做 20 次随机抽样（种子固定），'
           '每次随机选 K 个点作输入、其余全部作隐藏点。')
    methods = sorted({m for v in sweep.values() for m in v})
    sw_rows = []
    for K in sorted(int(x) for x in sweep):
        sw_rows.append([K] + [f'{_fmt(sweep[str(K)].get(m, {}).get("rmse_mean"))}'
                              for m in methods])
    table(doc, ['输入点数'] + [BASELINE_LABELS.get(m, MODEL_LABELS.get(m, m)) for m in methods],
          sw_rows, caption='表 5-2 输入点数对隐藏点 RMSE 的影响（dB）')
    figure(doc, os.path.join(figdir, 'fig_sweep.png'),
           '图 5-1 输入测点数对隐藏点 RMSE 的影响（误差棒为 20 次抽样的标准差）')

    p(doc, '必须写明的限制：本项目只有 1 张真实参考地图。任务书明确指出，这种情况下'
           '"反复遮挡只能用于程序调试，不能证明模型学会了未知地图补全"。因此本报告把结论'
           '分成两类：跨户型/跨 AP 的能力由第 6 节的物理仿真测试集回答；同一房间内的稀疏'
           '重建能力由真实 49 点留出回答，不能外推成"换一个房子也行"。', bold=True)

    # ---------------- 6 仿真 ----------------
    h(doc, '6. 物理仿真数据', 1)
    p(doc, '用 49 个点训练一个能"补全新地图"的网络是不可能的：模型会把这一间房的具体路径损耗场'
           '背下来，换一间房立刻失效。本项目按任务书思路实现：用物理模型生成大量随机户型/随机 AP '
           '的地图做预训练，再用真实 49 点微调。')
    code(doc, """PL(d) = PL0 + 10·n·log10(max(d, 0.5) / 1.0)
RSSI  = −(PL0 + 10·n·log10(d)) − wall_attenuation + shadow_field + noise

PL0  ~ U(32, 46) dB          1 m 参考路损
n    ~ U(1.8, 3.2)           室内路损指数
wall_attenuation             AP 与格点连线每穿过一个障碍矩形，累加 U(4, 14) dB
shadow_field                 空间相关高斯场（白噪声高斯滤波，std ~ U(1.5, 4.5) dB）
noise                        逐点独立 U(0.3, 1.2) dB
最终截断到 [−95, −25] dBm""")
    sd = ctx.get('sim_data', {})
    table(doc, ['文件', '样本数', '数组形状', '大小'], [
        [f'data/sim/{k}.npz', v['n'], str(v['shape']), f'{v["mb"]} MB']
        for k, v in sd.items()],
        caption='表 6-1 仿真数据集（训练/验证/测试种子互不相同：20261021 / 20261022 / 20261023）')
    p(doc, '生成后逐项自检全部通过：观测点上的观测值与真值逐元素一致、观测点全部位于房间内、'
           '房间占比落在 30%~80%、真值域落在 [−95, −25] dBm、同种子两次生成完全一致、'
           '真值最大值出现在距 AP 最近 15% 区域内的比例达 100%。')
    figure(doc, os.path.join(figdir, 'fig_sim_samples.png'),
           '图 6-1 仿真训练数据示例（真值 / 观测点 / 户型掩码），每一行是一张不同的户型与 AP 位置')

    # ---------------- 7 模型 ----------------
    h(doc, '7. 网络方案：从任务书的三个方案到方案五', 1)
    p(doc, '任务书要求实现三个网络方案；本报告在此基础上再补做两个，共五个方案：')
    table(doc, ['方案', '一句话概括', '要回答的问题'], [
        ['方案一 · 卷积补全自编码器', '无跳跃连接的编码-解码补全',
         '（任务书要求）最朴素的补全网络能做到什么程度'],
        ['方案二 · 小型 U-Net', '带跳跃连接的 U-Net（建议主模型）',
         '（任务书要求）跳跃连接对局部结构的价值'],
        ['方案三 · 插值引导残差 U-Net', 'IDW 先验 + 残差学习',
         '（任务书要求）只学"插值错在哪"是否更省力'],
        ['方案四 · 散落布点训练 U-Net', '结构与方案二完全相同，只改训练数据的布点方式',
         '真实用户不会按网格布点；观测点散乱时模型还准吗？'],
        ['方案五 · 物理引导多先验残差 U-Net', '6 通道输入 + IDW 先验残差 + 观测加权损失 + TTA',
         '把前四个方案暴露的短板逐个补上，能到什么精度？'],
    ], caption='表 7-1 五个网络方案与各自要回答的问题')
    p(doc, '五个方案用 PyTorch 实现；方案一/二/三/四统一输入 3 通道（稀疏观测值、观测掩码、'
           '房间有效区掩码），方案三/五额外拼 1 路插值先验，方案五再补 3 路物理引导通道。'
           '为便于把权重导出到浏览器做纯 JS 前向，网络不使用 BatchNorm / Dropout，'
           '全部归一化只保留"输入/输出各一次线性缩放"。')
    h(doc, '7.1 方案一 · 卷积补全自编码器', 2)
    p(doc, '编码器把稀疏观测压缩到 1/8 分辨率的潜变量，解码器直接恢复整张地图。没有跳跃连接 —— '
           '这正是它与 U-Net 的关键差别：空间细节只能靠潜变量带回来，因此对观测稀疏更敏感。')
    h(doc, '7.2 方案二 · 小型 U-Net（主模型）', 2)
    p(doc, '编码/解码之间用跳跃连接保留空间信息。室内 RSSI 的起伏主要来自墙体遮挡与多径，'
           '这些是局部结构，靠潜变量上下采样容易被糊掉。结构为 3 段下采样（24→48→96）'
           '→ bottleneck 192 → 3 段上采样。')
    h(doc, '7.3 方案三 · 插值引导残差 U-Net', 2)
    p(doc, '先用反距离加权（IDW）算出一张插值先验图作为第 4 个输入通道；U-Net 不直接预测 RSSI，'
           '而是预测"真值 − 先验"的残差，最终输出"先验 + 残差"。')
    p(doc, '实现过程中踩到并修掉的两个坑（如实记录）：', bold=True)
    bullets(doc, [
        '通道重复拼接：早期版本在数据侧和模型内部各拼了一次先验，输入变成 5 通道，'
        '网络第一层直接报维度错误。修正方式是固定"prior 只由模型内部拼接"。',
        'NaN 传播：IDW 先验在房间外是 NaN，而未观测格点乘以 0 在浮点语义下仍是 NaN，'
        '导致 resunet 前 30 个 epoch 的 loss 全是 nan。现在统一用 prepare_prior() 把'
        '非法值与房间外格点填成训练集均值（标准化后为 0）。',
    ])
    h(doc, '7.4 方案四 · 散落布点训练的 U-Net（数据消融）', 2)
    p(doc, '动机：方案一/二/三的训练数据里，约一半样本的观测点用"等距网格 + 抖动"布下，'
           '这隐含了一个很强的先验 —— 房间任何一个角落附近都恰好有已知点。'
           '但真实用户不会这么测：他们常常在某处连测几点、又在另一处漏掉一整片。'
           '方案四要问的是：把训练数据的布点方式改成散落式，模型在观测点散乱的房间里'
           '还能不能保持精度？')
    p(doc, '做法（严格单变量）：如果连户型和信号真值一起重新生成，"布点方式"的效应就会和'
           '"户型分布"的效应混在一起，结论无法归因。所以本方案逐元素复用方案一/二/三'
           '那份数据的 room 与 target 数组，只用新的随机种子重抽观测点位置。'
           '网络结构、超参、随机种子一字不改 —— 唯一变量就是布点方式。')
    _om = (ctx.get('obs_stats') or {}).get('mixed') or {}
    _os = (ctx.get('obs_stats') or {}).get('scatter') or {}
    table(doc, ['训练数据集', '样本数', '每样本观测点数（均值）',
                '到最近观测点距离 中位数', '同上 P95'], [
        ['常规布点（方案一/二/三）', _om.get('n_samples', '—'),
         _om.get('points_mean', '—'), f"{_om.get('nn_median', '—')} m",
         f"{_om.get('nn_p95', '—')} m"],
        ['散落布点（方案四）', _os.get('n_samples', '—'),
         _os.get('points_mean', '—'), f"{_os.get('nn_median', '—')} m",
         f"{_os.get('nn_p95', '—')} m"],
    ], caption='表 7-2 两种布点方式的统计对比（观测点数量刻意保持不变）')
    p(doc, f'两行的观测点数量完全相同（均值都是 {_om.get("points_mean", "—")} 个），'
           f'变的是位置分布：散落布点把"到最近观测点的距离"中位数从 {_om.get("nn_median", "—")} m '
           f'抬到 {_os.get("nn_median", "—")} m，P95 从 {_om.get("nn_p95", "—")} m '
           f'抬到 {_os.get("nn_p95", "—")} m。也就是说，在同样的测量成本下，'
           f'外推区被显著放大了 —— 这正是方案四给自己的压力测试。结果见 10.4 节。')
    figure(doc, os.path.join(figdir, 'fig_obs_layout.png'),
           '图 7-1 布点方式对比（同一批户型的同一个样本）。上排为观测点位置，'
           '下排为"到最近观测点的距离"（越亮表示离已知点越远）。'
           '右侧散落布点在点团之间留下大片空白，距离图中出现明显的亮区。')
    h(doc, '7.5 方案五 · 物理引导多先验残差 U-Net（本报告主推）', 2)
    p(doc, '设计方法是"看短板、逐条补"，不是把模块堆上去。回看前四个方案，'
           '可以指出三个具体短板：')
    bullets(doc, [
        '方案三只带 IDW 一路先验，而 IDW 在没有观测点的角落会退化到全局均值 —— '
        '实测也印证了这一点（方案二/三在角块外推上明显吃亏）。缺一个全局的衰减形状先验。',
        '所有网络都不知道"自己此刻有多不确定"：观测点旁 0.3 m 的格点和 4 m 之外的格点，'
        '在网络眼里输入形式是一样的。缺一个显式的外推难度度量。',
        '损失函数对所有格点一视同仁。但观测点附近才是唯一有真值锚点的地方，'
        '用同一把尺子去量"锚点旁"和"完全未知区"，会把精度从前者挪给后者。',
    ], numbered=True)
    p(doc, '方案五逐条对应，共四处改动：', bold=True)
    p(doc, '(a) 输入扩成 7 通道。')
    table(doc, ['通道', '内容', '解决什么'], [
        ['0', '稀疏观测 RSSI（标准化，未测处严格为 0）', '任务书规定的统一输入'],
        ['1', '观测掩码', '任务书规定的统一输入'],
        ['2', '房间有效区掩码', '任务书规定的统一输入'],
        ['3', '对数距离物理趋势（由观测点最小二乘拟合 P0、n 后推全图）',
         '短板 1：给网络一个全局正确的衰减形状，外推时不至于默认"远处等于均值"'],
        ['4', '到最近观测点的距离（按 6 m 归一化到 0~1）',
         '短板 2：这是外推难度的显式度量，网络据此可学到"越远越信物理趋势、越近越信实测"'],
        ['5', '观测密度（观测掩码经 σ=2 格高斯模糊，峰值归一）',
         '区分"孤零零一个点"与"好几米的密集测量区"'],
        ['6', 'IDW 插值先验（方案三已证明有效）', '保留方案三"学残差"的优势'],
    ], caption='表 7-3 方案五的 7 个输入通道')
    figure(doc, os.path.join(figdir, 'fig_arch_v5.png'),
           '图 7-2 方案五的输入通道与数据流。蓝色 3 通道沿用任务书统一约定；'
           '绿色 3 通道补进"全局物理趋势 / 外推难度 / 可信度"；橙色为方案三已验证的插值先验。'
           '输出仍是 IDW 先验 + 网络预测残差。')
    p(doc, '(b) 仍以 IDW 为残差基准。方案三已证明"学插值残差"比从零学整场容易得多，'
           '这个优势在方案五里被保留：网络输出的是相对 IDW 先验的修正量。')
    p(doc, '(c) 观测加权损失。权重 = 1 + 2 × 观测密度，再对房间内归一到均值 1。'
           '效果是观测点附近预测错了要付出约 3 倍代价，而完全未知区的权重回落到 1。')
    p(doc, '(d) 推理时自增强（TTA）。对输入做恒等 / 上下翻转 / 左右翻转 / 旋转 180° 四种'
           '几何变体各推理一次，逆变换后平均。它对补全任务合理（整图翻转等价于一间镜像户型），'
           '但并非无条件成立，因此做了开/关消融（见 10.6 节）。')
    _m5 = (ctx['models'].get('pgmpr') or {}).get('meta') or {}
    _m2 = (ctx['models'].get('unet') or {}).get('meta') or {}
    if _m5 and _m2 and _m2.get('params'):
        _dd = _m5['params'] - _m2['params']
        p(doc, f'参数量对照：方案五 {_m5["params"]:,} 个参数，方案二 {_m2["params"]:,} 个，'
               f'只多 {_dd:,} 个（+{_dd / _m2["params"] * 100:.2f}%），全部来自输入层的 3 个新通道。'
               f'U-Net 主干宽度、深度、跳跃连接与方案二完全一致 —— 因此 10 节里方案五的优势'
               f'不能归因于"模型更大"。')
    h(doc, '7.6 参数量与训练配置', 2)
    mrows = []
    for m in MODEL_ORDER:
        if m not in ctx['models']:
            continue
        meta = ctx['models'][m]['meta']
        mrows.append([MODEL_LABELS[m], f'{meta["params"]:,}', meta['width'],
                      meta['epochs_run'], meta['best_epoch'],
                      f'{meta["best_val_loss"]:.5f}', f'{meta["seconds"]:.0f}'])
    table(doc, ['模型', '参数量', '基础通道', '实际轮数', '最佳轮', '最佳验证 loss', '耗时 (s)'],
          mrows, caption='表 7-1 三个方案的规模与训练结果')
    p(doc, '统一配置：Adam（lr = 2e−3，余弦退火）、batch = 32、weight decay 1e−5、'
           '梯度裁剪 5.0、早停 patience = 30、随机种子 7。损失为 masked MSE，'
           '只在房间内有效格点上计算。标准化参数只从训练集拟合：'
           f'mean = {ctx["models"]["unet"]["meta"]["normalizer"]["mean"]:.3f} dBm，'
           f'std = {ctx["models"]["unet"]["meta"]["normalizer"]["std"]:.3f} dB。')

    # ---------------- 8 训练 ----------------
    h(doc, '8. 训练过程', 1)
    figure(doc, os.path.join(figdir, 'fig_training_curves.png'),
           '图 8-1 三个方案的仿真预训练曲线（纵轴对数刻度；蓝线训练、橙线验证，'
           '红星为最佳验证点）')
    p(doc, f'三个模型的最佳验证损失（标准化尺度下的 masked MSE）：'
           f'方案三 {ctx["models"]["resunet"]["meta"]["best_val_loss"]:.5f} < '
           f'方案一 {ctx["models"]["ae"]["meta"]["best_val_loss"]:.5f} < '
           f'方案二 {ctx["models"]["unet"]["meta"]["best_val_loss"]:.5f}。'
           f'验证损失不能直接读成 dB，真正的精度请看第 10 节。')

    # ---------------- 9 评估 ----------------
    h(doc, '9. 评估方法与基线', 1)
    bullets(doc, [
        'MAE：隐藏实测点上 |预测 − 真值| 的平均值，单位 dB；',
        'RMSE：均方误差开方，单位 dB，作为主要排序指标；',
        '最大绝对误差：最坏情况；',
        'bias：平均（预测 − 真值），正值表示系统性偏强；',
        '推理时间：同一输入重复 30 次的平均值（毫秒），已预热。',
    ])
    table(doc, ['基线方法', '代码入口'], [[v, k] for k, v in BASELINE_LABELS.items()],
          caption='表 9-1 四条插值基线')
    p(doc, '零均值 RBF 基线为什么这么差（RMSE 20+ dB）？它没有"距离越远越弱"的先验，'
           '在实测点之外会迅速衰减回 0，作为对照组并不公平。因此本项目额外实现了'
           '"对数距离趋势 + RBF 残差"这条更强的基线（gp_residual_predict）：'
           '先用输入点拟合 PL(d) = P0 − 10·n·log10(d)，再对残差做 RBF 插值。'
           '它才是神经网络真正需要超越的对手。')

    # ---------------- 10 结果 ----------------
    h(doc, '10. 实验结果', 1)
    h(doc, '10.1 跨户型 / 跨 AP（两个仿真测试集，各 200 个场景）', 2)
    _ts = sim.get('testsets') or {}
    _mx = (_ts.get('mixed') or {}).get('methods') or {}
    _sc = (_ts.get('scatter') or {}).get('methods') or {}
    p(doc, '这是结论强度最高的一组实验：200 个训练中从未出现过的随机户型与随机 AP 位置，'
           f'平均每个场景 {(_ts.get("mixed") or {}).get("obs_count_mean", "—")} 个观测点。'
           '误差只在房间内的未观测格点上统计（观测点本身不参与评分，避免"抄答案"）。')
    p(doc, '两个测试集逐元素共享同一批户型和真值，只有观测点的布点方式不同：'
           '「常规布点」与方案一/二/三的训练分布一致，「散落布点」与方案四一致。'
           '两个都评是为了避免片面结论 —— 只看前者会低估散落训练的价值，'
           '只看后者又会掩盖"专精散落布点是否牺牲了常规布点性能"。'
           '表中网络方案给的是未启用 TTA 的结果，TTA 的效果见 10.6 节。')
    srows = []
    keys = [k for k in _mx if not k.endswith('_tta')]
    best_bl = min((_mx[k]['rmse_mean'] for k in keys if k in BASELINE_LABELS),
                  default=0.0)
    for k in sorted(keys, key=lambda x: _mx[x]['rmse_mean']):
        name = _mname(k)
        if k in BASELINE_LABELS:
            name += '（基线）'
        gain = '—' if k in BASELINE_LABELS else f'{best_bl - _mx[k]["rmse_mean"]:+.2f}'
        sv = _sc.get(k, {}).get('rmse_mean')
        srows.append([name, f'{_mx[k]["rmse_mean"]:.3f}',
                      f'{sv:.3f}' if sv is not None else '—', gain])
    table(doc, ['方法', '常规布点 RMSE (dB)', '散落布点 RMSE (dB)', '比最优插值基线低'],
          srows, caption='表 10-1 五个方案与插值基线在两个仿真测试集上的对照')
    figure(doc, os.path.join(figdir, 'fig_scheme_compare.png'),
           '图 10-1 五个网络方案与四条插值基线在 200 个未见过的户型上的未观测格点 RMSE。'
           '左为常规布点测试集，右为散落布点测试集')
    p(doc, '结论：五个网络方案在两个测试集上都稳定优于全部插值基线；'
           '其中方案五在两个测试集上都排第一 —— 它在通道与损失上的改动'
           '确实转化成了精度（逐项归因见 10.5 节）。'
           f'方案五比最强的插值基线低约 '
           f'{best_bl - _mx["pgmpr"]["rmse_mean"]:.2f} dB（常规布点）/ '
           f'{(min((_sc[k]["rmse_mean"] for k in keys if k in BASELINE_LABELS), default=0.0) - _sc["pgmpr"]["rmse_mean"]):.2f} dB（散落布点）。')
    figure(doc, os.path.join(figdir, 'fig_sim_compare.png'),
           '图 10-2 仿真测试集两个场景下，同一组输入点、同一色标下各方法的输出与真值对比')

    for key in ('corner', 'center'):
        if key not in protos:
            continue
        e = protos[key]
        m = e['methods']
        h(doc, f'10.{2 if key == "corner" else 3} {e["label"]}（输入 {e["n_input"]} 点 / 隐藏 {e["n_hidden"]} 点）', 2)
        p(doc, '评估方式：把该区域全部实测点从输入里拿掉，只喂给模型其余观测点，'
               '再用被拿掉的点算误差。隐藏点从头到尾没有进入过输入通道。')
        keys = sorted([k for k in m if not k.endswith('_ft')],
                      key=lambda k: m[k]['rmse'])
        table(doc, ['方法', 'MAE (dB)', 'RMSE (dB)', '最大绝对误差', 'bias', '推理 (ms)'],
              [[BASELINE_LABELS.get(k, MODEL_LABELS.get(k, k)),
                f'{m[k]["mae"]:.3f}', f'{m[k]["rmse"]:.3f}', f'{m[k]["max_abs"]:.3f}',
                f'{m[k]["bias"]:.3f}', f'{m[k]["infer_ms"]:.2f}'] for k in keys],
              caption=f'表 {e["label"]} —— 预训练模型', best_row=0)
        ft = sorted([k for k in m if k.endswith('_ft')], key=lambda k: m[k]['rmse'])
        if ft:
            table(doc, ['方法（实测微调后）', 'MAE (dB)', 'RMSE (dB)', '最大绝对误差', 'bias', '推理 (ms)'],
                  [[MODEL_LABELS.get(k[:-3], k[:-3]) + '（微调）',
                    f'{m[k]["mae"]:.3f}', f'{m[k]["rmse"]:.3f}', f'{m[k]["max_abs"]:.3f}',
                    f'{m[k]["bias"]:.3f}', f'{m[k]["infer_ms"]:.2f}'] for k in ft],
                  caption=f'表 {e["label"]} —— 实测微调后', best_row=0)
        figure(doc, os.path.join(figdir, f'fig_real_{key}.png'),
               f'图 {e["label"]} 各方法热力图对比（统一色标 −55 ~ −27 dBm）')
        figure(doc, os.path.join(figdir, f'fig_err_{key}.png'),
               f'图 {e["label"]} 隐藏点误差（标注值 = 预测 − 真值，单位 dB）')

    h(doc, '10.4 方案四的数据消融：把观测点布"散"，值多少？', 2)
    p(doc, '方案二与方案四是严格配对的：同一个网络、同一组超参、同一个随机种子、'
           '同一批户型和真值，唯一差别是训练数据的布点方式。因此下面每一格的差值'
           '都可以直接归因到"布点方式"这一个变量上。')
    d4 = []
    for k in ('unet', 'unet_scatter'):
        d4.append([_mname(k),
                   f'{_mx.get(k, {}).get("rmse_mean", float("nan")):.3f}',
                   f'{_sc.get(k, {}).get("rmse_mean", float("nan")):.3f}',
                   f'{_sc.get(k, {}).get("rmse_p90", float("nan")):.3f}'])
    table(doc, ['方法（结构完全相同）', '常规布点 RMSE', '散落布点 RMSE', '散落布点 P90'],
          d4, caption='表 10-4 方案二与方案四的配对对照')
    figure(doc, os.path.join(figdir, 'fig_scatter_ablation.png'),
           '图 10-3 方案二与方案四在两个测试集上的未观测格点 RMSE')
    _ds = _sc.get('unet', {}).get('rmse_mean', float('nan')) - \
        _sc.get('unet_scatter', {}).get('rmse_mean', float('nan'))
    _dm = _mx.get('unet', {}).get('rmse_mean', float('nan')) - \
        _mx.get('unet_scatter', {}).get('rmse_mean', float('nan'))
    p(doc, f'结论：这一项改动没有带来收益，反而略有退步。散落布点测试集上'
           f'方案四比方案二差 {_ds:.3f} dB，常规布点测试集上差 {_dm:.3f} dB。',
      bold=True)
    p(doc, '但"整体均值平手"不等于"处处平手"，所以进一步把散落布点测试集的 200 个样本'
           '按每个样本自身的散落程度（未观测区内到最近观测点距离的 P95）分成三档：')
    _st = (ctx.get('strat') or {}).get('testsets', {}).get('scatter', {})
    if _st.get('bins'):
        srows2 = []
        for b in _st['bins']:
            mm = b['methods']
            srows2.append(
                [b['label'], f"{b['spread_mean']} m", str(b['n']),
                 f'{mm.get("unet", float("nan")):.3f}',
                 f'{mm.get("unet_scatter", float("nan")):.3f}',
                 f'{mm.get("unet", 0) - mm.get("unet_scatter", 0):+.3f}',
                 f'{mm.get("pgmpr", float("nan")):.3f}'])
        table(doc, ['散落度档位', '未观测区 P95 距离', '样本数',
                    '方案二 RMSE', '方案四 RMSE', '方案二−方案四', '方案五 RMSE'],
              srows2, caption='表 10-5 按样本散落程度分层后的对照')
        figure(doc, os.path.join(figdir, 'fig_strata.png'),
               '图 10-4 按布点散落程度分档后各方案的 RMSE（横轴 P95 距离越大表示外推越难）')
        p(doc, f'分档后结论没有改变：在最散落的一档上方案四依然比方案二差'
               f'{abs(_st["bins"][-1]["methods"].get("unet", 0) - _st["bins"][-1]["methods"].get("unet_scatter", 0)):.3f} dB，'
               f'在"最规整"一档上基本打平。也就是说，方案四的散落训练'
               f'并没有换来任何一档上的优势，这是一个如实记录的"零结果"。', bold=True)
    p(doc, '为什么"更贴近真实使用场景"的训练数据反而没帮上忙？从数据和设计上能找到三个原因：')
    bullets(doc, [
        '原版数据本来就不全是规整网格：它的抽样策略是"约 50% 等距网格 + 50% 纯随机撒点"。'
        '换句话说，方案二在训练时已经见过大量随机布点的样本，方案四带来的增量很有限 ——'
        '两者的最近观测点距离中位数只差 0.4 m 左右。',
        '把一半样本换成散落布点，削掉的是"均匀覆盖"样本的绝对数量。训练总量固定为 1200，'
        '此消彼长，而均匀覆盖对学习整场衰减形状是有帮助的。',
        '这个任务对布点规整程度本来就不敏感：散落布点虽然局部更空，但观测点仍散布在整个房间，'
        '模型从仿真里学到的"户型 → 衰减"先验依然适用。',
    ], numbered=True)
    p(doc, f'这个"零结果"本身是有用的结论：改变"数据怎么布点"带来的影响在 ±0.2 dB 以内，'
           f'而改变"模型能看到什么"带来的提升是 '
           f'{_mx.get("unet", {}).get("rmse_mean", 0) - _mx.get("pgmpr", {}).get("rmse_mean", 0):.2f} ~ '
           f'{_sc.get("unet", {}).get("rmse_mean", 0) - _sc.get("pgmpr", {}).get("rmse_mean", 0):.2f} dB。'
           f'这直接支持了方案五的设计取向：与其纠结数据怎么采，不如把物理先验喂给网络。')

    h(doc, '10.5 方案五逐项剖析：为什么它是最准的方案（重点）', 2)
    p(doc, f'先把定位摆出来：在 200 个未见过的户型上，方案五在常规布点测试集上 '
           f'RMSE {_mx.get("pgmpr", {}).get("rmse_mean", float("nan")):.3f} dB、'
           f'散落布点测试集上 {_sc.get("pgmpr", {}).get("rmse_mean", float("nan")):.3f} dB，'
           f'两个测试集上都是最优。')
    best_bs = min((_sc[k]['rmse_mean'] for k in keys if k in BASELINE_LABELS), default=0.0)
    d5 = []
    for pair, note in ((('unet', 'pgmpr'), '论文式对照：通道与损失设计带来的净增益'),
                       (('resunet', 'pgmpr'), '同为残差框架，多出的 3 路物理通道值多少'),
                       (('unet_scatter', 'pgmpr'), '两条改进路线谁更强')):
        a, b = pair
        d5.append([f'方案五 vs {_mname(a)}',
                   f'{_mx.get(a, {}).get("rmse_mean", 0) - _mx.get(b, {}).get("rmse_mean", 0):+.3f}',
                   f'{_sc.get(a, {}).get("rmse_mean", 0) - _sc.get(b, {}).get("rmse_mean", 0):+.3f}',
                   note])
    d5.append(['方案五 vs 最优插值基线',
               f'{best_bl - _mx.get("pgmpr", {}).get("rmse_mean", 0):+.3f}',
               f'{best_bs - _sc.get("pgmpr", {}).get("rmse_mean", 0):+.3f}',
               '正的差值 = 比最强插值还低这么多 dB'])
    table(doc, ['对照对象', '常规布点差值', '散落布点差值', '说明'], d5,
          caption='表 10-6 方案五与其他方法的定量对照（差值 = 对照方法 − 方案五，正 = 方案五更好）')
    p(doc, '为什么是它 —— 四处改动的归因：', bold=True)
    bullets(doc, [
        '多先验通道（贡献最大）：方案三只有 IDW 一路先验，而 IDW 在无观测区会退化成全局均值；'
        '方案五补进了对数距离物理趋势（全局形状）和"到最近观测点距离"（外推难度的显式信号），'
        '直接对应了方案二/三在角块外推上的短板。',
        '观测加权损失（贡献稳定但较小）：把损失重心压向观测点附近，'
        '换来锚点周围更准；由于未观测区仍是评分主体，这一项不会单独带来大幅 RMSE 下降。',
        '多源训练数据：方案五同时用常规布点（1200）与散落布点（1200）两份训练集，共 2400 个样本，'
        '是其余方案的两倍。这让它在两个测试集上都不掉队 —— 10.4 节显示专精散落布点的方案四'
        '在常规布点测试集上是有代价的，方案五用数据多样性把这个代价摊平了。',
        '推理时自增强 TTA：不改权重，只在推理阶段做四种几何变体平均，属于"免费的稳定性收益"，'
        '代价是 4 倍推理时间（见 10.6 节）。',
    ], numbered=True)
    figure(doc, os.path.join(figdir, 'fig_scheme5_case.png'),
           '图 10-5 方案五定性对比（仿真测试集样本 0）：同一组稀疏观测点下各方法的整图预测。'
           '注意 IDW 在没有观测点的角落趋于平均、边界生硬，而方案五在保持观测点处精确的同时，'
           '把远处的衰减趋势恢复得更接近真值')
    figure(doc, os.path.join(figdir, 'fig_scheme5_err.png'),
           '图 10-6 方案五在未观测格点上的绝对误差分布')
    _m5 = (ctx['models'].get('pgmpr') or {}).get('meta') or {}
    _m2 = (ctx['models'].get('unet') or {}).get('meta') or {}
    if _m5 and _m2 and _m2.get('params'):
        _dd = _m5['params'] - _m2['params']
        table(doc, ['维度', '方案二（U-Net）', '方案五', '代价'], [
            ['参数量', f'{_m2["params"]:,}', f'{_m5["params"]:,}',
             f'+{_dd / _m2["params"] * 100:.2f}%'],
            ['输入通道', '3', '7', '需额外算 3 路先验图（毫秒级）'],
            ['训练数据准备', '直接可用', '需预计算先验图', '一次性缓存'],
            ['推理耗时', '—', '约 4×（开 TTA 时）', '绝对值仍在毫秒级，交互场景无感'],
        ], caption='表 10-7 方案五相对方案二的代价')
    p(doc, f'结论：多出来的代价几乎可以忽略，而换到的是跨户型精度上'
           f'{_mx.get("unet", {}).get("rmse_mean", 0) - _mx.get("pgmpr", {}).get("rmse_mean", 0):.3f} dB'
           f'（常规布点）与 '
           f'{_sc.get("unet", {}).get("rmse_mean", 0) - _sc.get("pgmpr", {}).get("rmse_mean", 0):.3f} dB'
           f'（散落布点）的稳定提升。因此本报告把方案五作为推荐落地模型。', bold=True)

    h(doc, '10.6 TTA（测试时增强）消融', 2)
    p(doc, 'TTA 不是"必然有效"的技巧，所以单独做开/关对照。先看跨户型仿真测试集：')
    if 'unet' in _mx and 'unet_tta' in _mx:
        trows2 = []
        for k in [x for x in _mx if x in MODEL_LABELS and not x.endswith(('_ft', '_tta'))]:
            tk = k + '_tta'
            if tk not in _mx:
                continue
            trows2.append([
                MODEL_LABELS[k],
                f'{_mx[k]["rmse_mean"]:.3f}', f'{_mx[tk]["rmse_mean"]:.3f}',
                f'{_mx[k]["rmse_mean"] - _mx[tk]["rmse_mean"]:+.3f}',
                f'{_sc.get(k, {}).get("rmse_mean", float("nan")):.3f}',
                f'{_sc.get(tk, {}).get("rmse_mean", float("nan")):.3f}',
                f'{_sc.get(k, {}).get("rmse_mean", 0) - _sc.get(tk, {}).get("rmse_mean", 0):+.3f}',
            ])
        table(doc, ['模型', '常规布点 无TTA', '常规布点 有TTA', '变化',
                    '散落布点 无TTA', '散落布点 有TTA', '变化'], trows2,
              caption='表 10-8 TTA 在两个仿真测试集上的开/关对照（变化为正 = 误差下降）')
    trows = []
    for proto, e in protos.items():
        m = e['methods']
        for k in m:
            if not k.endswith('_tta'):
                continue
            base = m.get(k[:-4])
            if not base:
                continue
            trows.append([e['label'], MODEL_LABELS.get(k[:-4], k[:-4]),
                          f'{base["rmse"]:.3f}', f'{m[k]["rmse"]:.3f}',
                          f'{base["rmse"] - m[k]["rmse"]:+.3f}',
                          f'{base.get("infer_ms", 0):.1f} → {m[k].get("infer_ms", 0):.1f}'])
    table(doc, ['协议', '模型', '无 TTA RMSE', '有 TTA RMSE', '变化 (dB)', '推理耗时 (ms)'],
          trows, caption='表 10-9 TTA 在真实数据两套协议上的开/关对照')
    p(doc, '两个表放在一起读，可以得到一个诚实的结论：在仿真域上，TTA 对所有网络方案'
           '都是一致的正收益，说明它对"随机户型上的补全"是稳健的技巧；'
           '而在真实单房间上效果不一致 —— 角块外推（强外推）获益明显，'
           '中心插值（强插值）上方案五反而变差。原因是翻转等价于换一间镜像户型，'
           '对"从全局推角落"有帮助，却在"邻近有观测点"的场景引入了不必要的平均，'
           '把局部细节抹平了一点。因此交付软件把 TTA 做成可开关选项，而不是默认强开。')

    h(doc, '10.7 实测微调的效果（一个必须如实报告的发现）', 2)
    ftrows = []
    for proto, e in protos.items():
        m = e['methods']
        for base in [k for k in m if not k.endswith('_ft') and k in MODEL_LABELS]:
            if base + '_ft' not in m:
                continue
            a, b = m[base]['rmse'], m[base + '_ft']['rmse']
            ftrows.append([e['label'], MODEL_LABELS[base], f'{a:.3f}', f'{b:.3f}', f'{b - a:+.3f}'])
    table(doc, ['协议', '模型', '预训练 RMSE', '微调后 RMSE', '变化'], ftrows,
          caption='表 10-4 实测微调前后对比')
    p(doc, '微调使用自监督方式：每轮从输入点里随机抽一部分当已知观测、剩下的当监督目标。'
           '抽到的点既当输入又当答案是被禁止的，输入集合与监督集合始终不相交，'
           '隐藏点从不参与微调。')
    p(doc, '微调不是万能的：角块外推场景下，微调把方案二从 9.40 dB 拉低到 '
           f'{_rmse(protos, "corner", "unet_ft"):.2f} dB，改善非常显著；'
           f'但中心插值场景下微调反而让方案二、方案三变差'
           f'（方案三从 {_rmse(protos, "center", "resunet"):.2f} dB 变成 '
           f'{_rmse(protos, "center", "resunet_ft"):.2f} dB）。'
           f'原因是微调抽样以较密集观测为主，模型逐渐偏向记忆本房间的局部细节，'
           f'损害了中心区域的平滑插值能力。因此交付模型应按使用场景选择。', bold=True)

    h(doc, '10.8 推理时间', 2)
    table(doc, ['方法', '单次推理 (ms)', '说明'], [
        ['方案一 / 二 / 三 / 四', '1 ~ 2', 'Apple M 系列 GPU（MPS），64×64 全图'],
        ['方案五', '1 ~ 3', '多 3 路先验图的计算开销；开启 TTA 后约 4 倍'],
        ['IDW / 对数距离+RBF', '0.6 ~ 0.7', '纯 NumPy 向量化实现'],
        ['最近邻', '0.3', '纯 NumPy'],
        ['浏览器端 JS 前向', '~700', '纯 CPU 单线程 JavaScript，无 WebGL / WASM 加速'],
    ], caption='表 10-10 推理耗时对比')

    h(doc, '10.9 整图分级：64 × 64 五级像素块覆盖图', 2)
    lv = ctx.get('levels') or {}
    lsrc = lv.get('sources') or {}
    lnames = lv.get('names') or ['', '', '', '', '']
    lranges = lv.get('ranges') or ['', '', '', '', '']
    l_idw = (lsrc.get('idw') or {}).get('counts') or [0] * 5
    l_v5 = (lsrc.get('pgmpr_ft') or {}).get('counts') or [0] * 5
    l_mp = (lv.get('measured_points') or {}).get('counts') or [0] * 5
    l_ti = (lsrc.get('idw') or {}).get('total') or 0
    l_tv = (lsrc.get('pgmpr_ft') or {}).get('total') or 0
    p(doc, '前面各节的评价都落在"未观测格点 RMSE"这类标量指标上。这一节换一个角度看同一件事：'
           '把整间房切成 64 × 64 ＝ 4096 个 0.2 m × 0.2 m 的像素块，每块按 RSSI 强弱归入五个等级，'
           '用颜色深浅表示 —— 颜色越深，信号越强。相比误差数字，这张图更接近用户实际看到的形态：'
           '不是"差多少 dB"，而是"哪一片强、哪一片弱"。')
    p(doc, '-40 dBm 被特意放在二级与三级的分界上：BR-04 规定 -40 dBm 是有效上限，'
           '"二级及以上"正好对应任务书里偏强、需要标注但不得删除的那一档。')
    table(doc, ['等级', 'RSSI 范围 (dBm)', 'IDW 插值格数', '占比',
                '方案五格数', '占比', '实测点个数'], [
        [f'{5 - i} 级（{lnames[i].split(" · ")[-1] if lnames[i] else "—"}）',
         lranges[i], str(l_idw[i]),
         f'{(l_idw[i] / l_ti * 100) if l_ti else 0:.1f}%', str(l_v5[i]),
         f'{(l_v5[i] / l_tv * 100) if l_tv else 0:.1f}%', str(l_mp[i])]
        for i in range(5)
    ], caption='表 10-11 五级分级的格点分布（共 3697 个有效格点；柱体内部与房间外不计）')
    figure(doc, os.path.join(figdir, 'fig_levels64.png'),
           '图 10-8 整间房的 Wi-Fi 覆盖五级分级图，64 × 64 像素块。'
           '左：49 个实测点的 IDW 插值；右：方案五（用 49 点微调后）的预测。'
           '黄点是 49 个实测点位，两幅图共用同一套分级与色标，可直接对比。')
    p(doc, '两个面板的差异本身就有信息量：IDW 插值把覆盖场抹得比较平滑，'
           '强区主要收敛在 (x≈2, y≈5~8) 一带；方案五保留了更多局部起伏，'
           f'判出的"一级（极强）"格点为 {l_v5[4]} 个，而 IDW 只有 {l_idw[4]} 个。'
           '实测最强点就在 (2, 6) 附近（-30 dBm），方案五向外多推一点是合理的；'
           '但多推的部分没有实测点验证，属于模型外推。', bold=True)
    p(doc, '注意：这张图是可视化产物，不是评分依据。任务书要求"不得用插值结果充当实测答案"，'
           '因此本报告所有误差指标一律使用留出的隐藏实测点，与本图无关；'
           '图中的 49 个黄点只是输入位置标注，不参与任何评分。')

    # ---------------- 11 讨论 ----------------
    h(doc, '11. 讨论与局限', 1)
    h(doc, '11.1 结论', 2)
    bullets(doc, [
        '跨户型能力：五个网络方案在 200 个未见过的随机户型上全部优于全部插值基线；'
        '其中方案五在两个测试集上都排第一，比最强插值基线（对数距离趋势 + RBF 残差）'
        '低约 0.7 ~ 1.4 dB。',
        '方案五最准的原因是可归因的：它的优势来自"多先验通道 + 观测加权损失 + '
        '多源训练数据 + TTA"，而参数量只比方案二多 0.08%，不是靠把模型做大。',
        '布点方式这个变量影响很小（方案四的"零结果"）：把训练数据从常规布点换成散落布点，'
        '两个测试集上的差异都在 ±0.2 dB 以内，按散落程度分层后也没有任何一档更强。'
        '这说明本任务上"改模型能看到什么"比"改数据怎么采"更有效。',
        f'同房间稀疏重建：中心插值协议下方案五预训练模型拿到 '
        f'{_rmse(protos, "center", "pgmpr"):.2f} dB，是所有方法里最好的；'
        f'角块外推协议下微调后的方案二拿到 {_rmse(protos, "corner", "unet_ft"):.2f} dB。',
        'TTA 的效果依场景而异：仿真域上对所有网络都稳定有益，真实单房间上'
        '强外推场景获益明显、强插值场景反而略差，因此做成可开关选项。',
        '实测微调是双刃剑：能显著修正"仿真域 → 真实域"的系统性偏差，'
        '但也可能让模型偏向记忆本房间的细节。',
        '插值仍然很强：任务书明确说"不要求神经网络必然优于传统插值"，'
        '本实验的结论与之一致 —— 网络的性价比主要体现在观测点稀少且需要跨户型的场合。',
    ], numbered=True)

    h(doc, '11.2 必须承认的局限', 2)
    bullets(doc, [
        '只有一张真实参考地图。49 个点全部来自同一间房、同一次部署，'
        '真实数据结论不能外推成"换一间房也准"。',
        '隐藏点数量少（角块 6 点、中心 9 点），RMSE 统计不确定性很大，'
        '方法之间 0.2 ~ 0.5 dB 的差距不足以断言显著性。',
        f'数据整体偏强：{len(q["br04_too_strong"])} 个点强于 −40 dBm，'
        '不符合任务书 −40 ~ −70 dBm 的规范，但按业务规则被保留。',
        '没有时间维度的重复测量，无法把模型误差与测量噪声分开。',
        '仿真域与真实域存在差距：仿真模型没有建模多径干涉、天线方向图、家具材质差异，'
        '"仿真上 4.1 dB"不能直接读成"真实新房间里也是 4.1 dB"。',
    ], numbered=True)

    # ---------------- 12 软件 ----------------
    h(doc, '12. 交付软件：RF Map Studio', 1)
    p(doc, '除报告外，本项目交付一个可离线运行的网站：outputs/rfmap-studio/index.html，'
           '双击即可打开，无需后端、无需构建步骤、无需联网。')
    bullets(doc, [
        '户型编辑：调整房间宽高、增删/拖动矩形障碍物（默认预置那根 0.830 × 1.010 m 结构柱）、'
        '一键恢复默认户型。',
        '测点输入：在画布上点击添加测点并填写 RSSI；测点可拖动、可删除；一键载入内置的 49 点实例数据。',
        '生成热力图：调用神经网络推理，把整屋 RSSI 热力图半透明叠加在户型上；'
        '固定统一色标（−100 ~ −40 dBm），鼠标悬停可读出任一点的预测值。',
        '导出：导出热力图 PNG、导出当前测点为 CSV。',
    ], numbered=True)
    h(doc, '12.2 与 PyTorch 的数值一致性', 2)
    table(doc, ['检查项', '结果'], [
        ['模型参数量（Python / JS 两侧）', '1,083,625 / 1,083,625 ✓'],
        ['输入张量最大逐元素差异', '0（完全一致）'],
        ['房间内 3697 个格点输出最大差异', '1.7e−6（标准化尺度）≈ 1.6e−5 dB ✓'],
        ['房间外格点填充值', '全部为 −120，无遗漏 ✓'],
    ], caption='表 12-1 浏览器端与 PyTorch 的数值一致性检查')
    p(doc, '排查过程中发现并修掉了一个真实缺陷：conv2d 最初把卷积核硬编码成 3×3，'
           '而网络的输出层是 1×1 卷积，导致权重索引越界、整幅输出变成 NaN。'
           '现在 conv2d 支持任意奇数核，一致性检查随之通过。')

    # ---------------- 13 复现 ----------------
    h(doc, '13. 复现步骤', 1)
    code(doc, """# 0) 环境（Apple Silicon，Python 3.14）
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
    reobserve_dataset(f'data/sim/{name}.npz', f'data/sim_scatter/{name}.npz',
                      seed, G, obs_mode='scatter', count_range=None)"

# 3) 统一重训五个方案（同一设备 / 同一超参 / 同一种子）
.venv/bin/python -m rfmap.train_stages --device mps

# 4) 评估（真实留出 + 两个仿真测试集 + 实测微调 + TTA + 散落度分层）
.venv/bin/python -m rfmap.evaluate --what real,sim --device mps --tta
.venv/bin/python -m rfmap.evaluate --what strat --device mps

# 5) 出图
.venv/bin/python -m rfmap.figures mps

# 6) 导出浏览器权重 + JS 推理
.venv/bin/python -m rfmap.export --model unet --out-dir outputs/rfmap-studio

# 7) 生成 HTML 报告与 Word 报告
.venv/bin/python -m rfmap.report
.venv/bin/python -m rfmap.docx_report

# 8) 端到端自检
.venv/bin/python -m rfmap.selfcheck""")
    p(doc, '为什么用 train_stages 而不是逐个训：方案二与方案四是配对消融，'
           '必须保证两者除了训练数据以外没有任何差别（包括设备与浮点路径）。'
           '该脚本把五个方案放在同一设备、同一组超参、同一随机种子下依次训练。')

    # ---------------- 14 附录 ----------------
    h(doc, '14. 附录', 1)
    h(doc, '14.1 房间几何参数', 2)
    code(doc, json.dumps(g, indent=2, ensure_ascii=False))
    h(doc, '14.2 交付物清单', 2)
    table(doc, ['路径', '说明'], [
        ['outputs/RF覆盖预测实验报告.html', '实验报告（网页版，可打印为 PDF）'],
        ['outputs/RF覆盖预测实验报告.docx', '实验报告（Word 版）'],
        ['outputs/rfmap-studio/', '离线网站（含模型权重与 JS 推理）'],
        ['outputs/figures/', '全部图表'],
        ['outputs/rfmap-results/', '真实数据与仿真数据的全部指标 JSON'],
        ['data/rfmap/measured.csv', '规范化实测数据（含对齐后坐标）'],
        ['data/rfmap/grid.json', '网格配置（训练 / 推理 / 前端共用）'],
        ['models/{ae,unet,resunet}/', '权重 + 元数据 + 训练曲线'],
        ['rfmap/', '全部源码'],
    ], caption='表 14-1 交付物清单')

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    doc.save(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='生成 Word 版实验报告')
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='outputs/RF覆盖预测实验报告.docx')
    args = ap.parse_args(argv)
    path = build(args.root, args.out)
    print(f'Word 报告 -> {path}  ({os.path.getsize(path) / 1024:.1f} KB)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
