#!/usr/bin/env python3
"""
plot_motion.py —— 把 SSE 事件日志画成波形图（零依赖，输出 SVG）

不需要 matplotlib、不需要外网：直接生成 SVG，用浏览器或预览打开即可，
也能插进报告（Word/Pages 可插入 SVG，或转成 PNG）。

用法：
    python3 lab-tools/plot_motion.py out.svg data/*/*.log
    python3 lab-tools/plot_motion.py wave.svg .deepworks/tmp/block.log
"""
from __future__ import annotations

import sys
from typing import List, Tuple

sys.path.insert(0, "lab-tools")
from parse_events import load  # noqa: E402  复用同一个解析逻辑

W, H = 1080, 540
L, R, T, B = 70, 30, 60, 110
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]
THRESHOLD = 0.662185  # 上次校准后的阈值，仅作参考线


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build(series: List[Tuple[str, List[Tuple[int, str, float]]]], threshold: float) -> str:
    # 统一时间轴：每个文件用自己的相对秒
    tmax = 0.0
    for _, rows in series:
        if rows:
            tmax = max(tmax, (rows[-1][0] - rows[0][0]) / 1000.0)
    tmax = max(tmax, 1.0)

    def x(t: float) -> float:
        return L + (t / tmax) * (W - L - R)

    def y(s: float) -> float:
        return T + (1.0 - max(0.0, min(1.0, s))) * (H - T - B)

    p = []
    p.append('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="Helvetica,Arial,sans-serif">' % (W, H, W, H))
    p.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    p.append('<text x="%d" y="30" font-size="20" font-weight="bold" fill="#111">ESPectre 运动分数波形（score vs 时间）</text>' % L)
    p.append('<text x="%d" y="48" font-size="12" fill="#666">由真实设备事件流绘制 · 阈值参考线 %.3f · 红带 = motion 判定</text>' % (L, threshold))

    # 网格 + Y 轴刻度
    for i in range(6):
        v = i / 5.0
        yy = y(v)
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e8e8e8" stroke-width="1"/>' % (L, yy, W - R, yy))
        p.append('<text x="%d" y="%.1f" font-size="11" fill="#888" text-anchor="end">%.1f</text>' % (L - 8, yy + 4, v))
    # 阈值线
    p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#888" stroke-width="1.5" stroke-dasharray="6,4"/>' % (L, y(threshold), W - R, y(threshold)))
    p.append('<text x="%d" y="%.1f" font-size="11" fill="#555">threshold %.2f</text>' % (W - R - 130, y(threshold) - 6, threshold))

    # X 轴刻度
    for i in range(7):
        t = tmax * i / 6.0
        xx = x(t)
        p.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#bbb" stroke-width="1"/>' % (xx, H - B, xx, H - B + 5))
        p.append('<text x="%.1f" y="%d" font-size="11" fill="#888" text-anchor="middle">%.0f</text>' % (xx, H - B + 20, t))
    p.append('<text x="%d" y="%d" font-size="12" fill="#555" text-anchor="middle">时间（秒）</text>' % ((L + W - R) // 2, H - B + 40))
    p.append('<text x="18" y="%d" font-size="12" fill="#555" transform="rotate(-90 18 %d)" text-anchor="middle">score</text>' % ((T + H - B) // 2, (T + H - B) // 2))

    # 每个文件一条折线 + 底部 motion 状态带
    band_y = H - B + 50
    for idx, (name, rows) in enumerate(series):
        if not rows:
            continue
        color = COLORS[idx % len(COLORS)]
        t0 = rows[0][0]
        pts = " ".join("%.1f,%.1f" % (x((t - t0) / 1000.0), y(s)) for t, _, s in rows)
        p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="1.8" stroke-linejoin="round"/>' % (pts, color))
        # motion 段标在底部状态带（同一行会重叠，故每个文件错开 3px）
        for t, st, _ in rows:
            if st == "motion":
                p.append('<rect x="%.1f" y="%.1f" width="%.1f" height="4" fill="%s" opacity="0.75"/>'
                         % (x((t - t0) / 1000.0), band_y + idx * 5, max(1.0, (W - L - R) / max(1, len(rows))), color))

    # 图例 + 统计
    ly = T + 6
    # 图例白底，避免被折线穿过影响可读性
    p.append('<rect x="%d" y="%d" width="352" height="%d" rx="6" fill="#ffffff" fill-opacity="0.9" stroke="#e0e0e0"/>'
             % (W - R - 358, ly - 8, len(series) * 18 + 14))
    for idx, (name, rows) in enumerate(series):
        color = COLORS[idx % len(COLORS)]
        n = len(rows)
        if n:
            motion = sum(1 for _, st, _ in rows if st == "motion")
            avg = sum(s for _, _, s in rows) / n
            label = "%s   motion %.1f%%   score avg %.3f" % (name.split("/")[-1], motion * 100.0 / n, avg)
        else:
            label = "%s   （无数据）" % name.split("/")[-1]
        p.append('<rect x="%d" y="%d" width="14" height="3" fill="%s"/>' % (W - R - 346, ly + idx * 18, color))
        p.append('<text x="%d" y="%d" font-size="12" fill="#333">%s</text>' % (W - R - 326, ly + 4 + idx * 18, esc(label)))
    p.append('<text x="%d" y="%d" font-size="11" fill="#888">底部色带 = 该时刻判定为 motion 的区间</text>' % (L, band_y + 24 + len(series) * 5))
    p.append("</svg>")
    return "\n".join(p)


def main(argv: List[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    out = argv[1]
    threshold = THRESHOLD
    series = []
    for path in argv[2:]:
        rows = load(path)
        series.append((path, rows))
        print("%-40s %d 条事件" % (path, len(rows)))
    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(build(series, threshold))
    except IOError as exc:
        print("!! 写不出去 %s: %s" % (out, exc))
        return 1
    print("已生成波形图: %s（用浏览器打开即可）" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
