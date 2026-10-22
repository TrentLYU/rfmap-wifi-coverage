#!/usr/bin/env python3
"""
parse_events.py —— 把 ESPectre 的 SSE 事件日志变成实验结论

输入：curl 抓下来的 /espectre/v1/events 日志（SSE 文本）
输出：事件数、时长、motion 占比、score 统计、分段节奏、自动判定

用法：
    python3 lab-tools/parse_events.py .deepworks/tmp/block.log
    python3 lab-tools/parse_events.py .deepworks/tmp/*.log      # 批量对比
"""
from __future__ import annotations

import json
import re
import sys
from typing import Dict, List, Optional, Tuple

# 一次采样的分段时间（秒），用于观察"什么时候动的"
BUCKET_SEC = 5.0
# 判定"本底饱和"的参考门槛
SATURATION_SCORE = 0.80

JSON_RE = re.compile(r"\{.*\}")


def load(path: str) -> List[Tuple[int, str, float]]:
    """读出 [(timestamp_ms, state, score), ...]，坏行自动跳过。"""
    rows: List[Tuple[int, str, float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            m = JSON_RE.search(line)
            if not m:
                continue
            try:
                obj = json.loads(m.group(0))
            except ValueError:
                continue
            ts = int(obj.get("timestamp_ms", 0))
            state = str(obj.get("state", "?"))
            try:
                score = float(obj.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            rows.append((ts, state, score))
    return rows


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def verdict(n: int, motion_ratio: float, scores: List[float], first_bucket: float) -> str:
    if n == 0:
        return "❌ 没有解析到任何事件 —— 检查日志是否抓到数据（设备是否在推事件）"
    smin = min(scores) if scores else 0.0
    if motion_ratio >= 0.95 and smin >= SATURATION_SCORE:
        return ("⚠️ 全程 motion 且最低分就已达 %.2f —— 更像环境噪声饱和或持续遮挡，"
                "信噪比≈1，调参无效，建议换场地/时间或恢复安静后重采" % smin)
    if motion_ratio >= 0.80:
        return "✅ 持续触发 —— 典型的『沿链路走动』或『站在链路正中遮挡』"
    if motion_ratio >= 0.30:
        return "✅ 间歇触发 —— 走动有效但节奏不稳；若期望持续运动，检查走位是否穿过链路正中"
    if motion_ratio >= 0.10:
        return "🔸 少量触发 —— 可能有动作但偏弱；优先检查 RSSI 是否在 -40~-70 dBm"
    if first_bucket >= 0.95:
        return "🔸 开始时触发、随后安静 —— 可能是有人刚离开，本段更像『本底/静止』"
    return "✅ 基本静默 —— 适合作『静止对照』或『空场本底』基线"


def report(path: str) -> Dict[str, float]:
    rows = load(path)
    n = len(rows)
    print("=" * 68)
    print("文件: %s" % path)
    if n == 0:
        print(verdict(0, 0.0, [], 0.0))
        return {}

    ts = [r[0] for r in rows]
    scores = [r[2] for r in rows]
    states = [r[1] for r in rows]
    motion_n = sum(1 for s in states if s == "motion")
    motion_ratio = motion_n / n
    duration = (ts[-1] - ts[0]) / 1000.0
    gap = duration / (n - 1) if n > 1 else 0.0

    print("事件数: %d    时长: %.1f s    平均间隔: %.3f s (≈%.1f 条/秒)"
          % (n, duration, gap, 1.0 / gap if gap else 0.0))
    print("motion 占比: %.1f%%   (%d/%d)" % (motion_ratio * 100, motion_n, n))
    print("score: avg=%.4f  median=%.4f  p95=%.4f  max=%.4f  min=%.4f"
          % (sum(scores) / n, pct(scores, 0.5), pct(scores, 0.95), max(scores), min(scores)))

    # 每 BUCKET_SEC 秒一段：看运动出现在哪一段
    buckets: List[Tuple[float, int, float]] = []
    if duration > 0:
        n_bucket = int(duration // BUCKET_SEC) + 1
        for b in range(n_bucket):
            sel = [(st, sc) for (t, st, sc) in rows
                   if b * BUCKET_SEC <= (t - ts[0]) / 1000.0 < (b + 1) * BUCKET_SEC]
            if not sel:
                continue
            mo = sum(1 for st, _ in sel if st == "motion")
            avg = sum(sc for _, sc in sel) / len(sel)
            buckets.append((b * BUCKET_SEC, mo, avg))
    if buckets:
        print("分段(每 %.0f s): %s" % (BUCKET_SEC, "  ".join(
            "%.0fs:%d动/%.2f" % (t, mo, avg) for t, mo, avg in buckets)))
    first_bucket = (buckets[0][1] / max(1, len([1 for (t, _, _) in rows
                                              if (t - ts[0]) / 1000.0 < BUCKET_SEC]))) if buckets else 0.0
    print("判定: %s" % verdict(n, motion_ratio, scores, first_bucket))
    return {"n": float(n), "duration_s": duration, "motion_ratio": motion_ratio,
            "score_avg": sum(scores) / n, "score_p95": pct(scores, 0.95)}


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    summary: List[Tuple[str, Dict[str, float]]] = []
    for path in argv[1:]:
        try:
            summary.append((path, report(path)))
        except IOError as exc:
            print("!! 打不开 %s: %s" % (path, exc))
    # 多文件时输出对比表，便于写进报告
    if len(summary) > 1:
        print("=" * 68)
        print("对比表（可直接抄进实验报告）")
        print("%-28s %6s %8s %10s %9s" % ("文件", "事件", "时长s", "motion%", "score均"))
        for path, s in summary:
            if not s:
                continue
            print("%-28s %6.0f %8.1f %9.1f%% %9.4f"
                  % (path.split("/")[-1], s["n"], s["duration_s"],
                     s["motion_ratio"] * 100, s["score_avg"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
