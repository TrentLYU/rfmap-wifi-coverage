#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RF Map 点位 RSSI 采集工具（局域网离线可用，仅 Python 标准库）。

对应任务书《室内 RF Map 数据与网络训练指导》"数据怎样采集"一节：
  · 固定路由器位置与配置，只把 ESP32 移到房间不同位置；
  · 各点保持相同接收高度和朝向，连续记录一段时间；
  · 以统一的均值/中位数作为该点代表值，同时保留波动（标准差）；
  · 记录房间、测点坐标、AP 标识、时间与部署条件。

本工具只采"实测点"（任务书明确要求：不得用插值结果充当实测答案）。
输入点 / 隐藏实测点的划分留到训练阶段再做 —— 所以这里先把网格**采全**。

用法示例：
    # 0) 先看会采哪些点，不连设备
    python3 lab-tools/rssi_survey.py --grid 0,0,6,4,1 --dry-run

    # 1) 一个部署 = 一份参考地图（交互式，逐点做）
    python3 lab-tools/rssi_survey.py --ip 10.11.0.113 \\
        --map-id ROOM-A-dp1 --room ROOM-A --height 1.0 \\
        --grid 0,0,6,4,1 --seconds 20 --settle 3

    # 2) 只补采指定点位
    python3 lab-tools/rssi_survey.py --ip 10.11.0.113 \\
        --map-id ROOM-A-dp1 --points "1,2 3,4"

    # 3) 换了路由器位置 → 换一个 --map-id 再来一遍（必须多份独立地图）

中断后按同一条命令重跑即可续采：已记录过的点位会自动跳过。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.request
from datetime import datetime

PORT = 62587
ORIGIN = "https://espectre.dev"

CSV_COLUMNS = [
    "map_id", "room", "x_m", "y_m", "height_m", "ap_id",
    "ssid", "band", "channel",
    "rssi_mean", "rssi_median", "rssi_std", "rssi_min", "rssi_max",
    "n_samples", "n_errors", "ts_start", "ts_end", "operator", "note",
]


# ---------------------------------------------------------------- 设备访问

def fetch(ip: str, path: str, timeout: float = 4.0):
    """读 ESPectre 局域网接口；必须带 Origin 头，否则设备会拒绝。"""
    req = urllib.request.Request(
        f"http://{ip}:{PORT}/espectre/v1{path}",
        headers={"Origin": ORIGIN},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def preflight(ip: str):
    """采集前自检：设备在不在、频段对不对。返回摘要 dict 或 None。"""
    print(f"[i] 自检 http://{ip}:{PORT}/espectre/v1 ...")
    try:
        fetch(ip, "/health", timeout=4.0)
    except Exception as exc:  # noqa: BLE001
        print(f"[x] 连不上设备 —— {exc}")
        print("    依次排查：")
        print("      1) python3 lab-tools/find_espectre.py        找设备 IP")
        print("      2) 电脑与 ESP32 是否在同一个 2.4 GHz 网络（同网段）")
        print("      3) 路由器是否关闭 AP 隔离 / 客户端隔离")
        print("      4) 设备是否已上电并配好了网（串口跑 --state 看状态）")
        return None

    info = {"ip": ip}
    try:
        w = fetch(ip, "/wifi", timeout=4.0)
        info.update({
            "ssid": w.get("ssid", ""),
            "band": w.get("band", ""),
            "channel": w.get("channel", ""),
            "rssi_dbm": w.get("rssi_dbm"),
        })
        band = str(info.get("band") or "?")
        print(f"[i] /health OK   ssid={info['ssid']}  band={band}  "
              f"ch={info['channel']}  rssi={info['rssi_dbm']} dBm")
        if band and band not in ("2g", "?"):
            print(f"[!] 当前频段是 {band} —— ESP32-C6 只支持 2.4 GHz，请重配网络")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] /wifi 读不到（{exc}）—— 继续，但请留意采集是否全失败")
    return info


# ---------------------------------------------------------------- 点位解析

def parse_points(text: str):
    pts = []
    for token in text.replace(";", " ").replace("\n", " ").split():
        if not token:
            continue
        try:
            a, b = token.split(",")
            pts.append((float(a), float(b)))
        except ValueError:
            raise SystemExit(f"[x] 点位格式应为 x,y：{token!r}")
    return pts


def gen_grid(spec: str):
    """x0,y0,x1,y1,step -> 按 y 分行、行内按 x 递增的点位列表。"""
    try:
        x0, y0, x1, y1, step = (float(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit("[x] --grid 格式应为 x0,y0,x1,y1,step，例如 0,0,6,4,1")
    if step <= 0:
        raise SystemExit("[x] --grid 的步长必须 > 0")
    xs = []
    n = 0
    while x0 + n * step <= x1 + 1e-9:
        xs.append(round(x0 + n * step, 3))
        n += 1
    ys = []
    n = 0
    while y0 + n * step <= y1 + 1e-9:
        ys.append(round(y0 + n * step, 3))
        n += 1
    return [(x, y) for y in ys for x in xs]


# ---------------------------------------------------------------- CSV 读写

def load_recorded(path: str):
    """读回已采点位，用于中断续采。"""
    done = set()
    if os.path.exists(path):
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done.add((row["map_id"],
                              float(row["x_m"]),
                              float(row["y_m"])))
        except Exception as exc:  # noqa: BLE001
            print(f"[!] 读取已有 CSV 失败（将重新开始）：{exc}")
    return done


def append_row(path: str, row: dict):
    new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------- 采样

def sample(ip: str, seconds: float, interval: float):
    """连续读取 /wifi 的 rssi_dbm，返回该点位的统计与时间戳。"""
    vals, errors = [], 0
    ssid = band = channel = ""
    t0 = time.time()
    end = t0 + seconds
    last_tick = -1

    while True:
        now = time.time()
        if now >= end:
            break
        try:
            w = fetch(ip, "/wifi", timeout=3.0)
            v = w.get("rssi_dbm")
            if v is not None:
                vals.append(float(v))
            ssid = w.get("ssid", ssid)
            band = w.get("band", band)
            channel = w.get("channel", channel)
        except Exception:  # noqa: BLE001
            errors += 1

        tick = int(now - t0)
        if tick != last_tick:
            last_tick = tick
            pct = min(100, int((now - t0) / seconds * 100))
            sys.stdout.write(f"\r    采样中 {pct:>3}%  n={len(vals)}  "
                             f"err={errors}   ")
            sys.stdout.flush()
        time.sleep(interval)

    sys.stdout.write("\r" + " " * 56 + "\r")
    return vals, errors, ssid, band, channel, t0, time.time()


def summarize(vals):
    if not vals:
        return None
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {
        "rssi_mean": round(statistics.fmean(vals), 2),
        "rssi_median": round(statistics.median(vals), 2),
        "rssi_std": round(std, 2),
        "rssi_min": round(min(vals), 1),
        "rssi_max": round(max(vals), 1),
    }


def verdict(rssi: float) -> str:
    if rssi > -40:
        return "太近 -> 应把 ESP32 移远到 3~5 m"
    if rssi >= -70:
        return "合适"
    return "偏弱 -> 检查距离与遮挡"


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(
        description="RF Map 点位 RSSI 采集（ESPectre 局域网接口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="点位先采全，输入点/隐藏点的划分留到训练阶段做。",
    )
    ap.add_argument("--ip", help="设备局域网 IP（采集必需）")
    ap.add_argument("--map-id", default="MAP-1",
                    help="部署/地图标识，如 ROOM-A-dp1（默认 MAP-1）")
    ap.add_argument("--room", default="", help="房间名，如 ROOM-A")
    ap.add_argument("--height", type=float, default=1.0, help="接收高度 m")
    ap.add_argument("--ap-id", default="", help="AP 标识（BSSID 或路由器名）")
    ap.add_argument("--grid", help="自动生成网格点位：x0,y0,x1,y1,step")
    ap.add_argument("--points", help='手工点位："x,y x,y ..."')
    ap.add_argument("--seconds", type=float, default=20.0, help="每点采样时长 s")
    ap.add_argument("--interval", type=float, default=0.5, help="采样间隔 s")
    ap.add_argument("--settle", type=float, default=3.0, help="到点后稳定等待 s")
    ap.add_argument("--out", help="CSV 路径，默认 data/rfmap/<map-id>.csv")
    ap.add_argument("--operator", default=os.environ.get("USER", ""),
                    help="采集人")
    ap.add_argument("--note", default="", help="备注（环境/人员/空调等）")
    ap.add_argument("--yes", action="store_true", help="不再等回车，自动连采")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印点位清单，不连设备")
    args = ap.parse_args()

    # ---- 点位 ----
    if args.grid:
        pts = gen_grid(args.grid)
    elif args.points:
        pts = parse_points(args.points)
    else:
        ap.error("请用 --grid 或 --points 指定点位")

    if args.dry_run:
        print(f"[i] map_id={args.map_id}  room={args.room}  "
              f"height={args.height} m  共 {len(pts)} 个点位")
        print("[i] 按 y 分行、行内按 x 递增（方便一行一行走）：")
        for i, (x, y) in enumerate(pts, 1):
            print(f"    {i:>3}. x={x:>6.2f}  y={y:>6.2f}")
        print("\n[i] 去掉 --dry-run、加上 --ip <设备IP> 即可开始采集。")
        return 0

    if not args.ip:
        ap.error("采集需要 --ip <设备IP>（可用 find_espectre.py 查到）")

    out = args.out or os.path.join("data", "rfmap", f"{args.map_id}.csv")
    outdir = os.path.dirname(out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    # ---- 自检 ----
    dev = preflight(args.ip)
    if dev is None:
        return 1

    done = load_recorded(out)
    todo = [(x, y) for (x, y) in pts if (args.map_id, x, y) not in done]

    print(f"\n[i] 地图 {args.map_id} / 房间 {args.room or '-'} / 高度 {args.height} m")
    print(f"[i] 点位共 {len(pts)} 个，已完成 {len(pts) - len(todo)} 个，"
          f"本次待采 {len(todo)} 个")
    print(f"[i] 每点稳定 {args.settle:g}s + 采样 {args.seconds:g}s "
          f"（间隔 {args.interval:g}s）")
    print(f"[i] 输出：{out}")
    if not todo:
        print("[i] 没有待采点位，直接结束。")
        return 0

    print("\n执行规范（保证数据可比，任务书要求）：")
    print("  · 路由器位置/配置全程不动，只移动 ESP32；")
    print("  · 每点保持**相同高度和朝向**（朝哪边要一致）；")
    print("  · 放好设备后**人退到固定位置**（如门口），别站在设备旁；")
    print("  · 一次只改一个变量；环境（空调/人）尽量保持一致。\n")

    # ---- 元数据 ----
    meta = {
        "map_id": args.map_id, "room": args.room, "height_m": args.height,
        "ap_id": args.ap_id, "device": dev, "grid": args.grid,
        "points_total": len(pts), "points_recorded_before": len(pts) - len(todo),
        "seconds_per_point": args.seconds, "sample_interval_s": args.interval,
        "operator": args.operator, "note": args.note,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    metapath = os.path.splitext(out)[0] + ".meta.json"
    with open(metapath, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[i] 元数据：{metapath}\n")

    # ---- 逐点采集 ----
    results = []
    try:
        for i, (x, y) in enumerate(todo, 1):
            print(f"--- [{i}/{len(todo)}] 点位 x={x} m, y={y} m ---")
            print("    1) 把 ESP32 放到该点（高度/朝向与前一点一致）")
            print(f"    2) 人退到固定位置，设备不要被身体挡住")
            if not args.yes:
                try:
                    input("    >> 就绪后按回车开始（Ctrl-C 可随时中断）... ")
                except EOFError:
                    pass

            if args.settle > 0:
                print(f"    等稳定 {args.settle:g}s ...")
                time.sleep(args.settle)

            vals, errors, ssid, band, channel, t0, t1 = sample(
                args.ip, args.seconds, args.interval)
            st = summarize(vals)

            if st is None:
                print("    [x] 采样全部失败 —— 这个点没有数据，先排查链路再重跑")
                continue

            row = {
                "map_id": args.map_id, "room": args.room,
                "x_m": x, "y_m": y, "height_m": args.height,
                "ap_id": args.ap_id, "ssid": ssid, "band": band,
                "channel": channel, "n_samples": len(vals),
                "n_errors": errors,
                "ts_start": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
                "ts_end": datetime.fromtimestamp(t1).isoformat(timespec="seconds"),
                "operator": args.operator, "note": args.note,
                **st,
            }
            append_row(out, row)
            results.append(row)
            print(f"    RSSI 均值 {st['rssi_mean']:>7} dBm  "
                  f"中位数 {st['rssi_median']:>7}  "
                  f"标准差 {st['rssi_std']:>5}  "
                  f"范围 [{st['rssi_min']}, {st['rssi_max']}]  n={len(vals)}")
            print(f"    -> {verdict(st['rssi_mean'])}")
            if errors:
                print(f"    [!] 有 {errors} 次读取失败（设备移动中/丢包）")
            print()
    except KeyboardInterrupt:
        print("\n[i] 已中断。按同一条命令重跑会从没采的点继续。\n")

    # ---- 摘要 ----
    print("=" * 60)
    print(f"[i] 本次新增 {len(results)} 个点位，文件：{out}")
    if results:
        means = [r["rssi_mean"] for r in results]
        print(f"[i] 本次均值范围 {min(means):.1f} ~ {max(means):.1f} dBm")
        bad = [r for r in results if not (-70 <= r["rssi_mean"] <= -40)]
        if bad:
            print(f"[!] 有 {len(bad)} 个点不在 -40~-70 dBm 的合适区间：")
            for r in bad:
                print(f"      x={r['x_m']} y={r['y_m']}  "
                      f"{r['rssi_mean']} dBm")
            print("    提示：过强通常是离路由器太近；过弱的点可检查遮挡，")
            print("          但**不要删点**——任务书要求原始实测数据留档。")
    print("\n下一步：")
    print("  · 换路由器位置后，用新的 --map-id 再采一份（必须多份独立地图）")
    print("  · 全部采完后进入训练阶段：先划分场景组与输入/隐藏点，再建模型")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
