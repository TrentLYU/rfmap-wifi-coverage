#!/usr/bin/env python3
"""ESPectre 链路质量监视器。

实时显示 ESP32 与路由器之间的 RSSI，并直接给出"摆放是否合适"的判断。
CSI 运动检测的灵敏度取决于这条链路的质量：太强（设备挨太近）人走不进
链路、太弱（太远）信号不稳，都会导致测不到运动。

用法：
    python3 lab-tools/link_monitor.py                # 默认 192.168.31.66
    python3 lab-tools/link_monitor.py 192.168.1.50   # 指定 IP
    python3 lab-tools/link_monitor.py 192.168.31.66 60   # 监视 60 秒
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

ORIGIN = "https://espectre.dev"
PORT = 62587


def fetch(ip: str, path: str, timeout: float = 4.0):
    req = urllib.request.Request(
        f"http://{ip}:{PORT}/espectre/v1{path}",
        headers={"Origin": ORIGIN},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def verdict(rssi: int) -> str:
    """按 RSSI 给出摆放建议。"""
    if rssi > -45:
        return "太近 !! 人走不进链路 -> 把 ESP32 移远到 3~5 米"
    if rssi > -50:
        return "偏近  -> 建议再移远一点"
    if rssi >= -70:
        return "合适  -> 运动检测最灵敏, 可以开始实验"
    if rssi >= -80:
        return "偏远  -> 信号开始变弱, 可能不稳定"
    return "太远  -> 可能丢包, 拉近一些"


def bar(rssi: int, width: int = 30) -> str:
    """把 RSSI 映射成一个直观的强度条（右端最强）。"""
    level = max(0, min(width, int((rssi + 95) / 65 * width)))
    return "#" * level + "." * (width - level)


def main() -> int:
    ip = sys.argv[1] if len(sys.argv) > 1 else "192.168.31.66"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0  # 0 = 一直跑

    print(f"监视 ESPectre 链路质量 @ {ip}:{PORT}")
    print("现在可以移动 ESP32，观察 RSSI 变化，目标是 -50 ~ -70 dBm")
    print("-" * 68)

    start = time.time()
    best = None
    try:
        while True:
            try:
                w = fetch(ip, "/wifi")
                rssi = int(w.get("rssi_dbm", 0))
                ch = w.get("channel", "?")
                band = w.get("band", "?")
                ssid = w.get("ssid", "?")
                print(
                    f"RSSI {rssi:>4} dBm [{bar(rssi)}]  ch={ch:<3} {band}  "
                    f"ssid={ssid}"
                )
                print(f"      -> {verdict(rssi)}")
                if best is None or rssi > best:
                    best = rssi
            except Exception as exc:  # 设备重启/移出范围时不要崩
                print(f"      !! 读取失败（设备移动中或断网）: {exc}")

            if seconds and (time.time() - start) >= seconds:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n已停止。")

    print("-" * 68)
    if best is not None:
        print(f"本次最强 RSSI: {best} dBm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
