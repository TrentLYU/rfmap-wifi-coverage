#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 串口体检工具 (macOS)
------------------------------------------------------------
用途:
    1. 列出所有 USB 串口设备
    2. 显示对应的 USB 桥接芯片 (CP2102N / CH34x / 原生 USB-JTAG)
    3. 检查串口是否被其他程序占用 (浏览器烧录时会占用)
    4. 可选: 读取数秒串口日志, 判断 ESPectre 是否已启动

用法:
    python3 check_serial.py                # 只做体检
    python3 check_serial.py -r 5           # 另外读取 5 秒串口日志
    python3 check_serial.py -d /dev/cu.usbserial-10 -r 5
"""

from __future__ import annotations

import argparse
import glob
import os
import select
import subprocess
import sys
import time


def list_ports():
    return sorted(glob.glob("/dev/cu.*"))


def usb_report() -> str:
    return subprocess.run(["system_profiler", "SPUSBDataType"],
                          capture_output=True, text=True).stdout


def chip_hint(report: str) -> list[str]:
    keys = ("CP210", "CH34", "CH910", "FTDI", "Silicon Labs",
            "Espressif", "USB JTAG", "USB Serial")
    lines = []
    for raw in report.splitlines():
        line = raw.strip()
        if any(k.lower() in line.lower() for k in keys):
            lines.append(line)
    return lines


def occupied(port: str) -> bool:
    out = subprocess.run(["lsof", port], capture_output=True, text=True).stdout
    rows = [r for r in out.splitlines()[1:] if r.strip()]
    return bool(rows)


def read_log(port: str, seconds: float) -> None:
    try:
        subprocess.run(["stty", "-f", port, "115200", "cs8", "-cstopb", "-parenb",
                        "-crtscts", "raw", "-echo"], check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or "串口被占用或参数不兼容"
        print(f"[x] 无法配置串口: {detail}")
        print("    如果浏览器(Edge)仍连着该串口, 请先在烧录页点 Disconnect 或关闭该标签页,")
        print("    再重新运行本脚本读取日志。")
        return
    fd = os.open(port, os.O_RDONLY | os.O_NONBLOCK | os.O_NOCTTY)
    print(f"[i] 监听 {port} {seconds}s (115200 8N1) ...")
    buf = bytearray()
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            r, _, _ = select.select([fd], [], [], 0.3)
            if not r:
                continue
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if chunk:
                buf.extend(chunk)
    finally:
        os.close(fd)

    if not buf:
        print("[x] 无串口输出。可能: 空固件 / 未运行 / 波特率不符 / 需按 RESET。")
        return
    text = buf.decode("utf-8", "replace")
    print(f"[+] 收到 {len(buf)} 字节:")
    print("-" * 50)
    print(text[:4000])
    print("-" * 50)
    low = text.lower()
    if "espectre" in low or "csi" in low or "wifi" in low or "ip:" in low:
        print("[✓] 串口日志出现 ESPectre / Wi-Fi / CSI 字样，固件很可能已运行。")
    else:
        print("[?] 有输出但未见 ESPectre 关键字，请人工查看上面的日志。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--device", help="指定串口, 如 /dev/cu.usbserial-10")
    ap.add_argument("-r", "--read", type=float, default=0.0,
                    help="读取串口日志的秒数, 0 表示不读")
    args = ap.parse_args()

    print("=" * 58)
    print(" ESP32 串口体检")
    print("=" * 58)

    ports = list_ports()
    print("\n[1] 可用串口设备:")
    if not ports:
        print("    (无) —— 请检查数据线/接口，优先插标 UART 的口")
    for p in ports:
        mark = ""
        if any(k in p for k in ("usbserial", "usbmodem", "ttyACM", "SLAB")):
            mark = "  <- 可能是 USB 串口桥 (ESP32)"
        print(f"    {p}{mark}")

    print("\n[2] USB 桥接芯片:")
    hints = chip_hint(usb_report())
    if hints:
        for h in hints:
            print(f"    {h}")
    else:
        print("    (未检测到常见 USB 串口芯片)")

    target = args.device
    if not target:
        cands = [p for p in ports if any(k in p for k in ("usbserial", "usbmodem", "ttyACM"))]
        target = cands[0] if cands else None

    if target:
        print(f"\n[3] 目标串口: {target}")
        if not os.path.exists(target):
            print("    [x] 设备节点不存在")
            return 1
        if occupied(target):
            print("    [!] 已被进程占用 —— 烧录前请关闭串口监视器等程序:")
            print(subprocess.run(["lsof", target], capture_output=True, text=True).stdout)
        else:
            print("    [✓] 未被占用, 浏览器可以打开该端口")
        if args.read > 0:
            print()
            read_log(target, args.read)
    else:
        print("\n[3] 未找到可用的 USB 串口设备")

    print("\n[提示] 正常情况下先插 UART 口; 若串口不识别或配网失败, 再换 USB 口。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
