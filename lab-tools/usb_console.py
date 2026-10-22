#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32-C6 原生 USB 控制台诊断
================================================================
作用:
    1. 打开原生 USB Serial/JTAG 串口 (usbmodem*)
    2. 用 esptool 风格的 DTR/RTS 时序复位芯片
    3. 抓取启动日志与运行日志
    4. 顺便探测 Improv Serial 是否存在

用法:
    python3 usb_console.py /dev/cu.usbmodem101 [读取秒数]
"""

from __future__ import annotations

import sys
import time

import serial


def usb_reset(s: serial.Serial) -> None:
    """ESP32(-S3/-C3/-C6) 原生 USB Serial/JTAG 的复位时序。"""
    s.dtr, s.rts = False, False
    time.sleep(0.2)
    s.dtr, s.rts = True, True
    time.sleep(0.1)
    s.dtr, s.rts = False, True
    time.sleep(0.1)
    s.dtr, s.rts = True, False
    time.sleep(0.1)
    s.dtr, s.rts = False, False
    time.sleep(0.3)
    s.dtr = True          # 保持 DTR, 让设备持续发送


def main() -> int:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem101"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    print(f"[i] 打开 {port} @115200")
    s = serial.Serial(port, 115200, timeout=0.3)
    try:
        print("[i] 执行 USB 复位时序 (触发芯片重启) ...")
        usb_reset(s)
        s.reset_input_buffer()

        print(f"[i] 抓取 {seconds:.0f} 秒日志:")
        print("-" * 56)
        buf = bytearray()
        deadline = time.time() + seconds
        while time.time() < deadline:
            chunk = s.read(4096)
            if chunk:
                buf.extend(chunk)
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
        print()
        print("-" * 56)

        if not buf:
            print("[x] 完全没有输出。")
            print("    可能: 固件未运行 / console 不在该口 / 需要按板上 RESET 键。")
            return 1

        text = buf.decode("utf-8", "replace")
        low = text.lower()
        print(f"[+] 收到 {len(buf)} 字节")
        for key, label in [("espectre", "ESPectre"), ("csi", "CSI"),
                           ("improv", "Improv"), ("wifi", "Wi-Fi"),
                           ("ip:", "IP 地址"), ("rst:", "复位原因")]:
            if key in low:
                print(f"    ✓ 日志中出现 {label}")
        return 0
    finally:
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
