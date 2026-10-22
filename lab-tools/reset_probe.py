#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32-C6 复位方式探测
================================================================
依次尝试几种 DTR/RTS 时序, 找出能把芯片从下载模式带回"正常启动"的那一种。
判据: 输出中出现 boot 日志且【不】出现 "waiting for download"。
"""

from __future__ import annotations

import sys
import time

import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/cu.usbmodem101"


def drain(s: serial.Serial, seconds: float) -> bytes:
    out = bytearray()
    end = time.time() + seconds
    while time.time() < end:
        chunk = s.read(4096)
        if chunk:
            out.extend(chunk)
    return bytes(out)


def attempt(s: serial.Serial, name: str, steps) -> tuple[bool, bytes]:
    print(f"\n=== 尝试序列: {name} ===")
    s.reset_input_buffer()
    for dtr, rts, delay in steps:
        s.dtr = dtr
        s.rts = rts
        time.sleep(delay)
    data = drain(s, 6)
    text = data.decode("utf-8", "replace")
    ok = ("boot:" in text) and ("waiting for download" not in text)
    print(text[:1200] if text else "  (无输出)")
    print(f"  -> {'✓ 正常启动' if ok else '✗ 仍为下载模式/无输出'}")
    return ok, data


def main() -> int:
    s = serial.Serial(PORT, 115200, timeout=0.2)
    try:
        sequences = [
            ("A: DTR 脉冲 (idle->assert->release)", [
                (False, False, 0.3), (True, False, 0.3), (False, False, 0.4)]),
            ("B: 释放 DTR 后保持", [
                (False, True, 0.2), (True, False, 0.2), (False, False, 0.5)]),
            ("C: 仅置 DTR=True 保持", [
                (False, False, 0.3), (True, True, 0.2), (True, False, 0.5)]),
            ("D: RTS 脉冲", [
                (True, False, 0.2), (True, True, 0.2), (True, False, 0.5)]),
        ]
        for name, steps in sequences:
            ok, _ = attempt(s, name, steps)
            if ok:
                print("\n[✓] 找到可用复位序列, 芯片已正常启动, 正在保持连接读取日志 ...")
                extra = drain(s, 8)
                print(extra.decode("utf-8", "replace")[:2000])
                return 0
        print("\n[x] 所有软件时序都没能带回正常启动。请手动按一下板上的 RESET 键。")
        return 1
    finally:
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
