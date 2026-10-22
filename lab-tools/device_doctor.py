#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ESP32 现场体检助手（离线可用，仅标准库）。

一条命令回答："我的 ESP32 现在到底走到哪一步了？"

体检三步：
    1) 找 ESP32 的原生 USB 串口 (/dev/cu.usbmodem*)；
    2) 有串口时读 Improv 配网状态（Ready / Provisioning / Provisioned / Stopped）；
    3) 扫局域网找 ESPectre 的 HTTP 服务（端口 62587）。

用法：
    python3 lab-tools/device_doctor.py             # 全量体检（含局域网扫描）
    python3 lab-tools/device_doctor.py --no-scan   # 只查串口 + 配网状态，最快
    python3 lab-tools/device_doctor.py --fast      # 局域网只扫本机 /24
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def find_ports():
    """ESP32-C6 的原生 USB 串口（不是 UART 口）。"""
    return sorted(glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/tty.usbmodem*"))


def run(cmd, timeout=180.0):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "<超时>"
    except Exception as exc:  # noqa: BLE001
        return 125, f"<无法执行: {exc}>"


def improv_state(port: str, wait: float = 5.0):
    """调 improvision_provision.py --state，返回 (状态文本, 原始输出)。"""
    code, out = run([sys.executable, os.path.join(HERE, "improv_provision.py"),
                     "--port", port, "--state", "--wait", str(wait)],
                    timeout=wait + 20)
    state = None
    for line in out.splitlines():
        if "[Current State]" in line:
            state = line.split("[Current State]", 1)[1].strip()
        if "[Error State]" in line and "No error" not in line:
            state = line.split("[Error State]", 1)[1].strip()
    return state, out


def main() -> int:
    ap = argparse.ArgumentParser(description="ESP32 现场体检")
    ap.add_argument("--no-scan", action="store_true", help="不扫局域网")
    ap.add_argument("--fast", action="store_true", help="局域网只扫本机 /24")
    args = ap.parse_args()

    ip = local_ip()
    print("=" * 62)
    print("  ESP32 现场体检")
    print("=" * 62)
    print(f"本机 IP: {ip}   (采集时 ESP32 必须与它在同一网段)")

    # ---- 1) 串口 ----
    print("\n[1/3] ESP32 原生 USB 串口")
    ports = find_ports()
    if not ports:
        print("  ✗ 没有发现 /dev/cu.usbmodem*")
        print("    → ESP32 没有用数据线接在电脑上（或没上电）")
        print("      注意插板上标 USB 的那个口，不是 UART 口")
    else:
        for p in ports:
            print(f"  ✓ {p}")

    # ---- 2) Improv 配网状态 ----
    print("\n[2/3] Improv 配网状态")
    state = None
    if not ports:
        print("  - 跳过（没有串口）")
    else:
        print(f"  读取 {ports[0]} ...")
        state, out = improv_state(ports[0])
        for line in out.splitlines():
            if any(k in line for k in ("[Current State]", "[Error State]",
                                       "[RPC Result]", "未收到")):
                print("  " + line.strip())
        if state is None:
            print("  ? 没读到状态（固件可能不是 Native/release，或需要按 RESET）")

    # ---- 3) 局域网扫描 ----
    found = None
    if args.no_scan:
        print("\n[3/3] 局域网扫描：已跳过 (--no-scan)")
    else:
        print("\n[3/3] 局域网扫描（端口 62587，可能要几十秒）")
        cmd = [sys.executable, os.path.join(HERE, "find_espectre.py")]
        if args.fast:
            cmd.append(ip + "/24")
        code, out = run(cmd, timeout=300)
        for line in out.splitlines():
            print("  " + line)
        if code == 0:
            for line in out.splitlines():
                if "★ ESPectre 设备" in line:
                    found = line.strip().split()[1].split(":")[0].lstrip("- ")

    # ---- 结论 ----
    print("\n" + "=" * 62)
    print("  结论与下一步")
    print("=" * 62)
    if found:
        print(f"  ✅ 设备在线：{found}")
        print("     直接开始采集（示例，改成你的房间尺寸）：")
        print(f"     python3 lab-tools/rssi_survey.py --ip {found} \\")
        print("         --map-id ROOM-A-dp1 --room ROOM-A --height 1.0 \\")
        print("         --grid 0,0,6,4,1 --seconds 20 --settle 3")
    elif ports and state and "Provisioned" in state:
        print("  ⚠️ 已配过网，但局域网里找不到它")
        print("     可能：连的是**别的**网络（比如旧的 192.168.x.x），或 IP 变了")
        print("     重配到当前 Wi-Fi：")
        print(f"     python3 lab-tools/improv_provision.py --port {ports[0]} \\")
        print("         --ssid '当前2.4G名称' --password '密码'")
    elif ports and state and "Ready" in state:
        print("  ⚠️ 设备就绪但还没配网")
        print("     先看有哪些网络：")
        print(f"     python3 lab-tools/improv_provision.py --port {ports[0]} --scan")
        print("     再配网：")
        print(f"     python3 lab-tools/improv_provision.py --port {ports[0]} \\")
        print("         --ssid '当前2.4G名称' --password '密码'")
    elif ports:
        print("  ⚠️ 串口在，但没读到明确的配网状态")
        print("     抓一段启动日志看看它连了什么：")
        print(f"     python3 lab-tools/usb_console.py {ports[0]} 12")
    else:
        print("  ✗ 设备既不在串口上，也不在局域网里")
        print("     1) 用数据线把 ESP32 接到电脑（板上标 USB 的口）")
        print("     2) 重新运行：python3 lab-tools/device_doctor.py")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
