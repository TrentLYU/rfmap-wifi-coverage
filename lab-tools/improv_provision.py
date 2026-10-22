#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improv Serial Wi-Fi 配网工具 (ESPectre Native 固件)
================================================================
严格按官方规范实现: https://www.improv-wifi.com/serial/

帧格式:
    "IMPROV" | version(1) | type(1) | length(1) | data(length) | checksum(1)
    checksum = sum(前面所有字节) & 0xFF

包类型:
    0x01 Current State  (设备 -> 客户端)
    0x02 Error State    (设备 -> 客户端)
    0x03 RPC Command    (客户端 -> 设备)
    0x04 RPC Result     (设备 -> 客户端)

RPC Command 数据布局:
    [command(1)] [data_len(1)] [data...]

SET_WIFI_SETTINGS(0x01) 的 data:
    [ssid_len(1)] [ssid...] [password_len(1)] [password...]

ESP32-C6 的主控制台在【原生 USB 口】(macOS: /dev/cu.usbmodem*), 不在 UART 口。

用法:
    python3 improv_provision.py --port /dev/cu.usbmodem101 --state
    python3 improv_provision.py --port /dev/cu.usbmodem101 --info
    python3 improv_provision.py --port /dev/cu.usbmodem101 --scan
    python3 improv_provision.py --port /dev/cu.usbmodem101 --ssid MyWiFi --password 'p@ss'
"""

from __future__ import annotations

import argparse
import os
import time

import serial

HEADER = b"IMPROV"
VERSION = 0x01

PKT_CURRENT_STATE = 0x01
PKT_ERROR_STATE = 0x02
PKT_RPC_COMMAND = 0x03
PKT_RPC_RESULT = 0x04

CMD_SET_WIFI_SETTINGS = 0x01
CMD_GET_CURRENT_STATE = 0x02
CMD_GET_DEVICE_INFO = 0x03
CMD_SCAN_WIFI_NETWORKS = 0x04
CMD_GET_HOSTNAME = 0x05
CMD_GET_DEVICE_NAME = 0x06
CMD_GET_NETWORK_STATE = 0x07

CMD_NAMES = {
    0x01: "SET_WIFI_SETTINGS",
    0x02: "GET_CURRENT_STATE",
    0x03: "GET_DEVICE_INFO",
    0x04: "SCAN_WIFI_NETWORKS",
    0x05: "GET_HOSTNAME",
    0x06: "GET_DEVICE_NAME",
    0x07: "GET_NETWORK_STATE",
}

CURRENT_STATES = {
    0x00: "Stopped (配网不可用, Wi-Fi 关闭)",
    0x02: "Ready / Authorized (就绪, 可以接收凭据)",
    0x03: "Provisioning (已收到凭据, 正在连接)",
    0x04: "Provisioned (连接成功)",
}

ERROR_STATES = {
    0x00: "No error",
    0x01: "Invalid RPC packet",
    0x02: "Unknown RPC command",
    0x03: "Unable to connect (连不上该 Wi-Fi)",
    0x05: "Bad Hostname",
    0xFF: "Unknown error",
}


def build_frame(pkt_type: int, data: bytes) -> bytes:
    body = HEADER + bytes([VERSION, pkt_type, len(data)]) + data
    return body + bytes([sum(body) & 0xFF])


def build_rpc(command: int, data: bytes = b"") -> bytes:
    """RPC Command 的 data = [command][data_len][data...]"""
    return build_frame(PKT_RPC_COMMAND, bytes([command, len(data)]) + data)


def open_port(port: str) -> serial.Serial:
    s = serial.Serial(port, 115200, timeout=0.2)
    s.dtr = True          # USB Serial/JTAG 需 DTR 置位才通信
    s.rts = False
    time.sleep(0.2)
    s.reset_input_buffer()
    return s


def parse_string_list(payload: bytes):
    """RPC Result 里剩余部分是 [len][str][len][str]... 的字符串列表。"""
    out, i = [], 0
    while i < len(payload):
        n = payload[i]
        i += 1
        out.append(payload[i:i + n].decode("utf-8", "replace"))
        i += n
    return out


def read_frames(s: serial.Serial, seconds: float):
    buf = bytearray()
    deadline = time.time() + seconds
    while time.time() < deadline:
        chunk = s.read(4096)
        if chunk:
            buf.extend(chunk)

    frames, i = [], 0
    while i + 9 <= len(buf):
        if bytes(buf[i:i + 6]) != HEADER:
            i += 1
            continue
        pkt_type, length = buf[i + 7], buf[i + 8]
        if i + 9 + length + 1 > len(buf):
            break
        data = bytes(buf[i + 9:i + 9 + length])
        checksum = buf[i + 9 + length]
        ok = checksum == (sum(buf[i:i + 9 + length]) & 0xFF)
        frames.append({"type": pkt_type, "data": data, "ok": ok})
        i += 9 + length + 1
    return frames, bytes(buf)


def describe(frames) -> None:
    for f in frames:
        t, d = f["type"], f["data"]
        if t == PKT_CURRENT_STATE and d:
            print(f"  [Current State] {CURRENT_STATES.get(d[0], hex(d[0]))}")
        elif t == PKT_ERROR_STATE and d:
            print(f"  [Error State] {ERROR_STATES.get(d[0], hex(d[0]))}")
        elif t == PKT_RPC_RESULT and len(d) >= 2:
            cmd = d[0]
            strings = parse_string_list(d[2:])   # 跳过 cmd 与 data_len
            print(f"  [RPC Result] {CMD_NAMES.get(cmd, hex(cmd))} -> {strings}")
        else:
            print(f"  [未知/空包] type={t} data={d!r}")
        if not f["ok"]:
            print("    (校验和不匹配)")


def send(s: serial.Serial, frame: bytes) -> None:
    s.write(frame)
    s.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--ssid")
    ap.add_argument("--password", default="")
    ap.add_argument("--state", action="store_true")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--wait", type=float, default=6.0)
    args = ap.parse_args()

    if not os.path.exists(args.port):
        print(f"[x] 串口不存在: {args.port}")
        return 2

    print(f"[i] 打开 {args.port} @115200 (DTR=1, Improv Serial)")
    s = open_port(args.port)
    try:
        if args.state or args.info or args.scan:
            cmd = (CMD_GET_CURRENT_STATE if args.state else
                   CMD_GET_DEVICE_INFO if args.info else
                   CMD_SCAN_WIFI_NETWORKS)
            print(f"[i] 发送 {CMD_NAMES[cmd]} ...")
            send(s, build_rpc(cmd))
            frames, raw = read_frames(s, args.wait)
            if not frames:
                print("[x] 未收到 Improv 响应。")
                if raw:
                    print(f"    (收到 {len(raw)} 字节非 Improv 数据)")
                return 1
            describe(frames)
            return 0

        if not args.ssid:
            print("[x] 需要 --ssid")
            return 2

        ssid_b, pwd_b = args.ssid.encode(), args.password.encode()
        data = bytes([len(ssid_b)]) + ssid_b + bytes([len(pwd_b)]) + pwd_b
        print(f"[i] SET_WIFI_SETTINGS  ssid={args.ssid!r}  密码长度={len(pwd_b)}")
        send(s, build_rpc(CMD_SET_WIFI_SETTINGS, data))

        frames, _ = read_frames(s, args.wait)
        describe(frames) if frames else print("[!] 无即时响应, 继续监听 ...")

        print(f"[i] 监听 {args.wait:.0f}s 观察连接状态 ...")
        frames, _ = read_frames(s, args.wait)
        describe(frames) if frames else print("[x] 没有状态更新。")

        print("\n[i] 出现 'Provisioned (连接成功)' 即配网成功, 之后运行:")
        print("    python3 lab-tools/find_espectre.py")
        return 0
    finally:
        s.close()


if __name__ == "__main__":
    raise SystemExit(main())
