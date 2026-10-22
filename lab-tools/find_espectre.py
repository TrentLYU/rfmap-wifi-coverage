#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESPectre 局域网设备发现工具
------------------------------------------------------------
用途:
    在电脑所在网段里扫描 ESPectre 的本地 HTTP 服务端口(默认 62587),
    找到 ESP32 被路由器分配的局域网 IP 地址, 并验证接口是否可用。

用法:
    python3 find_espectre.py                 # 自动使用本机 /24 网段
    python3 find_espectre.py 192.168.31.0/24 # 指定网段
    python3 find_espectre.py 192.168.31.50   # 只探测单个 IP

注意:
    - ESPectre 本地接口要求带 Origin 请求头, 否则会被拒绝。
    - /events 是流式接口, 这里只读一小段用于验证连通性。
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

PORT = 62587
ORIGIN = "https://espectre.dev"
CONNECT_TIMEOUT = 0.6
HTTP_TIMEOUT = 2.0

# 掩码比 /20 更宽时（地址数超过这个值）不再全扫，回退到本机 /24，避免扫太久
MAX_SCAN = 4096


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        out = subprocess.run(["ipconfig", "getifaddr", "en0"],
                             capture_output=True, text=True).stdout.strip()
        return out or "127.0.0.1"
    finally:
        s.close()


def port_open(ip: str) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(CONNECT_TIMEOUT)
        return s.connect_ex((ip, PORT)) == 0


def http_probe(ip: str, path: str = "/espectre/v1/events", nbytes: int = 600) -> str:
    try:
        with socket.create_connection((ip, PORT), timeout=HTTP_TIMEOUT) as s:
            req = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {ip}:{PORT}\r\n"
                f"Origin: {ORIGIN}\r\n"
                f"Accept: */*\r\n"
                f"Connection: close\r\n\r\n"
            )
            s.sendall(req.encode())
            data = b""
            while len(data) < nbytes:
                try:
                    chunk = s.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                data += chunk
            return data.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def local_mask() -> str | None:
    """取本机 Wi-Fi 接口的真实子网掩码（硬编码 /24 在 /20 之类网络下会漏扫）。"""
    for iface in ("en0", "en1"):
        try:
            out = subprocess.run(["ipconfig", "getoption", iface, "subnet_mask"],
                                 capture_output=True, text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            continue
        if out.count(".") == 3 and out != "0.0.0.0":
            return out
    return None


def build_targets(arg: str | None):
    if arg:
        if "/" in arg:
            return [str(h) for h in ipaddress.ip_network(arg, strict=False).hosts()]
        return [arg]
    ip = local_ip()
    mask = local_mask()
    net = ipaddress.ip_network(f"{ip}/{mask}", strict=False) if mask \
        else ipaddress.ip_network(ip + "/24", strict=False)
    hosts = list(net.hosts())
    if len(hosts) > MAX_SCAN:
        net24 = ipaddress.ip_network(ip + "/24", strict=False)
        print(f"[!] 本机网段 {net} 含 {len(hosts)} 个地址, 超过上限 {MAX_SCAN}")
        print(f"    本次只扫 {net24}; 要全扫请显式指定网段:")
        print(f"    python3 {os.path.basename(sys.argv[0])} {net}")
        hosts = list(net24.hosts())
    return [str(h) for h in hosts]


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    targets = build_targets(arg)
    my_ip = local_ip()
    print(f"[i] 本机 IP: {my_ip}   端口: {PORT}   待扫描主机数: {len(targets)}")
    print("[i] 正在并发扫描, 请稍候 ...")

    found = []
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(port_open, ip): ip for ip in targets}
        for fut in as_completed(futures):
            ip = futures[fut]
            try:
                if fut.result():
                    found.append(ip)
            except Exception:
                pass

    if not found:
        print("[x] 未发现开放 62587 端口的设备。")
        print("    可能原因: ESP32 还没配网 / 不在同一网段 / AP 隔离开启。")
        print("    可让路由器管理页的\"已连接设备\"列表帮忙确认。")
        return 1

    print(f"[+] 发现 {len(found)} 个候选设备, 正在验证 ESPectre 接口 ...\n")
    for ip in sorted(found, key=lambda x: tuple(int(p) for p in x.split("."))):
        resp = http_probe(ip)
        head = resp.splitlines()
        status = head[0] if head else "<无响应>"
        looks_like_espectre = ("espectre" in resp.lower()
                               or "event" in resp.lower()
                               or "motion" in resp.lower()
                               or "score" in resp.lower())
        tag = "★ ESPectre 设备" if looks_like_espectre else "? 其他服务"
        print(f"--- {ip}:{PORT}  [{tag}]")
        print(f"    HTTP 状态行: {status}")
        body = resp.split("\r\n\r\n", 1)[-1].strip()
        if body:
            print(f"    响应片段: {body[:300]}")
        print()

    print("[i] 下一步: 用发现的 IP 运行 events 监听:")
    print("    ./events_live.sh <IP>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
