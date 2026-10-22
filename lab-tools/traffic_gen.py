#!/usr/bin/env python3
"""
traffic_gen.py —— 在局域网内自造 Wi-Fi 流量（不需要外网）

用途：Wi-Fi 没有互联网时，设备自身的 ping / DNS 激励会失效。
      把设备的 traffic_generator_mode 设为 external，再用本脚本从电脑持续发包，
      让 ESP32 一直有帧可测（CSI 需要"有流量"才有数据）。

用法：
    # 让设备有流量可测：向设备 UDP 5555 端口持续发 60 秒
    python3 lab-tools/traffic_gen.py --target 192.168.31.66 --duration 60

    # 组播方式（设备默认组播组 239.255.0.1）
    python3 lab-tools/traffic_gen.py --target 192.168.31.66 --group 239.255.0.1 --duration 60

    # 反向排查：如果设备是"往外发"的，用监听模式看有没有包进来
    python3 lab-tools/traffic_gen.py --listen --duration 10

配合：先把设备切到 external 模式
    curl -X PATCH -H "Origin: https://espectre.dev" -H "Content-Type: application/json" \
      -d '{"traffic_generator_mode":"external"}' http://<IP>:62587/espectre/v1/sensing
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

DEFAULT_PORT = 5555
DEFAULT_GROUP = "239.255.0.1"


def send_loop(target: str, port: int, pps: int, duration: float, size: int, group: str) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    if group:
        dest = group
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
    else:
        dest = target
    # ⚠️ 官方硬性要求：external 模式下 UDP 标记必须正好是这 4 个字节
    #    = "👻".encode("utf-8") = F0 9F 91 BB，多一字节少一字节都不产生 CSI
    payload = MARKER
    if size != 4:
        print("!! 提示：external 模式的 UDP 标记必须正好 4 字节（👻），已忽略 --size %d" % size)
    interval = 1.0 / max(1, pps)
    sent = 0
    t0 = time.time()
    print("开始向 %s:%d 发包（%d pps, %d 字节/包, %.0f 秒）..."
          % (dest, port, pps, size, duration or 0))
    try:
        while True:
            now = time.time()
            if duration and now - t0 >= duration:
                break
            sock.sendto(payload, (dest, port))
            sent += 1
            if sent % (pps * 5) == 0:            # 每 5 秒报一次，方便和 Diagnostics 对表
                print("  已发 %d 包  已过 %.0f 秒  实际 %.0f pps"
                      % (sent, now - t0, sent / max(0.001, now - t0)))
            nxt = t0 + sent * interval
            sleep = nxt - time.time()
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\n用户中断")
    print("结束：共发出 %d 个包，用时 %.1f 秒" % (sent, time.time() - t0))
    return sent


def listen_loop(port: int, duration: float, group: str) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", port))
    except OSError as exc:
        print("!! 绑定 %d 端口失败: %s" % (port, exc))
        return -1
    if group:
        try:
            mreq = socket.inet_aton(group) + socket.inet_aton("0.0.0.0")
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            print("已加入组播组 %s" % group)
        except OSError as exc:
            print("!! 加入组播失败（忽略）: %s" % exc)
    sock.settimeout(1.0)
    print("监听 UDP %d 端口，%.0f 秒..." % (port, duration or 0))
    got = 0
    t0 = time.time()
    while not duration or time.time() - t0 < duration:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue
        got += 1
        if got <= 5 or got % 100 == 0:
            print("  收到 %d 字节  来自 %s:%d  (第 %d 包)" % (len(data), addr[0], addr[1], got))
    print("结束：共收到 %d 个包" % got)
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description="局域网内自造 Wi-Fi 流量（无需外网）")
    ap.add_argument("--target", help="设备 IP，例如 192.168.31.66")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="UDP 端口，默认 %d" % DEFAULT_PORT)
    ap.add_argument("--pps", type=int, default=100, help="每秒包数，默认 100")
    ap.add_argument("--duration", type=float, default=60, help="持续秒数，默认 60")
    ap.add_argument("--size", type=int, default=1400, help="每包字节数，默认 1400")
    ap.add_argument("--group", nargs="?", const=DEFAULT_GROUP, default=None,
                    help="改用组播发送（默认组 %s）" % DEFAULT_GROUP)
    ap.add_argument("--listen", action="store_true", help="改为监听模式（排查反向流量）")
    args = ap.parse_args()

    if args.listen:
        return 0 if listen_loop(args.port, args.duration, args.group) >= 0 else 1
    if not args.target:
        ap.error("发送模式必须给 --target <设备IP>")
    sent = send_loop(args.target, args.port, args.pps, args.duration, args.size, args.group)
    print("\n提示：发包期间去看设备判据 —— "
          "curl -H 'Origin: https://espectre.dev' http://%s:62587/espectre/v1/diagnostics"
          % args.target)
    print("      要求 Traffic TX/RX 与 CSI callbacks 持续增长；若 callbacks 仍为 0，"
          "改用 --group 组播方式再试一次。")
    return 0 if sent > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
