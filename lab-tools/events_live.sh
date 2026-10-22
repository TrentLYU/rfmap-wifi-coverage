#!/usr/bin/env bash
# ESPectre 运动事件实时监听 (指南第 6 节)
# 用法: ./events_live.sh <ESP32_IP> [端口]
# 说明: 运行后保持窗口打开, 在路由器与 ESP32 之间走动,
#       终端应陆续出现含 state / score 字段的 motion 事件。
set -u

IP="${1:?用法: ./events_live.sh <ESP32_IP> [端口]}"
PORT="${2:-62587}"

echo "[i] 连接 http://${IP}:${PORT}/espectre/v1/events"
echo "[i] 走动几步观察 state / score 变化, Ctrl+C 退出"
echo "----------------------------------------------------------"

curl -N \
  -H "Origin: https://espectre.dev" \
  -H "Accept: */*" \
  --no-buffer \
  "http://${IP}:${PORT}/espectre/v1/events"
