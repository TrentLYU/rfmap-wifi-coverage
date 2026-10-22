# ESP32-C6 无线 CSI 采集 · 第一节课操作手册

> 依据：`ESP32_C6_ESPectre_CSI_Student_Guide.pdf`（本科生软件工程项目 · 两个小组共用的第一阶段操作指南）
> 目标一句话：**让 ESP32-C6 连上 2.4 GHz 路由器，把这条链路上的 CSI 读出来；人在无线链路中走动时 CSI 出现波动，即成功。**
> 配套文档：**`docs/ESPectre-纯浏览器操作指南.md`** —— 只想用官方网页（不敲命令）完成实验时，直接看那份。
> 无外网场景：**`docs/ESP32-C6-CSI-无外网完整实验流程.md`** —— Wi-Fi 没有互联网时的完整可执行流程（网页工具不可用，改走 curl + `lab-tools/`）。

---

## 一、这节课到底在做什么

我们不做"从零写 Wi-Fi 驱动"这种事。整条数据通路是：

```
路由器(2.4GHz AP)  ──无线──▶  ESP32-C6 (跑 ESPectre 固件)
                                    │  ① 接收无线帧
                                    │  ② 芯片驱动提交 CSI 给 ESPectre
                                    │  ③ 板上做运动检测 / 或按需输出原始 CSI
                                    ▼
                              同一路由器局域网
                                    │  HTTP :62587
                                    ▼
                          电脑浏览器(Edge/Chrome) / curl
                          (Live motion / CSI Visualizer / 命令行采集)
```

三个角色的分工，先记牢：

| 角色 | 是什么 | 负责什么 |
|---|---|---|
| **ESP32-C6-DevKitC-1-N8** | 采集设备（硬件） | 作为普通 Wi-Fi 设备连上路由器，接收无线帧并做信道测量 |
| **ESPectre** | 跑在 ESP32 上的**开源固件** | 连 Wi-Fi、采集 CSI、做运动检测、在局域网开 HTTP 接口 |
| **电脑** | 查看 / 保存工具 | 用官方网页看波形和运动分数，或用命令把数据存到硬盘 |

**关键纠正（考试/验收常见误解）**：CSI 不是"信号强度"一个数字，也不是"扫描整屋 Wi-Fi 得到的 CSI"。它测的是**路由器 ↔ 这一个 ESP32 之间那条特定链路**的信道状态。

---

## 二、核心原理：CSI 到底是什么

- Wi-Fi 的主要用途是传数据，但无线电波从路由器到 ESP32 的途中会被**距离、墙壁、人体运动**影响。
- CSI（Channel State Information，信道状态信息）是 ESP32 从**实际收到的 Wi-Fi 数据包**里提取的一组**信道测量值**：
  - 不是单一 RSSI，而是**多个子载波（subcarrier）的复数测量**（幅度 + 相位 / I Q）。
  - 因此它对传播过程的变化**非常敏感**。
- 人体在链路中走动 → 多径被扰动 → **CSI 波动** → 运动分数（movement score）上升，状态在 `motion` / `idle` 之间切换。

### 为什么"连上 Wi-Fi"还不够

无线芯片**必须真的收到数据包**，才有机会去测量这些包经过信道后的变化。所以 ESPectre 默认会**定期向网关发 ping**，网关回复 → ESP32 收到回复帧 → 驱动把对应 CSI 交给 ESPectre。

> 部分路由器不响应默认 ping，此时 CSI 可能一直为空 —— 这是本实验最常见的坑之一，见第六节排障。

同一个 ESP32 同时干两件事：**无线电部分**测量收到的帧；**网络部分**再借同一个路由器，把结果传给电脑。

---

## 三、你的实验环境（本机实测结果）

已在本机实际检测并**全程跑通**，结果如下：

| 检查项 | 实测结果 | 结论 |
|---|---|---|
| **UART 口** `/dev/cu.usbserial-10` | CP2102N 串口桥；读 8 秒**无任何输出**，Improv 也不响应 | ❌ **不是主控制台**，日志与配网都不在这里 |
| **USB 口** `/dev/cu.usbmodem101` | **USB JTAG/serial debug unit**；复位后完整输出 ESP-IDF 启动日志 | ✅ **主控制台 + Improv Serial 都在这个口** |
| 固件 | **ESPectre Native 3.0.0**，ESP-IDF v5.5.5，`esp32c6`，设备名 `ESPectre C6 15c7be` | ✅ 烧录正确 |
| 配网方式 | **Improv Serial**（见第 3 步），已用命令行完成 | ✅ 全程无需浏览器 |
| 配网结果 | **Provisioned（连接成功）**，DHCP 分配 IP **`192.168.31.66`** | ✅ |
| 运动检测 | `/espectre/v1/events` 实时输出：走动 `score≈0.9999`，静止 `score≤0.4950` | ✅ **课程成功判据达成** |
| 系统浏览器 | **Chrome 154.0.8037.92** ✅ 推荐使用；另有 Edge 155.0.4283.45（**仅够用来烧录**） | ⚠️ 官方要求 **Chrome 151+**；**Device settings / Live motion 在 Edge 里可能不工作**，务必用 Chrome |
| 命令行工具 | curl 8.7.1 / git 2.54.0 / Python 3.9.6 + **pyserial 3.5** | ✅ |
| 本机网络 | IP `192.168.31.150`，网关 `192.168.31.1`，网段 `192.168.31.0/24` | ✅ 与设备同网段、可互访 |
| 路由器 | 小米路由器（`http://192.168.31.1/`），SSID 与 Mac 当前所连相同 | ✅ 设备与电脑同网段 |

> 🔑 **最重要的实测结论**：ESP32-C6-DevKitC-1 上，**ESPectre 的主串口控制台在标 `USB` 的那个口（原生 USB Serial/JTAG）**，**不在**标 `UART` 的口。
> 烧录两个口都能做，但**读日志、Improv Serial 配网必须在 `USB` 口**。
> 这正是"烧录成功、配网却始终不成"的真正根因 —— 修正后立刻一次成功。

---

## 四、分步操作讲解

### 第 0 步 · 先弄清概念

见第一、二节。一句话：**ESP32 是采集设备，ESPectre 是跑在 ESP32 上的软件，电脑是查看和保存结果的工具。**

---

### 第 1 步 · 材料、接线与网络准备

**材料清单**
- ESP32-C6-DevKitC-1-N8 开发板
- 带电源的普通 Wi-Fi 路由器
- **一根 USB-A 转 Type-C 数据线**（必须是数据线，不是只能充电的线）
- 一台电脑

**接线要点（本次实测已证实，这是最容易踩的坑）**
- 板上有两个 Type-C 口：标 **UART**（内部走 CP2102N 转串口电路）和标 **USB**（ESP32-C6 原生 USB Serial/JTAG）。
- **烧录**：两个口都能烧。
- **读日志 / Improv 配网**：**必须插 `USB` 口**。实测插 `UART` 口时串口完全无输出、Improv 无响应，配网必然失败。
- **建议**：全程就插 `USB` 口，烧录、看日志、配网一步到位。
- 你的实际情况：已从 `UART`（`/dev/cu.usbserial-10`）换成 `USB`（`/dev/cu.usbmodem101`），之后一切正常。

**路由器要求**
1. 开启**独立的 2.4 GHz** Wi-Fi（ESP32-C6 **只支持 2.4 GHz**）。
2. SSID 例如 `NovaLab-Test`，**WPA2** 密码。
3. **关闭"客户端隔离 / AP 隔离"**（否则电脑和 ESP32 互相不可达）。
4. **确保 DHCP 开启**。
5. 路由器不要求联外网，但**第一次下载固件时电脑要能上互联网**。
6. 电脑与 ESP32 最终必须在**同一个可互相访问的局域网**里。

---

### 第 2 步 · 第一次把 ESPectre 烧录到 ESP32（只需初次做）

**准备**
- 电脑上用**桌面版 Chrome 151+**（你的 **Chrome 154** 满足；**不要用 Edge** —— 它只能烧录，Device settings 与 Live motion 可能不工作）。
- ESP32 用 USB 线连电脑，确认**电源灯点亮**。
- **关闭 Arduino IDE 串口监视器和一切占用串口的软件**，否则浏览器打不开设备端口。

**操作顺序（务必按此顺序）**
1. 打开官方安装地址：`https://espectre.dev/tools/flash/`
2. 点击 **Connect USB device**。
3. 在弹出的设备列表里选**刚插入开发板后新增的那个串口**：
   - Windows：`COMx`
   - macOS / Linux：`usbserial`、`usbmodem`、`ttyACM`
   - **你应选 `/dev/cu.usbserial-10`**
   - ⚠️ 这一步是选**电脑↔ESP32 的 USB 通信口**，不是"选择附近的 Wi-Fi"。
4. 安装器识别芯片后，选择适用于 **ESP32-C6 的 Native 固件** + **稳定发布通道（release）**。
5. 按页面提示 **Install / Flash**。
6. 若页面提示确认**擦除旧固件**，确认即可（**会清除板上原有程序**）。
7. **保持 USB 连接**，直到页面报告安装完成、板子重启。

> **说明**：官方安装器界面随版本更新可能调整按钮名称，但基本顺序永远是：
> **选 USB 串口 → 确认芯片 → 选 Native → 烧录 → 重启**。
> 如果没有串口：先换数据线或 USB 口；仍失败时按住 **BOOT**、接线/松开 BOOT 后重试（进入下载模式）。

**无法自动化的部分**：第 3 步"选择串口"必须由你在 Chrome 里**亲自点击授权**——WebSerial 是浏览器安全机制，要求真实用户手势，任何脚本都无法替代。这一步之后的板子状态，我可以帮你验证。

---

### 第 3 步 · 给 ESP32 配 Wi-Fi，让电脑能找到它

1. 首次安装完成后，安装器通常**直接显示 Wi-Fi 配置表单**。
2. 输入你的 **2.4 GHz SSID 和密码**，保存，等待 ESP32 加入网络。
3. 如果板上已存了旧网络：在 USB 安装器里用 **Configure Wi-Fi** 重新设置。以后换路由器或改密码都从这里配网，**不必重新烧录整套固件**。
4. 连接成功后，路由器通过 **DHCP** 给 ESP32 分配一个局域网 IP，例如 `192.168.1.50`。
   - **这个 IP 非常重要**：电脑要靠它找到 ESP32。
   - 查看途径：安装器结果页；或路由器管理页的"已连接设备 / DHCP 客户端"列表。
   - **你的网段是 `192.168.31.0/24`**，所以 ESP32 大概率拿到类似 `192.168.31.x` 的地址。
5. 让**电脑也连到同一个路由器**，然后进入 `https://espectre.dev/tools/device-settings/`，使用**自动发现**或**手工填入刚才的 IP**。
6. 此时 USB 线**不再是通信必需品**：ESP32 可以改用 USB 充电器供电，电脑仍能通过路由器访问它。

> 本次实测中，**IP 由配网结果直接返回**，无需翻路由器后台：
> 配网成功时设备返回 `https://espectre.dev/tools/device-settings/?target=192.168.31.66`，末尾的 `192.168.31.66` 就是设备 IP。
> 也可以用 `lab-tools/find_espectre.py` 自动扫描整个网段找出它。

#### 方式 B · 命令行 Improv 配网（本次实际采用，最稳）

ESPectre Native 的配网走 **Improv Serial** 协议，直接跑在 `USB` 口的主控制台上，**可以完全用命令行完成，不必打开浏览器**。本项目已封装好工具：

```bash
# 0) 安装依赖（只需一次）
python3 -m pip install --user pyserial

# 1) 确认设备在线且处于就绪状态
python3 lab-tools/improv_provision.py --port /dev/cu.usbmodem101 --state
#   -> [Current State] Ready / Authorized (就绪, 可以接收凭据)

# 2) 查看设备信息（确认烧的确实是 ESPectre Native）
python3 lab-tools/improv_provision.py --port /dev/cu.usbmodem101 --info
#   -> ['ESPectre Native', '3.0.0', 'esp32c6', 'ESPectre C6 15c7be', 'ESP-IDF', 'v5.5.5']

# 3) 正式配网（把 SSID / 密码换成你自己的）
python3 lab-tools/improv_provision.py --port /dev/cu.usbmodem101 \
    --ssid "你的2.4G名称" --password "你的密码"
#   -> [Current State] Provisioning (正在连接)
#   -> [Current State] Provisioned (连接成功)
#   -> [RPC Result] ... -> ['https://espectre.dev/tools/device-settings/?target=192.168.31.66']
```

**Improv `Current State` 含义速查**

| 值 | 含义 |
|---|---|
| `Stopped` | 配网不可用（Wi-Fi 未开） |
| `Ready / Authorized` | **就绪，可以接收凭据** |
| `Provisioning` | 已收到凭据，正在连接 |
| `Provisioned` | **连接成功** |

**失败常见返回**：`Unable to connect` = SSID 或密码不对。
（本次第一次就是**密码填错**，换成正确密码后**一次成功**。）

> 换路由器或改密码时，直接重跑第 3 条命令即可，**不必重新烧录**。
> 若怎么都配不上：按住板上 **BOOT** 键 **3 秒**（ESP32-C6 的 BOOT 是 GPIO9），清除已存凭据后回到就绪状态，再重试。

---

### 第 4 步 · CSI 到底是怎么读出来的（原理，理解即可）

1. ESP32 先以**普通 Wi-Fi 设备**身份加入路由器。
2. 仅仅连上 Wi-Fi **不代表一直有足够的 CSI**——芯片需要**接收到实际的数据包**才有机会测量。
3. ESPectre 默认**定期向网关发 ping**；网关回复后，ESP32 的 Wi-Fi 芯片收到无线帧，驱动把对应 **CSI 提交给 ESPectre**。
4. ESPectre 再把 CSI 用于**运动检测**，或在电脑请求时作为**原始数据**发送。

注意：CSI 测量的是**特定收发链路**的信道，不是扫描房间里所有 Wi-Fi 的 CSI。**第一次实验尽量固定路由器和 ESP32 的位置。**

---

### 第 4.5 步 · ⭐ 实验成败的关键：设备摆放与走位（本次实测总结）

**这一节比任何参数设置都重要。** 下面是本次实验反复踩坑后才总结出来的经验，手册原文没有写。

#### ① 摆放：两个设备必须拉开距离

| 项目 | ❌ 错误（本次一开始就是） | ✅ 正确 |
|---|---|---|
| ESP32 ↔ 路由器距离 | 同桌（< 1 米） | **3 ~ 5 米** |
| RSSI | **-22 dBm**（过强） | **-50 ~ -70 dBm** |
| 中间 | 人走不进去 | **留 1~2 米人可横穿的通道** |

> 🔴 **设备挨在一起时，无线路径只有几十厘米，人根本无法进入这条路径**，CSI 就测不到任何变化。
> **这是 CSI 实验最常见的失败原因**，而且现象很迷惑：设备显示一切正常、信号满格，就是测不到运动。

用配套工具实时观察 RSSI，把板子移到目标区间：

```bash
python3 lab-tools/link_monitor.py 192.168.31.66
# 目标：显示 "合适 -> 运动检测最灵敏, 可以开始实验"
```

#### ② 走位：必须"穿过"两个设备之间的连线

本次在同一房间、同一时刻的实测对比：

| 动作 | motion 占比 | 平均 score |
|---|---|---|
| 静止不动 | 49.0% | 0.5626 |
| 在旁边走动（**路径偏了**） | 59.0% | 0.6617 |
| **站在两者连线正中** | **100.0%** | **1.0000** |
| **沿连线来回走动** | **84.6%** | **0.8828** |

→ 在旁边走动几乎无效；**必须让身体反复横穿两个设备之间那条看不见的直线**。
→ 最简单的做法：想象两者之间拉着一根绳子，你在中间约 1 米范围内**反复跨过这根绳子**。

#### ③ 黄金诊断法：遮挡测试

分不清"是设备坏了"还是"走位不对"时，做这个测试：

> **站到 ESP32 和路由器连线的正中间，站住不动 15 秒。**

| 结果 | 结论 |
|---|---|
| score 冲到 **1.0** | 链路完全正常，问题一定在**走位** |
| score 没反应 | 才是**摆放位置或环境**问题 |

本次遮挡测试得到 **70/70 全部 motion、score 恒定 1.0000** —— 一次就定位了根因。

#### ④ 环境噪声是"时变"的，别跟它较劲

同一条链路，本底噪声会突然从 `0.0001` 跳到 `0.9999`。风扇、空调气流、其他人在场都会造成。

当本底饱和到 1.0 时，**信噪比 = 1，任何阈值和 `hits` 参数都救不回来** ——
只能物理降噪（关风扇/清场）或者换时间再测。

> ⚠️ **实测过的无效手段**：把 `motion_on_hits` 从 4 调到 12，误报率 5.9%；而 `hits=4` 对照组是 **3.4%**。
> **时间滤波对本底噪声完全无效** —— 它只能滤掉偶发尖峰，滤不掉持续噪声。别在这上面浪费时间。

---

### 第 5 步 · 先看运动结果，再看原始 CSI 波形

**第一步：验证运动检测（Live motion）**
1. 打开 `https://espectre.dev/tools/monitor/?target=192.168.31.66`（把末尾 IP 换成你实际拿到的）。
2. 等待**自动校准**；用默认 **Lightweight 算法**时，房间尽量保持安静约 **10 秒**。
3. 之后让同学在**路由器与 ESP32 的传播区域内走动**。
4. 应当看到：
   - **movement score（运动分数）变化**
   - **motion / idle 状态切换**

**本次实测证据**（设备 `/espectre/v1/events` 实时流）：

| 场景 | 状态 | score 范围 |
|---|---|---|
| 人在链路中扰动（站在正中 / 沿连线走） | `motion` | **0.88 ~ 1.0000** |
| 静止、且环境安静 | `idle` | **0.0018 ~ 0.4950** |
| 静止、但环境噪声爆发（风扇/气流/他人） | `motion` | 0.8596 ~ 1.0000 (**信噪比=1，属环境问题**) |

→ **"人走动 → CSI 波动 → 检测到 motion"闭环成立，课程成功判据达成。**
→ **摆放距离与走位要求见 第 4.5 步**（这是最容易踩的坑）；环境噪声导致的误报见该节 ④。

> 这个网页只是官方现成的查看工具，**不需要大家开发网页**。

**第二步：验证原始 CSI（CSI Visualizer）**
1. 打开 `https://espectre.dev/tools/csi-visualizer/?target=192.168.31.66`（同样换成你的 IP）
2. 选同一个 ESP32，点击 **Start**。
3. 固件与流量条件正常时，浏览器会显示：
   - **子载波幅度、相位或 I/Q** 随时间变化的图
   - **packets/s、RSSI** 等统计信息
4. 记住：**真正的 CSI 不是浏览器生成的**——它是在 **ESP32 上测到**、通过 **Wi-Fi 传到电脑**、再由**浏览器画出来**的。

> ⚠️ **两种模式不要当成并行输出**：原始 CSI 流运行时，ESPectre 会**暂时停止输出普通运动检测结果**；关闭 CSI 流后才会恢复。

**如果监视器完全没有结果**：先打开 Live motion 页面底部的 **Diagnostics**：
- `Traffic TX/RX` = 制造和收到的流量
- `CSI callbacks/accepted` = 硬件回调和可用测量是否存在

先排查这里，再判断算法。部分路由器不响应默认 ping；**ESP32-C6 不支持该项目的 Wi-Fi raw 备用流量方式**，此时可在 **Advanced sensing settings** 中尝试 **DNS (UDP)** 或 **DNS (TCP)** 作为流量来源。

---

### 第 6 步 · 电脑怎样真正接收数据（命令行铁证）

ESPectre Native 在 ESP32 上提供一个**局域网 HTTP 服务（端口 `62587`）**。电脑连到相同 Wi-Fi 后，访问这个地址获取的数据**来自 ESP32 本机**。接口分两种：

| 接口 | 内容 | 说明 |
|---|---|---|
| `/espectre/v1/health` | 存活 / 在线 / 运行时长 | 测连通性最快的方法 |
| `/espectre/v1/device` | 设备信息（型号 / 固件 / 芯片） | 确认烧的确实是 ESPectre |
| `/espectre/v1/sensing` | 当前检测配置（算法 / 阈值 / 流量模式） | **排障最有用** |
| `/espectre/v1/wifi` | 网络状态（SSID / IP / 信道 / RSSI） | 确认已联网、看信号强度 |
| `/espectre/v1/events` | **文字**形式的运动事件（状态 + 分数） | 可直接读，SSE 流 |
| `/espectre/v1/csi` | **二进制**原始 CSI 流 | 需按 v8 格式解码 / 用 Visualizer |

> ⚠️ **注意**：是先有 `/espectre/v1` 这个契约前缀的。旧资料里写的 `/csi`、`/events` **在本固件上都会 404**（实测确认），必须带 `/espectre/v1`。
> 所有接口都要求请求头 `Origin: https://espectre.dev`。

**证明"电脑确实通过网络收到 ESP32 的运动结果"**

macOS / Linux（你的环境，已实测可用的完整命令）：

```bash
curl -N -H "Origin: https://espectre.dev" \
  http://192.168.31.66:62587/espectre/v1/events
```

本次实际收到的输出（片段）：

```
retry: 3000
: connected

event: motion
data: {"timestamp_ms":591105,"state":"motion","score":0.999876}

event: motion
data: {"timestamp_ms":591362,"state":"motion","score":0.999999}
```

Windows 命令提示符（把 `curl.exe` 保留）：

```bat
curl.exe -N -H "Origin: https://espectre.dev" http://192.168.1.50:62587/espectre/v1/events
```

- 命令运行后，**保持窗口打开、走动几步**，终端应陆续出现包含 `state`、`score` 等字段的 **motion 事件**。
- ⚠️ 这是 **ESP32 在板上计算好的检测结果，不能称作"原始 CSI"**。
- ⚠️ **此处必须带 `Origin` 请求头**，这是 ESPectre 本地 HTTP 接口的访问要求。

**设备当前真实状态快照（本次实测抓取，可作对照排障的"标准答卷"）**

```
/espectre/v1/health   -> {"status":"ok","online":true,"uptime_s":743}
/espectre/v1/device   -> firmware 3.0.0, chip esp32c6, frontend "native", csi_profile "ht20"
/espectre/v1/wifi     -> connected:true, ssid:"<你的2.4G名称，已隐去>",
                         ip:192.168.31.66, band:"2g", channel:11, rssi_dbm:-22
/espectre/v1/sensing  -> enabled:true, ready:true, calibrating:false, detector:"lightweight",
                         threshold:0.662185, motion_on_hits:4, motion_off_hits:3,
                         traffic_generator_mode:"ping", csi_target_pps:100
/espectre/v1/csi      -> HTTP 200，8 秒约 200 KB 二进制流（magic "ESPR"）
```

看到上面这些值，说明：**已联网、信号极好（-22 dBm）、检测算法在跑、原始 CSI 也在往外送** —— 全链路打通。

> 本项目已附一键脚本：`lab-tools/events_live.sh <ESP32_IP>`

**（可选，第二次实验再做）把原始 CSI 保存到硬盘**

前提：电脑安装 Git 和 Python，克隆 <https://github.com/francescopace/espectre>，按其文档安装 `requirements.txt`。在仓库目录执行：

```bash
# 准备（只做一次）
git clone https://github.com/francescopace/espectre && cd espectre
python3 -m pip install -r requirements.txt

# macOS / Linux（把 target 换成你的 IP）
./espectre collect --target 192.168.31.66 --label motion --duration 30 --pps 100

# Windows
.\espectre.cmd collect --target 192.168.31.66 --label motion --duration 30 --pps 100
```

- 采集程序会**主动产生实验所需的网络流量**，连接 ESP32 的 **`/espectre/v1/csi`** 原始数据流，把收到的记录保存为 **NumPy `.npz`** 文件。
- ⚠️ 它运行后会把 ESP32 切到 **external 流量模式，并保留这一设置**；后续再做普通运动检测时，应在 Live motion 的 **Advanced sensing settings** 中**恢复 internal/ping 等合适流量源**。
- ✅ **本次已确认该数据流可用**：`/espectre/v1/csi` 返回 **HTTP 200**，8 秒约 **200 KB** 二进制（magic `ESPR`，每记录 76 字节）—— 所以第二次实验可以直接采集，无阻塞。
- 指南建议：**初次实验优先完成网页看波形 + 命令看事件**，保存原始 CSI 留到第二次实验。

---

### 第 7 步 · 验收标准与快速排障

**两组第一次实验只需共同完成四件事：**

| # | 验收项 | 判定标准 | 本次状态 |
|---|---|---|---|
| ① | 电脑识别 USB 串口并烧录成功 | 能把 **Native 固件**烧进 ESP32，页面报告完成并重启 | ✅ **已达成**（ESPectre Native 3.0.0 / IDF v5.5.5） |
| ② | ESP32 连接指定 2.4 GHz Wi-Fi 并拿到 IP | 路由器 DHCP 列表 / 安装器结果页可见 IP | ✅ **已达成**（`192.168.31.66`，信道 11，RSSI -22 dBm） |
| ③ | **不接数据线**也能看到运动事件 | Live motion 显示 movement score / motion-idle 切换 | ✅ **已达成**（motion ≈ 0.9999 / idle ≤ 0.4950） |
| ④ | 看见原始 CSI | CSI Visualizer 出现**子载波曲线**或**采样统计** | ✅ **接口已验证**（`/espectre/v1/csi` HTTP 200，约 200 KB / 8 s）｜建议再补一张 Visualizer 曲线截图 |

> 四项验收**全部达成**，第一节课主线完成 ✅

**建议提交材料**
- 一张**接线照片**
- 一张 **IP 或 Device settings 截图**
- 一张 **Live motion 截图**
- 一张 **CSI 曲线截图**
- 文字记录：**ESP32 型号、路由器 SSID、信道、所用固件版本**

---

## 五、排障速查表

| 现象 | 按顺序排查 |
|---|---|
| **信号满格却完全测不到运动** | ⭐ **设备挨太近了**（最常见！）：把 ESP32 与路由器拉开 **3~5 米**，用 `link_monitor.py` 把 RSSI 调到 **-50~-70 dBm**。用"站在两者正中间"的遮挡测试确认链路 |
| **在旁边走动没反应** | ⭐ **走位错了**：必须让身体**横穿**两设备之间的连线，不是在旁边绕圈走 |
| **静止却一直报 motion** | 环境噪声饱和（风扇/空调/他人）。看 `score`：本底若达 0.86~1.0 则**信噪比=1**，**调阈值和 `hits` 都无效**，必须物理降噪或换时间 |
| **找不到串口** | ① 换数据线（可能是充电线）② 换 USB 口 ③ 换板上的另一个 Type-C 口 ④ 关闭占用串口的软件 |
| **烧录成功但串口无任何输出** | ⭐ **插错口了**：拔下来插到板上标 **`USB`** 的口（主控制台在那里，不在 `UART`） |
| **烧录成功但配网始终不成功** | ⭐ **同一根因**：`UART` 口上没有 Improv Serial。换到 **`USB`** 口后重试；用 `improv_provision.py --state` 验证是否已能收到 `Ready / Authorized` |
| **官网配网页打不开 / 选不到串口** | 改用**命令行配网**：`python3 lab-tools/improv_provision.py --port /dev/cu.usbmodem101 --state`（完全不依赖浏览器） |
| **配网返回 `Unable to connect`** | ① 确认是 **2.4 GHz**（不是 5 GHz）② **密码错**（最常见）③ 关闭**路由器隔离** ④ 重跑配网命令 |
| **想清空已存 Wi-Fi 重新配** | 按住板上 **BOOT** 键 **3 秒**（ESP32-C6 = GPIO9），回到就绪状态再配 |
| **卡在 `waiting for download`** | 用 `python3 lab-tools/reset_probe.py /dev/cu.usbmodem101` 救回正常启动 |
| **烧录失败** | ① 关闭占用串口的程序 ② 按住 **BOOT** 进下载模式 ③ 重新插拔 |
| **已联网但电脑找不到设备** | 直接用**设备返回/路由器分配的 IP**；或用 `find_espectre.py` 扫描 |
| **有网络却没有 CSI** | ① 检查 **Traffic RX** 和 **CSI callbacks** ② 切换 **DNS (UDP/TCP)** 流量源 |
| **波形有但不随人体变化** | ① 重新放置**路由器与 ESP32**（拉开距离、避免贴墙）② 在**安静环境重新校准** |

---

## 六、配套工具（本项目 `lab-tools/`）

| 脚本 | 用途 | 用法 |
|---|---|---|
| `check_serial.py` | 串口体检：列设备 / 看芯片 / 查占用 / 读日志 | `python3 lab-tools/check_serial.py -r 5` |
| `usb_console.py` | **原生 USB 口日志抓取**（含 esptool 风格复位，触发启动日志） | `python3 lab-tools/usb_console.py /dev/cu.usbmodem101 12` |
| `reset_probe.py` | 从下载模式救回正常启动（卡在 `waiting for download` 时用） | `python3 lab-tools/reset_probe.py /dev/cu.usbmodem101` |
| **`improv_provision.py`** | **命令行 Improv 配网**：查状态 / 查设备信息 / 设置 Wi-Fi | `python3 lab-tools/improv_provision.py --port /dev/cu.usbmodem101 --state` |
| **`link_monitor.py`** | **链路质量监视器**：实时显示 RSSI，并判断设备摆放是否合适 | `python3 lab-tools/link_monitor.py 192.168.31.66` |
| `find_espectre.py` | **自动扫描局域网找到 ESP32 的 IP** 并验证接口 | `python3 lab-tools/find_espectre.py` |
| `events_live.sh` | 实时监听运动事件（第 6 节命令的封装） | `./lab-tools/events_live.sh <ESP32_IP>` |

> 依赖：`python3 -m pip install --user pyserial`（`usb_console.py` / `reset_probe.py` / `improv_provision.py` 需要）。

---

## 七、官方参考

> 📄 **纯浏览器（不敲命令）的完整分步流程**：见 `docs/ESPectre-纯浏览器操作指南.md`
> —— 含页面地图、连接选择器（自动发现 / 手填 IP / Demo）、每个页面上要点的按钮、预期现象与浏览器专属排障。

- 乐鑫 ESP32-C6 开发板说明书
- ESPectre 安装与配网：`https://espectre.dev/tools/flash/`
- 设备设置：`https://espectre.dev/tools/device-settings/`
- Live motion：`https://espectre.dev/tools/monitor/`
- CSI Visualizer：`https://espectre.dev/tools/csi-visualizer/`
- ⭐ **摆放指南（必读）**：`https://espectre.dev/guides/placement/` —— 官方同样强调"3~8 米、RSSI -40~-70 dBm、走位要穿过链路"，与手册第 4.5 步实测结论一致
- 烧录与配网指南：`https://espectre.dev/guides/setup/`
- 运动游戏 / 特雷门琴：`https://espectre.dev/tools/game/`、`https://espectre.dev/tools/theremin/`
- 本地 HTTP 接口参考：`https://espectre.dev/sdk/http-api/`
- ESPectre CLI / 仓库：`https://github.com/francescopace/espectre`
