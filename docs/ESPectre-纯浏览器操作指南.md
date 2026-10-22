# ESPectre 纯浏览器操作指南（ESP32-C6 CSI 第一节课）

> 配套文档：`outputs/ESP32-C6-CSI-第一节课操作手册.md`（命令行/原理版）
> 本文只讲**怎么在浏览器里点出来**。全程不敲命令，用官方网站在线工具完成：**烧录 → 配网 → 看运动 → 看原始 CSI 波形**。

---

## 一、先记住三件事

1. **官方工具全部是网页**，不需要写任何前端代码：`https://espectre.dev/tools/`
2. **只有三步需要插 USB 线**（烧录、配 Wi-Fi、改名/换网络）；**看数据和调参数走 Wi-Fi，不需要线**。
   - 所以：**验收项③"不接数据线也能看到运动事件"在这里天然成立**。
3. **浏览器决定成败**：必须用**桌面版 Chrome 151 或更高**。
   - ⚠️ Edge 能烧录，但官方明确说明 **Device settings 和 Live motion 在 Edge 里可能不工作**；Firefox / Safari 干脆不能烧录。
   - 本机可用：**Chrome 154 ✅**（另一个 Edge 155 只用来烧录用）。

---

## 二、页面地图（照着开就行）

| 页面 | 网址 | 用途 | 需要设备在线？ | 需要 USB 线？ |
|---|---|---|---|---|
| 工具总览 | `https://espectre.dev/tools/` | 所有工具的入口 | ❌ | ❌ |
| **安装器** | `https://espectre.dev/tools/flash/` | 烧固件 + 配 Wi-Fi | ❌（要板子插线） | ✅ |
| **设备设置** | `https://espectre.dev/tools/device-settings/` | 改名 / 看 Wi-Fi / 选 AP / MQTT | ✅ | ❌ |
| **Live motion** | `https://espectre.dev/tools/monitor/` | 看运动分数、调阈值、校准、诊断 | ✅ | ❌ |
| **CSI 可视化** | `https://espectre.dev/tools/csi-visualizer/` | 看子载波幅度/相位/IQ 曲线 | ✅ | ❌ |
| 动一动游戏 | `https://espectre.dev/tools/game/` | 用身体控制小幽灵 | ✅ | ❌ |
| 运动特雷门琴 | `https://espectre.dev/tools/theremin/` | 用身体"弹琴" | ✅ | ❌ |

> 💡 **省事技巧**：任何工具页面加 `?target=设备IP` 就自动连上，不用手点。
> 例：`https://espectre.dev/tools/monitor/?target=192.168.31.66`

---

## 三、⭐ 连接选择器：页面说"连不上设备"时先看这里

每个数据类工具（Device settings / Live motion / CSI visualizer / Game / Theremin）**首次打开都会让你选连接方式**，共三种：

| 方式 | 什么时候用 | 说明 |
|---|---|---|
| **Auto-discovery（自动发现）** | 首选 | 网页通过 `.local` 名字在局域网里找设备。**需要至少一台设备能应答**，且只支持 IPv4 |
| **手填 IP 地址** | 自动发现失败时**最可靠** | 输入形如 `192.168.31.66`。IP 从路由器后台"已连接设备"里查，或用 `lab-tools/find_espectre.py` 扫 |
| **Demo（演示模式）** | **设备不在身边时** | **用鼠标模拟运动**，不需要任何硬件。用于**先熟悉界面**（见第六节） |

> ⚠️ 三种方式里 **Demo 是模拟数据，不是真实 CSI**，不能当实验结果写进报告。
> ⚠️ 自动发现在 Windows / Linux 上**尚未验证**（需要系统 mDNS、放行 UDP 5353、允许 Local Network Access）；macOS 上已验证。**失败就手填 IP**，别纠缠。

**一次授权提醒**：Chrome 首次访问会弹 **Local Network Access（本地网络访问）**权限 → **必须点允许**，否则页面永远连不上设备。
页面顶部横幅里可以直接点**设备 ID / 设备名**来改名（**回车或点空白处保存，Esc 取消**）。

---

## 四、流程 A · 全新设备：烧录 + 配网（插 USB 线，只在第一次做）

### A1 烧录固件

1. 用**数据线**把板子插到电脑，确认电源灯亮。
   - **插板上标 `USB` 的那个口**（原生 USB Serial/JTAG），不是 `UART` 口 —— 这是本项目实测确认的关键点。
2. **先关掉一切占用串口的软件**（Arduino IDE 串口监视器、另一个安装器标签页等）。
3. 打开 `https://espectre.dev/tools/flash/`，点 **Connect USB device**。
4. 在弹窗里选**刚插线后新增的串口**：macOS 上名字含 `usbmodem` / `usbserial` / `ttyACM`。
   - 本机应选 **`/dev/cu.usbmodem101`**
   - ⚠️ 这里选的是**电脑↔板子的 USB 通信口**，不是选 Wi-Fi 网络。
5. 安装器会自动识别芯片（应显示 **ESP32-C6**）和板上已有的 ESPectre。
6. 选 **Native** 固件 + **release（稳定通道）**，然后：
   - 识别出已有固件 → 点 **Update / Reinstall**（**保留设备设置**）
   - 换固件类型或识别不出 → 会要求确认**整片擦除**（会清掉板上设置）
7. **保持页面打开和 USB 连接**，直到提示完成、板子自动重启。

**预期现象**：安装器报告完成 → 板子重启 → 页面出现下一步（Wi-Fi 表单或完成提示）。

**失败处置**：
- 看不到串口 → ① 换数据线（充电线是头号原因）② 换 USB 口 ③ **按住 BOOT 再插线，松开 BOOT**，重新点 Connect USB device（进入下载模式）
- 烧到一半失败 → 关掉其他占用串口的程序；插到电脑后置 USB 口或带电源的 HUB；用"擦除并重装"

### A2 配 Wi-Fi

1. 烧录完成后，如果板上**还没有 Wi-Fi 设置**，安装器页面里会**直接弹出 Wi-Fi 表单**。
2. 填 **2.4 GHz** 网络名和密码（**大小写敏感**），保存。
   - ⚠️ ESP32-C6 **只支持 2.4 GHz**，5 GHz 网络连不上。
3. 等页面提示连接成功，记下它显示的 IP（形如 `192.168.31.66`）。
4. **以后要换网络/改密码**：回 `https://espectre.dev/tools/flash/` 用 **Configure Wi-Fi** 重新配，**不用重烧固件**。
5. **忘了 / 配错了想重来**：按住板上 **BOOT 键 3 秒**清空 Wi-Fi 设置，再从第 1 步配一次。

> 本次实测：配网成功后设备返回 `https://espectre.dev/tools/device-settings/?target=192.168.31.66`，**末尾那串 IP 就是设备地址**，直接拿去用。
> ⚠️ 路由器重启/DHCP 租约变化后 IP 可能变，重新查一次即可。

### A3 去设备设置页确认

1. 打开 `https://espectre.dev/tools/device-settings/?target=你的IP`。
2. 在这里可以：**改设备名**、看 **Wi-Fi 连接状态**、**指定接入点(AP)**、设置 **MQTT**（做 Home Assistant 才需要）。
3. 顺利看到设备信息 = **验收项② 达成**（已连上 2.4 GHz Wi-Fi 并拿到 IP）。

---

## 五、流程 B · 设备已配好：只看效果（不插线，你下次实验走这条）

### B1 确认设备在线

打开 `https://espectre.dev/tools/device-settings/?target=你的IP`，能读出设备信息即在线。

### B2 ⭐ 先摆好位置，再接页面看数据

**这一步比任何参数都重要**，也是本课最容易失败的地方：

- ESP32 与路由器拉开 **3 ~ 8 米**（本次实测 **3~5 米** 最合适）。
- 让**人平时走动的路线正好穿过两者之间的连线**，而不是躲在某一侧。
- 天线那一端朝房间敞开，远离金属、电视、机柜、大障碍物。
- **不要让板子、天线或电源线晃** —— 晃动本身就是"运动"。

摆放合不合格，**Live motion 页面底部 Diagnostics 里的 `Wi-Fi RSSI` 就是判据**：

| Diagnostics 里的 RSSI | 说明 | 该怎么做 |
|---|---|---|
| 高于 `-40 dBm` | **离路由器太近了** | 先搬远一点，再谈调阈值 |
| **`-40` ~ `-70 dBm`** | ✅ **最合适的区间** | 直接做下面的安静/运动测试 |
| `-70` ~ `-80 dBm` | 链路偏弱，摆放更挑位置 | 仔细测，优先选更通透的路径 |
| 低于 `-80 dBm` | 信号太弱，噪声更容易进来 | 挪设备或去掉大障碍物 |

> 🔴 **本次实验的根因就是这个**：板子和路由器在**同一张桌子上**，RSSI 高达 **-22 dBm**，无线路径只有几十厘米，**人根本走不进这条路径** → 表现成"信号满格、一切正常、就是测不到运动"。
> 把板子移到 3~5 米后 RSSI 降到 **-51 ~ -57 dBm**，立刻就能测到。

> 摆放没定下来之前，**不要调阈值和滤波器**。先找位置，再校准，最后才调参。

### B3 打开 Live motion 看运动

1. 打开 `https://espectre.dev/tools/monitor/?target=你的IP`，选好设备。
2. **校准（Calibrate）**：房间保持**静止约 10 秒**，等校准完成。
   - 默认是 **Lightweight** 检测；如果这 10 秒里有突发噪声，校准会**按 5 秒一档自动延长，最多约 30 秒**。
   - 想手动重来：点 **Calibrate** 按钮（"Calibrate the room"）。
3. 走动测试：**从 ESP32 与路由器之间横穿过去**，或在 2~3 米外挥手。
4. 应当看到：
   - **Movement score（运动分数）**曲线上升并**越过阈值线**
   - **Current state** 在 `motion` / `idle` 之间切换
   - 官方预期：**运动约 1 秒内触发，停下后很快恢复**
5. 顺利看到分数变化 = **验收项③ 达成**（不接数据线也能看到运动事件）。

**本次实测对照**（命令行抓的同一条流，页面显示的是同一份数据）：

| 动作 | 状态 | score |
|---|---|---|
| 站在两者连线正中（遮挡） | `motion` | **1.0000**（70/70 全触发） |
| 沿连线来回走动 | `motion` | 0.8828（84.6% 时间触发） |
| 在旁边走动（**路径偏了**） | 偏弱 | 0.6617（59%） |
| 静止 + 环境安静 | `idle` | 0.0018 ~ 0.4950 |

> 🔑 **黄金诊断法**（分不清是设备坏了还是走位不对时用）：
> **站到两个设备连线的正中间，站着不动 15 秒。**
> score 冲到 **1.0** → 链路完全正常，问题在**走位**；score 毫无反应 → 才是**摆放/环境**问题。

### B4 在同一页面上调参（改动会立即存到设备）

**Motion settings** 区：

| 控件 | 作用 |
|---|---|
| **Motion threshold** | 运动阈值，**越低越灵敏**（本次实测校准后从 `0.411` 变成 `0.662`） |
| **Detection mode** | `Lightweight`（默认，需校准）/ `High accuracy`（**不需要校准**） |
| **Calibrate** | 手动重新校准（换位置、改朝向之后必须重做） |
| **Advanced sensing settings** → Readings to turn motion on / off | 即 `motion_on_hits` / `motion_off_hits`，连续多少次读数才切换状态 |
| **Advanced sensing settings** → Wi-Fi traffic source | 流量来源：`Ping (ICMP)` / `DNS (UDP)` / `DNS (TCP)` / `Wi-Fi raw (Experimental)` / `External device` |

> 📌 **页面上的 "Readings to turn motion on/off" 经验**：本项目实测把 on-readings 从 4 调到 12，**误报率 5.9%，而对照组是 3.4%** —— **时间滤波对本底噪声无效**，别在这上面花时间。
> 📌 **CSI 一直空 / 波形不出现时**，把流量源从 `Ping (ICMP)` 换成 **`DNS (UDP)` 或 `DNS (TCP)`**（有些路由器不回 ping）。`Wi-Fi raw` 是实验特性，**ESP32-C6 不支持**。

**Diagnostics** 区（页面底部，排障第一步）：

| 读数 | 含义 | 本次实测 |
|---|---|---|
| `Traffic TX` / `Traffic RX` | 发出 / 收到的流量 | 有正常数值 = 流量通路通 |
| `CSI callbacks` / `CSI accepted` | 硬件回调次数 / 可用 CSI 数 | **必须都不为 0**，否则不是算法问题而是没拿到 CSI |
| `Wi-Fi RSSI` | 信号强度 | 实测 **-22 dBm**（太近）→ 移到 3~5 米后 **-51~-57 dBm** |
| `CSI occupancy` / `Free heap` / `Loop time` | 运行健康度 | 用于判断设备是否过载 |

### B5 看原始 CSI 波形（验收项④）

1. 打开 `https://espectre.dev/tools/csi-visualizer/?target=你的IP`，选同一个设备，点 **Start**。
2. 正常时应出现：**子载波幅度 / 相位 / I-Q** 随时间变化的曲线，以及 `packets/s`、`RSSI` 等统计。
3. ⚠️ **两种模式不并行**：**原始 CSI 流运行时，设备的普通运动检测会暂停**；关掉 CSI 流后才恢复。所以**先做完 B3 截图，再来看波形**。
4. 顺手截一张曲线图 → **验收项④** 从"接口已验证"升级为"已看到曲线"。

### B6 趣味验证（可选，但很适合验收演示）

- **Motion Game** `https://espectre.dev/tools/game/`：用身体在板子前移动，控制小幽灵躲障碍。
- **Motion theremin** `https://espectre.dev/tools/theremin/`：身体一动就发声，**"人来了就有声音"比数字更直观**。

---

## 六、流程 C · 设备不在身边怎么办（现在就能做）

**完全可以用 Demo 模式先把界面摸熟**：

1. Chrome 打开 `https://espectre.dev/tools/monitor/`。
2. 连接选择器里选 **Demo**（不需要任何设备）。
3. 在曲线上**拖动鼠标**模拟运动（页面提示：*Drag on the chart to simulate motion*）。
4. 观察：分数如何越过阈值、`motion` / `idle` 怎么切换、Diagnostics 各栏长什么样。

这样等你回到设备边上，**第五节的每一步你都已经知道要点哪儿了**，现场只需 5 分钟。

> ⚠️ 但请注意：**Demo 是网页模拟的假数据**。报告里的任何数字都必须来自真实设备（用 `?target=IP` 连）。

---

## 七、浏览器方式 ↔ 命令行方式 对照

| 想做的事 | 浏览器（本文） | 命令行（手册第 6 步） |
|---|---|---|
| 烧录固件 | `/tools/flash/` | 需 ESP-IDF 工具链 |
| 配 Wi-Fi | 安装器的 Wi-Fi 表单 | `lab-tools/improv_provision.py` |
| 看运动分数 | `/tools/monitor/` | `curl .../espectre/v1/events` |
| 改阈值 / 校准 | monitor 的 Motion settings | `PATCH /espectre/v1/sensing`、`POST /sensing/calibrations` |
| 看原始 CSI | `/tools/csi-visualizer/` | `curl .../espectre/v1/csi`（二进制）/ `./espectre collect` 存 `.npz` |
| 看设备状态 | device-settings 页面 | `GET /espectre/v1/device|wifi|sensing` |

> 两条路**本质是同一个数据源**：都是 ESP32 上的那个 HTTP 服务（端口 `62587`）。
> 浏览器方便、好看、可截图；命令行适合**批量记录和写进实验数据**。

---

## 八、浏览器专属排障表

| 现象 | 原因 / 处理 |
|---|---|
| 页面一直连不上设备 | ① Chrome 弹的 **Local Network Access 权限要点"允许"** ② **改用 `?target=你的IP` 手填地址**（自动发现依赖 mDNS，容易失败）③ 确认电脑和设备在**同一个 Wi-Fi**、没开客户端隔离 |
| 自动发现找不到设备 | 换**手填 IP**；IP 从路由器后台或 `find_espectre.py` 拿。**设备名/ID 不能代替 IP**（找名字也要靠 mDNS） |
| Edge 里 Device settings / Live motion 打不开 | **换成 Chrome 151+**，Edge 只用来烧录 |
| 点 Connect USB device 没有串口 | ① 换**数据线** ② 换 USB 口 ③ **按住 BOOT 插线**进下载模式 ④ 关掉占用串口的软件 |
| 烧录失败 | 关掉占用串口的程序；用后置 USB 口/带电源 HUB；选"擦除并重装" |
| Wi-Fi 连不上 | ① 确认是 **2.4 GHz** ② **密码大小写** ③ 板子先靠近路由器再试 |
| 网页设备信息正常，但分数一直是 0 / 不动 | ① **先看 Diagnostics 的 `CSI callbacks` / `CSI accepted`** 是否非 0 ② 换流量源为 `DNS (UDP)` ③ 按 **B2** 重新摆位置 |
| 静止却一直报 motion | 环境噪声饱和（风扇/空调/其他人）。score 本底若到 0.86~1.0，**信噪比=1，调阈值和 readings 都无效**，只能物理降噪或换时间 |
| 想重新配 Wi-Fi | **按住 BOOT 3 秒**清设置，再回 `/tools/flash/` 配一次 |
| 想看完整启动日志 | 浏览器看不到串口日志，用 `lab-tools/usb_console.py /dev/cu.usbmodem101` |

---

## 九、下次实验：3 步恢复清单

设备目前**已断电离线**，电脑也不在同一网段。回到原来的环境后按顺序做：

1. **把电脑连回实验用的 2.4 GHz Wi-Fi**（就是给 ESP32 配网的那个网络）。
2. **给板子插回 USB 供电**（配网信息存在板内 NVS，**不会丢**，不用重新配网）。
3. **重新确认设备 IP**（可能不再是 `192.168.31.66`）：
   - 浏览器：路由器后台 → 已连接设备；
   - 或命令行：`python3 lab-tools/find_espectre.py`
4. 然后按 **第五节 B2 → B3 → B5** 走一遍，并补齐 4 张截图（接线 / IP / Live motion / CSI 曲线）。

---

## 十、官方参考

- 浏览器工具总览：`https://espectre.dev/tools/`
- 安装与配网（Setup 指南）：`https://espectre.dev/guides/setup/`
- **摆放指南**：`https://espectre.dev/guides/placement/` ← 必读，本课成败关键
- 检测原理：`https://espectre.dev/guides/detection/`
- 排障：`https://espectre.dev/guides/troubleshooting/`
- 本地 HTTP 接口：`https://espectre.dev/sdk/http-api/`
