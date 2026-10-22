# 无外网（Wi-Fi 没有互联网）完整实验流程

> 适用场景：路由器/校园网**连不上互联网**，但**局域网是通的**（ESP32 和电脑能互相访问）。
> 本次实验的全部数据都走局域网，**采集过程不需要一字节外网流量**。
> 配套：`outputs/ESP32-C6-CSI-第一节课操作手册.md`（原理+命令行）、`docs/ESPectre-纯浏览器操作指南.md`（有外网时用）。

---

## 〇、先明确：没网到底影响什么

| 环节 | 需要外网？ | 没网时怎么办 |
|---|---|---|
| ESP32 连路由器（局域网 DHCP） | ❌ 不需要 | 正常做，路由器本地就发 IP |
| 烧录固件 | ✅ **需要** | **提前在有网时做完**，之后永不重复 |
| 给设备配 Wi-Fi | ✅ **需要** | 同上（配网信息存在板内 NVS，断电不丢） |
| **网页工具**（monitor / CSI visualizer / 游戏） | ✅ **需要** | ❌ **没网就用不了** → 全部改用下面的 curl |
| 设备自身产生流量（`ping` / `dns`） | ❌ **不需要**（实测修正） | 默认 `ping` 打的是**局域网网关**，网关照常回包，**不依赖外网** |
| 读运动数据、改参数、校准 | ❌ 不需要 | 正常做 |
| 采集原始 CSI 二进制流 | ❌ 不需要 | 正常做 |

### 为什么网页工具没网就用不了（重要，别浪费时间）

官方工具页固定来自 `https://espectre.dev`，而 ESP32 固件**只信任这一个来源**（请求必须带 `Origin: https://espectre.dev`）。
所以：
- 打不开 `espectre.dev` → 页面根本加载不出来；
- 就算提前把页面存到本地用 `python -m http.server` 打开，**已发布的固件也会拒绝本地页面的连接**（只有编译时开了 `CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED=y` 的固件才接受本地预览）；
- 用 `curl` 则完全不受影响——命令行没有跨源限制，**这就是没网时的正路**。

> ✅ **结论：没网 ⇒ 全部走命令行（curl）+ 本仓库的 `lab-tools/` 脚本。**
> 如果只是"路由器没插网线"，而你**能开手机热点**，那建议直接用热点（算有外网），走 `docs/ESPectre-纯浏览器操作指南.md` 会更直观。

---

## 一、网络拓扑（没网也成立）

```
        ┌──────────────── 无外网路由器 / 热点 ────────────────┐
        │  2.4 GHz，SSID=实验用网络，局域网互通，关掉 AP 隔离   │
        └───────┬───────────────────────────┬────────────────┘
                │ Wi-Fi (2.4G)              │ Wi-Fi (2.4G)
        ┌───────┴────────┐          ┌───────┴─────────┐
        │  ESP32-C6      │←— 3~5 m —→│  你的 Mac        │
        │  192.168.x.66  │  链路      │  192.168.x.150  │
        │  HTTP :62587   │            │  curl / 发包     │
        └────────────────┘            └─────────────────┘
                 ↑ 人从这条连线正中横穿过去
```

三条硬性要求：
1. **2.4 GHz**（ESP32-C6 不支持 5 GHz）
2. **电脑和设备在同一网段**（前三位相同，如都是 `192.168.31.x`）
3. **路由器关闭"AP 隔离 / 客户端隔离"**，否则电脑访问不到设备；校园网常见此限制 → 换手机热点

---

## 二、唯一一次"需要外网"的准备（提前做完）

在有外网的环境下把这两件事做掉，之后实验全程离线：

1. **烧录**：开 `https://espectre.dev/tools/flash/` → Connect USB device → 选 `/dev/cu.usbmodem101`（板上标 **USB** 的口，不是 UART 口）→ 选 **Native / release** → 安装完重启。
2. **配网**：安装器里弹出 Wi-Fi 表单 → 填 **2.4 GHz** 的 SSID 和密码 → 记下它显示的 IP。

同时把本节需要的东西**离线备好**（没网时下不了）：
- 本仓库（含 `lab-tools/`）已在电脑上；
- Python 3.9.6 已装（这些脚本**只用标准库**，不需要 pip 装任何东西）。

> 记录卡（填一次，后面一直用）：
> 设备 IP `____________` 设备名 `____________` 采集日期 `____________`

---

## 三、到场后先自检（3 分钟，不通过就别往下做）

```bash
# 0) 设好变量，后面所有命令直接复用
BASE=http://192.168.31.66:62587/espectre/v1     # ← 改成你的设备 IP
ORIGIN='Origin: https://espectre.dev'

# 1) 设备活着吗
curl -s -H "$ORIGIN" $BASE/health

# 2) 身份和固件对不对
curl -s -H "$ORIGIN" $BASE/device

# 3) 网络状态：看 connected / ssid / band / rssi
curl -s -H "$ORIGIN" $BASE/wifi

# 4) 最关键的运行判据
curl -s -H "$ORIGIN" $BASE/diagnostics
```

**自检判据表**：

| 检查项 | 合格 | 不合格怎么办 |
|---|---|---|
| `/health` 有返回 | HTTP 200 + JSON | 000/超时 → 电脑和设备不同网段、AP 隔离、或设备没上电 |
| `/device` | `esp32c6` / `Native` / 版本号 | 固件类型不对 → 回第二节重烧 |
| `/wifi` 的 `band` | `2g` | 连成 5G → 重配 2.4 GHz |
| `/wifi` 的 `rssi` | **`-40` ~ `-70 dBm`** | 高于 -40 → **太近，移远**；低于 -70 → 挪位置/去障碍 |
| `/diagnostics` 的 `CSI callbacks` / `CSI accepted` | **非 0 且持续增长** | 为 0 → **先做第四节**，不是算法问题 |

> 也可以用脚本一把过：`python3 lab-tools/find_espectre.py`（扫局域网找设备 IP）、
> `python3 lab-tools/link_monitor.py 192.168.31.66 30`（连续看 RSSI 并给出"太近/合适/太远"判语）。

⚠️ **自检没过就不要采集**。上次的教训就是：RSSI `-22 dBm`（板子和路由器同在一张桌上），链路只有几十厘米，人走不进这条路径，表现成"一切都正常、就是测不到运动"。

---

## 四、流量激励：先确认它工作，多数情况不用改

CSI 是**信道状态**，必须**有帧在收发**才测得到。

**实测修正（关键）**：默认的 `ping` 模式打的是**局域网网关**，**不需要外网**。设备串口日志：

```
I TrafficGen: started (mode=ping, target=100 CSI pps, send=100 pps, target_ip=192.168.31.1)
I ... | gen:99.7 tx:106.7 rx:100.7 cb:100.7 accepted:99.7 hwerr:0.0 occ:97% | ch:1 rssi:-36
```

`rx≈100 pps` + `accepted≈100 pps` = **激励与 CSI 都正常** ✅。
**只有这里异常**，才需要做下面 4.1~4.3 的切换。

### 4.1 切成外部流量模式

```bash
curl -s -X PATCH -H "$ORIGIN" -H "Content-Type: application/json" \
  -d '{"traffic_generator_mode":"external"}' $BASE/sensing
```

### 4.2 由电脑在局域网内持续发包（推荐）

开一个终端**保持发包**（实验全程别关）：

```bash
python3 lab-tools/traffic_gen.py --target 192.168.31.66 --duration 900
```

发不出去/设备仍无反应时，换组播方式再试：

```bash
python3 lab-tools/traffic_gen.py --target 192.168.31.66 --group 239.255.0.1 --duration 900
```

反向排查（怀疑是设备往外发）：

```bash
python3 lab-tools/traffic_gen.py --listen --duration 10
```

### 4.3 验证激励真的生效（**唯一客观判据**）

```bash
curl -s -H "$ORIGIN" $BASE/diagnostics
```

| 现象 | 判断 | 处理 |
|---|---|---|
| `CSI callbacks` / `CSI accepted` **持续变大** | ✅ 激励生效 | 进入第五节 |
| 一直是 **0** | ❌ 没有帧被测量 | 换 `--group` 组播；确认电脑和设备同网段；确认设备确实在 `external` 模式 |
| `Traffic TX` 有数、`CSI accepted` 很小 | ⚠️ 流量在但 CSI 少 | 换 `dns_udp` / `dns_tcp` 试试；或把电脑挪近设备 |

**三种模式的取舍**：

| `traffic_generator_mode` | 没网时可用？ | 说明 |
|---|---|---|
| `ping`（默认） | ✅ **可用（首选）** | 目标默认＝**局域网网关**（实测 `target_ip=192.168.31.1`），网关回 ICMP，**不依赖外网** |
| `dns` / `dns_tcp` | ⚠️ 视路由器而定 | 没外网时路由器多回 NXDOMAIN，通常仍有流量；无流量再换回 `ping` |
| **`external`** | ✅ **备用** | 设备侧激励异常时改由电脑发包（标记必须是**正好 4 字节 👻**） |
| `wifi_raw` | ❌ | 实验特性，**ESP32-C6 不支持** |

---

## 五、校准（无网可做）

把设备摆在 **3~5 米**、人**退出房间或站到链路之外**，保持**静止约 10 秒**：

```bash
curl -s -X POST -H "$ORIGIN" $BASE/sensing/calibrations
# 若报参数错误，补上 JSON 头：
# curl -s -X POST -H "$ORIGIN" -H "Content-Type: application/json" -d '{}' $BASE/sensing/calibrations
```

校准后确认阈值已更新：

```bash
curl -s -H "$ORIGIN" $BASE/sensing      # 看 threshold / detector / motion_on_hits
```

> 实测参考：上一轮校准把 `threshold` 从 `0.411086` 更新为 `0.662185`。
> 校准期间若有噪声（有人走动、风扇），校准会按 5 秒一档自动延长（最多约 30 秒）——**等它结束再做下一步**。

---

## 六、标准采集时序（一轮 4.5 分钟，做 3 轮 ≈ 13.5 分钟）

每段都用 `curl -N --max-time N` 抓 SSE 事件流，**存成独立文件**（`-N` 关缓冲，保证实时落盘）：

```bash
mkdir -p data
D=data/$(date +%m%d_%H%M)          # 每轮一个时间戳目录
mkdir -p $D

# S0 空场本底 60s —— 所有人退出房间
curl -sN -H "$ORIGIN" --max-time 60 $BASE/events > $D/S0_quiet.log

# S1 静止对照 60s —— 人在链路外站着不动
curl -sN -H "$ORIGIN" --max-time 60 $BASE/events > $D/S1_still.log

# S2 走动 60s —— 沿胶带标记的走位线，来回横穿链路正中
curl -sN -H "$ORIGIN" --max-time 60 $BASE/events > $D/S2_walk.log

# S3 遮挡 30s —— 站在 ESP32 与路由器连线正中，静止不动
curl -sN -H "$ORIGIN" --max-time 30 $BASE/events > $D/S3_block.log

# S4 恢复 60s —— 立刻离开，看是否回落
curl -sN -H "$ORIGIN" --max-time 60 $BASE/events > $D/S4_recover.log
```

**每段命令一敲，立刻按脚本里的动作去做**（S0/S1 要退出，S2 要走动，S3 要站正中），因为计时是从命令开始算的。

**做满 3 轮**（`S0→S4` 重复三次），并记录环境：

| 轮次 | 时间 | 场地 | 人数 | 空调/风扇 | 备注 |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

> 同一场地、同一走位、同一时长 —— **可复现**才是实验数据，单次碰巧的漂亮数字没有意义。

---

## 七、分析（一条命令出结论）

```bash
python3 lab-tools/parse_events.py $D/*.log
```

输出示例（用上次真实数据跑的）：

```
文件: block.log
事件数: 70    时长: 17.7 s    平均间隔: 0.256 s (≈3.9 条/秒)
motion 占比: 100.0%   (70/70)
score: avg=1.0000  median=1.0000  p95=1.0000  max=1.0000  min=0.9999
分段(每 5 s): 0s:20动/1.00  5s:20动/1.00  10s:19动/1.00  15s:11动/1.00
判定: ⚠️ 全程 motion 且最低分就已达 1.00 —— 更像环境噪声饱和或持续遮挡 …
```

**期望看到的规律**（这是"实验成功"的量化证据）：

| 阶段 | 期望 motion 占比 | 期望 score 均值 |
|---|---|---|
| S0 空场本底 | 越低越好（< 10%） | 接近 0 |
| S1 静止对照 | < 20% | 低 |
| **S2 走动** | **明显升高（> 50%）** | **明显升高** |
| **S3 遮挡** | **接近 100%** | **接近 1.0** |
| S4 恢复 | 回落 | 回落 |

**上次实测对照**（同样方法，可以拿来比）：

| 条件 | motion 占比 | score 均值 |
|---|---|---|
| 站在链路正中（遮挡，70/70 全触发） | **100.0%** | **1.0000** |
| 沿连线来回走动 | **84.6%** | **0.8828** |
| 走动但路径偏了 | 59.0% | 0.6617 |
| 环境噪声爆发期 | 100%（min 0.86） | 0.9960 ← **信噪比≈1，这一轮作废** |

> 🔑 **黄金诊断法**：S3 站在链路正中若得到 **100% / 1.0**，说明设备、链路、流量全都正常，问题只会出在走位；
> 若 S3 也毫无反应，才是设备/摆放/激励的问题（回到第一节和第四节）。

**分段一栏怎么用**：`分段(每 5 s)` 里某几段特别高、其他段接近 0，说明运动发生时你**还没走到位**（或走位没穿过链路正中）→ 检查胶带标记。

---

## 八、验收对照（第一节课 4 项）

| # | 验收项 | 没网方案怎么达成 | 证据文件 |
|---|---|---|---|
| ① | 设备能烧录并启动 | 第二节（提前做） | 烧录完成截图 |
| ② | 连上 2.4 GHz 并拿到 IP | `curl $BASE/wifi` | `wifi` 输出（band=2g） |
| ③ | 不接数据线能看到运动事件 | 全程 Wi-Fi + `curl $BASE/events` | `$D/S2_walk.log`、`$D/S3_block.log` |
| ④ | 能读到原始 CSI | `curl .../csi` 存二进制 | 见下 |

补第④项（原始 CSI，不需要外网）：

```bash
curl -s -H "$ORIGIN" --max-time 8 $BASE/csi > $D/csi.bin
ls -l $D/csi.bin && head -c 4 $D/csi.bin | xxd     # 应看到魔数 ESPR
```

> 实测：8 秒约 **200 KB**，首 4 字节为 `ESPR`（小端 magic）。
> ⚠️ 抓原始 CSI 时设备的普通运动检测会暂停，所以**先做完第六节再抓**。

---

## 九、没网专属排障表

| 现象 | 原因 / 处理 |
|---|---|
| `/health` 返回 000 / 超时 | 不同网段；路由器开了 **AP 隔离**；设备没上电；IP 变了（`python3 lab-tools/find_espectre.py` 重扫） |
| 能访问设备但打不开 espectre.dev | **正常**，就是没网。本流程全程不需要它 |
| `CSI callbacks` 一直是 0 | 激励没生效 → 先看串口日志的 `rx` / `accepted`；确实为 0 再看**第四节**，切 `external` + `traffic_gen.py`（可加 `--group` 组播） |
| 网页上的"流量来源"改了没效果 | 默认 `ping` 打**网关**其实就好使；真要改就优先试 `dns_udp`，再考虑 `external` |
| 静态时 motion 就爆满 | 环境噪声饱和（风扇/空调/其他人）→ 换时间或换房间，**调阈值和 readings 无效** |
| 改了 `motion_on_hits` 没用 | 实测：4→12 时误报 5.9%，对照组 3.4% —— 时间滤波对本底噪声无效 |
| 设备连不上 2.4 GHz | 只能 2.4 GHz；密码大小写；连不上时把板子靠近路由器再配 |
| 想重新配网但没网 | 配网需要外网页面 → **按住 BOOT 键 3 秒**清设置，等有网时再配（或换手机热点） |
| 电脑想同时上网和连设备 | 别用双网卡硬凑：跨源 + 混合内容会被浏览器拦。**没网就用 curl**，要网页就用手机热点让两边都有网 |

---

## 十、命令速查卡（打印带走）

```bash
# —— 变量 ——
BASE=http://<设备IP>:62587/espectre/v1
ORIGIN='Origin: https://espectre.dev'

# —— 自检 ——
curl -s -H "$ORIGIN" $BASE/health
curl -s -H "$ORIGIN" $BASE/wifi
curl -s -H "$ORIGIN" $BASE/diagnostics        # 看 CSI callbacks 是否增长

# —— 没网时的流量激励 ——
curl -s -X PATCH -H "$ORIGIN" -H "Content-Type: application/json" \
  -d '{"traffic_generator_mode":"external"}' $BASE/sensing
python3 lab-tools/traffic_gen.py --target <设备IP> --duration 900

# —— 校准（人静止 10 s）——
curl -s -X POST -H "$ORIGIN" $BASE/sensing/calibrations

# —— 采集（改时长即可：60 / 30）——
curl -sN -H "$ORIGIN" --max-time 60 $BASE/events > S2_walk.log

# —— 原始 CSI ——
curl -s -H "$ORIGIN" --max-time 8 $BASE/csi > csi.bin

# —— 分析 ——
python3 lab-tools/parse_events.py data/*/*.log
```

**没网时唯一不能做的事：用官方网页工具看曲线。** 其余全部可做，且数据与网页显示的是同一个源（设备 `:62587` 上的 HTTP 服务）。

> ⚠️ **易混点**：**电脑有外网 ≠ 能打开设备页面**。官方页面是从 `https://espectre.dev` 加载的，页面会去请求设备的局域网 IP；只要**电脑和设备不在同一张网**（例如电脑在校园网 `10.12.0.x`、设备在路由器 `192.168.31.x`），页面能打开但**永远连不上设备**。要么把两者放进同一张网（**手机热点最省事**），要么放弃网页、用 `curl` + `parse_events.py`。

---

## 十一、执行清单（照勾）

- [ ] 板子已烧录 Native/release 并配好 2.4 GHz（**提前、有网时完成**）
- [ ] 电脑与设备同网段，能 `curl` 到 `/health`
- [ ] 胶带标好：路由器点 A、设备点 B、走位线
- [ ] `rssi` 落在 `-40 ~ -70 dBm`
- [ ] `CSI callbacks` 在增长（默认 `ping` 打网关即可；确实为 0 才切 `external` + 发包脚本）
- [ ] 已校准（人静止 10 s）
- [ ] 3 轮 × 5 阶段全部采完并记了环境
- [ ] `parse_events.py` 出对比表
- [ ] S3 遮挡段达到 ≈100% / ≈1.0（黄金诊断通过）
- [ ] 已抓 `csi.bin` 并确认首 4 字节为 `ESPR`
- [ ] 4 张截图/文件齐备（烧录、wifi、events、csi）
