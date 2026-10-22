# AGENTS.md

> 本文件是给 AI 编码助手（及新成员）的项目指令，按课程要求"保持简短、相关、与项目实际一致"。
> 版本 v0.3 ｜ 2026-10-18 ｜ 负责人：【待填】

> ✅ **同步状态（2026-10-18）**：`docs/01`~`docs/04` 四份文档已**统一到同一条主线**——
> **「用少量实测点位预测整间房的 Wi-Fi（RSSI）覆盖」**（Proposal v1.0 / PRD v1.0 / Tech Design v2.0 / Decision Record v0.3），
> 需求编号（`FR-01~FR-20`、`BR-01~BR-17`、`AC-01~AC-19`）在四份文档间通用。
> **需求冲突时一律以 `docs/02-MVP-PRD.md` v1.0 为准。**
> 第一阶段的「CSI 运动检测」已完成且在仓库中仍可用，但**不是**需求主线，只作为遗留能力保留（见 `docs/ESP32-C6-CSI-无外网完整实验流程.md`）。
> 注意：训练/划分/推理脚本**尚未实现**（`docs/03` 标为 `未实现`），不要在报告里当成已完成。

## 项目是什么

ESP32-C6 无线测量 + RF Map 覆盖预测原型：让 ESP32-C6（跑 ESPectre 固件）连上 2.4 GHz 路由器，
在房间的若干点位读出 RSSI（CSI 可一并保存），再由一个**本地训练**的小型卷积网络，从**少量实测点**
补全出**整间房的二维 RSSI 覆盖图**，并给出 MAE/RMSE（dB）误差与插值基线对比。
运行范围：**仅局域网**，采集与推理都不依赖外网；产品不含云端 AI，模型在本机训练与加载。
> 注：第一阶段的「CSI 运动检测」能力**已完成且仍可用**（见 `docs/ESP32-C6-CSI-无外网完整实验流程.md`），
> 但**不是**本项目的需求主线；需求请一律以 `docs/02-MVP-PRD.md` 为准。

## 权威文档（有冲突时以这些为准）

| 文档 | 内容 |
|---|---|
| `docs/01-Proposal.md` | 问题、用户、证据、里程碑 |
| `docs/02-MVP-PRD.md` | **需求权威来源**：F-x / FR-xx / BR-xx / AC-xx / NFR-xx / SM-xx |
| `docs/03-Tech-Design.md` | 架构、选型（DR-xx）、数据、失败处理（FH-xx）、命令与验证 |
| `docs/04-Agent-Decision-Record.md` | 决策记录（AD-xx）与 bug 记录 |
| `RF_Map数据与网络训练.pdf` | **任务书**：采集规范、划分规则、三个网络方案与评估口径（上游要求的出处） |
| `RFMap离线采集操作卡.md` | 采集现场怎么操作、成本估算 |
| `docs/ESP32-C6-CSI-无外网完整实验流程.md` | 无外网时的可执行流程（第一阶段遗留能力） |
| `docs/ESPectre-纯浏览器操作指南.md` | 有外网时的浏览器路径（仅作排查辅助，非 MVP 交付路径） |

不要新增与上述文档**互相矛盾的重复摘要**；改动请在原文档内更新并写变更记录。

## 硬性约束（改代码/写文档前必看）

1. **只支持 2.4 GHz**，ESP32-C6 不支持 5 GHz。
2. 设备与电脑**必须同网段**，且路由器需关闭 **AP 隔离**（现场掩码可能不是 /24）。
3. 设备摆放 **3~5 米**，RSSI 应落在 **-40 ~ -70 dBm**；过近（如 -22 dBm）读到的是**饱和值**，既测不出运动、也不能用于覆盖建模（PRD BR-04）。
4. 调用设备接口**必须带 `Origin: https://espectre.dev`** 请求头（curl 无跨源限制，网页有）。
5. 采集一份地图期间**路由器位置/配置全程不动**、只移设备；各点**统一接收高度与朝向**；人退到固定位置（任务书要求）。
6. **不在 -40~-70 dBm 的点要标注但不得删除**，也不得用插值结果充当实测答案。
7. **推理程序不得读取隐藏实测点/参考答案**；评价只能用留出的隐藏实测点（PRD BR-05）。
8. **原始 CSI 抓取期间，运动检测会暂停**；两者不能并行。
9. **CSI 计数为 0 时不要调阈值**——先查流量激励与链路。
10. 主串口是板上标 **USB** 的原生口（`/dev/cu.usbmodem101`），**不是** `UART` 口。
11. 无外网时**不要尝试用官方网页工具**，直接走 `curl` + `lab-tools/`。
12. 不要把 Wi-Fi 口令等凭据写进仓库；**来源不明的数据**（如 `data.xlsx`）不得当训练数据（PRD OQ-02）。

## 常用命令

### RF Map 采集（当前主线）

```bash
# 先只看要采哪些点（不连设备）
python3 lab-tools/rssi_survey.py --grid 0,0,6,4,1 --dry-run
# 一份部署 = 一份参考地图；中断后重跑同一条命令会自动续采
python3 lab-tools/rssi_survey.py --ip <设备IP> --map-id ROOM-A-dp1 --room ROOM-A \
  --height 1.0 --grid 0,0,6,4,1 --seconds 20 --settle 3
# 产出：data/rfmap/<map-id>.csv（每点 均值/中位数/标准差）+ <map-id>.meta.json
```

> 采集规范（任务书要求，不可变）：路由器位置/配置全程不动、只移设备；**各点统一接收高度与朝向**；
> 人退到固定位置；不在 -40~-70 dBm 的点要标注但**不得删除**，也不得用插值结果充当实测答案。

### 划分 / 训练 / 推理（**未实现**，命令形式先固定在此）

```bash
python3 lab-tools/rfmap_split.py --maps data/rfmap/*.csv --out data/rfmap/splits/p1.json
python3 lab-tools/rfmap_train.py --split data/rfmap/splits/p1.json --model unet --out models/run-1
python3 lab-tools/rfmap_infer.py --model models/run-1 --points <输入点.json> --out outputs/rfmap/run-1
```

### 设备与链路自检（采集前必做）

```bash
BASE=http://<设备IP>:62587/espectre/v1
ORIGIN='Origin: https://espectre.dev'

python3 lab-tools/find_espectre.py                       # 找设备 IP
python3 lab-tools/link_monitor.py <设备IP> 30            # 看 RSSI / 摆放判定
curl -s -H "$ORIGIN" $BASE/health                        # 自检
curl -s -H "$ORIGIN" $BASE/wifi                          # 频段与 RSSI（band 应为 2g）
```

### 第一阶段遗留能力（CSI 运动检测，仍可用，非主线）

```bash
curl -s -H "$ORIGIN" $BASE/diagnostics                   # CSI 计数须增长
curl -s -X POST -H "$ORIGIN" $BASE/sensing/calibrations  # 校准（人静止 10s）
curl -sN -H "$ORIGIN" --max-time 60 $BASE/events > S0.log  # 采集事件流
curl -s -H "$ORIGIN" --max-time 8 $BASE/csi > csi.bin    # 原始 CSI（首 4 字节 ESPR）
python3 lab-tools/parse_events.py data/*/*.log           # 出统计结论
```

依赖：Python 3 标准库即可；串口类脚本需 `python3 -m pip install --user pyserial`；训练需 PyTorch + `segmentation_models_pytorch`（版本待定，见 `docs/02` OQ-05）。

## ID 规范（引用需求时统一使用）

`FR-xx` 功能需求｜`NFR-xx` 非功能需求｜`BR-xx` 业务规则｜`AC-xx` 验收标准｜
`F-x` 核心功能｜`SM-xx` 成功度量｜`AS-xx` 假设｜`OQ-xx` 待决问题｜
`DR-xx` 设计决策｜`FH-xx` 失败处理｜`AD-xx` 决策记录｜`TS-xx` 测试场景

## 验证期望（改完必须做）

- 命令没实际跑过，就在文档里标 `❓未验证`；跑通后改成 `已实测`；**还没写的东西标 `未实现`**，不要说成已完成。
- 任何声称"完成"的行为，都要给出**可复查的证据**：命令输出、采集文件、截图或测试结果。
- 最小回归（当前主线）：
  1. `GET /health` 正常，`GET /wifi` 的 `band=2g`；
  2. `rssi_survey.py --dry-run` 点位清单与预期一致（`--grid 0,0,6,4,1` = 35 点）；
  3. 抽 1 个点采 20 s，`n_samples > 0`；
  4. 划分自检通过（无 `map_id` 跨集合、输入点与隐藏点无交集）——脚本**未实现**前跳过并注明。
- 遗留能力回归（第一阶段）：`/diagnostics` 计数增长 → 一次 S0+S3 采集 → `parse_events.py` 判定符合预期 → `csi.bin` 首 4 字节为 `ESPR`。
- 重要 bug 按「症状 / 预期 / 调查 / 修复 / 验证」记入 `docs/04-Agent-Decision-Record.md`（**bug ID 不重编号**）。
- 采纳或否决 AI 建议时，在同一文件补一行决策记录，写清**理由与验证方式**。

## 不要做

- 不要为了让设计"更完整"而引入认证、数据库、支付、微服务、公网托管、云端 AI —— 除非 `docs/02-MVP-PRD.md` 里有对应需求。
- 不要编造访谈原话、市场规模、用户数量、竞品缺陷；未经证实的说法一律标注为假设。
- 不要静默改写已定稿文档；用变更记录说明范围变化。
