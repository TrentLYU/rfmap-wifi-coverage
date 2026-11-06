# 目录整理与参考

2026-11-06 查看以下公开仓库的首页和根目录后，采用“首页导航 + 进度入口 + 功能分类 + 实验证据”的结构。仅参考组织方式，没有复制对方源码或实验结论。

| 参考 | 可借鉴的方式 | 本组采用方式 |
|---|---|---|
| [第 6 组 CSI](https://github.com/shiyudev/wifi-csi-presence-detection) | 进度单列，成员贡献可追溯；工具、文档、测试分开 | 根目录 `PROGRESS.md` 与带账号的个人进度 |
| [第 8 组 RSSI](https://github.com/Jingyiz613/WiFi-RSSI-Map) | 首页包含结果、样例命令、图和已知限制 | 保留本组既有首页，增加个人成果导航；本地单房间实验与既有 CNN 结果分开 |
| [第 7 组 CSI](https://github.com/handsome-liujunting/ESPectre_Experiment_Group1) | 代码、数据、结果各有入口，未完成项明确 | 工程文件与汇总证据可直接打开；标明整图尚未实测 |
| [第 11 组 EPBL](https://github.com/TPProjectPBL/EPBL) | 依功能模块分目录，有协作说明 | 小车工程放 `hardware/`，独立基线实验放 `experiments/` |

## 本次增加的目录

```text
PROGRESS.md                         进度与贡献导航
hardware/grid_car/                  小车 Mixly / Uno 工程与使用说明
experiments/rssi_baselines/          独立六方法基线实验（含模拟输入）
docs/progress/                      按日期和账号记录个人进度
docs/assets/                        路线图、模拟演示、汇总结果图
outputs/member-progress/cby47/       仅汇总实验指标
```

原有 `rfmap/`、`lab-tools/`、`data/`、`models/`、`outputs/` 及历史报告保留。后续新增任务通过独立分支与 PR 提交，标题描述改动，正文注明证据、限制和贡献者。
