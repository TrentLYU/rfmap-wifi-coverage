# 第 9 组项目进度

此页是成果导航。既有 RF Map 主线的方案、报告和复现命令见 [README](README.md)。下面按贡献者记录本次提交，不代填其他成员的分工或完成状态。

| 贡献者 | 工作 | 状态与证据 |
|---|---|---|
| [@cby47](https://github.com/cby47) | 49 点与放家具后 140 点的数据整理、六方法比较 | 本地已运行；[个人阶段报告](docs/progress/2026-11-06-cby47.md)、[汇总指标](outputs/member-progress/cby47/) |
| [@cby47](https://github.com/cby47) | miniAuto / Uno 网格巡测程序，前进、后退、左移调试 | 单项动作实车调试；[最终工程](hardware/grid_car/10_全图检测_顺序积木/)、[路线图](docs/assets/grid-car-route.svg) |
| [@cby47](https://github.com/cby47) | 144 点整图覆盖、位置误差、ESP32 记录与测点对齐 | **待联调验证**，不能以编译通过代替整图实测 |

下一步：先跑两列验证转接和点距，再验证整图；固定高度、朝向和无线设置，保存逐条采样时间戳；冻结模型后采新位置盲测。
