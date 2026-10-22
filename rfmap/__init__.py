"""RF Map 覆盖预测工程（Project 5）。

模块划分
--------
geometry : 房间几何（来自户型图实测标注）
grid     : 规则网格定义与掩码
data     : 实测数据读取与坐标对齐
simulate : 物理仿真数据生成（多户型）
models   : 三个网络方案 + 插值基线
train    : 训练流程（仿真预训练 + 实测微调）
evaluate : 评估指标（MAE / RMSE / 推理时间 / 跨场景）
plotting : 出图
report   : HTML / Word 报告生成
export   : 权重导出（供 Web 端推理）
"""

__version__ = '1.0.0'
__all__ = ['geometry', 'grid', 'data', 'simulate', 'models', 'train',
           'evaluate', 'plotting', 'report', 'export']
