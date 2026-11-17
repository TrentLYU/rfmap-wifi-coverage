"""全局配置：路径、几何、网格、随机种子。

所有模块共用同一份配置，保证训练 / 评估 / 前端导出使用完全一致的网格。
"""
from __future__ import annotations

import os

import numpy as np

# ---------------------------------------------------------------- 路径
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_XLSX = os.path.join(
    ROOT, "uploads", "ses_e1c1a4cf4ffeK4Pp4puarU4OCL", "工作簿1(1).xlsx"
)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
MODEL_DIR = os.path.join(ROOT, "models")
TMP_DIR = os.path.join(ROOT, ".deepworks", "tmp")
for _d in (DATA_DIR, OUT_DIR, FIG_DIR, MODEL_DIR, TMP_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- 几何
# 实测坐标：x ∈ [0.9, 12.9]（7 列，步长 2 m），y ∈ [0, 12]（7 行，步长 2 m）。
# 实测只有 6 列（x ≤ 10.9），x = 12.9 一列无数据，故房间净宽取 13.0 m。
ROOM_W = 13.0     # 内轮廓宽（m）
ROOM_H = 12.0     # 内轮廓高（m）
WALL_T = 0.24     # 墙厚（m），仅用于展示

# AP 位置：(9, 12)。表格中 x=8.9 / y=12 单元格写的是 "9"（不是 RSSI），就是 AP 标记。
AP_POS = (9.0, 12.0)

# 物理路径损耗拟合（只用 29 个标定点拟合，见 data.py）
PL0 = 33.72397218598958
NEXP = 0.7145038792871445

# 网格：48×48，0.28 m 方格，覆盖 13.44×13.44 m（略大于房间，超出部分为无效区）
GRID_N = 48
CELL = 0.28
DOMAIN = GRID_N * CELL          # 13.44

SEED_SPLIT = 20261116
SEED_SIM_TRAIN = 20261116
SEED_SIM_VAL = 20261117
SEED_SIM_TEST = 20261118
SEED_TRAIN = 7

# 五级分级：以训练集分位数确定边界（本房间动态范围小，用数据驱动分档）
LEVEL_LABELS = ["5 级 · 极好", "4 级 · 好", "3 级 · 一般", "2 级 · 偏弱", "1 级 · 很弱"]
LEVEL_COLORS = ["#0b6b3a", "#3fa55f", "#e8c547", "#e08a3c", "#b42318"]

# 物理信道参数（用于障碍物穿透衰减）
FURNITURE = {
    "wall":     {"label": "隔断墙", "db": 8.0, "w": 2.0, "h": 0.24, "kind": "wall"},
    "wardrobe": {"label": "衣柜/柜子", "db": 5.0, "w": 1.6, "h": 0.6, "kind": "furniture"},
    "bed":      {"label": "床", "db": 3.0, "w": 2.0, "h": 1.8, "kind": "furniture"},
    "sofa":     {"label": "沙发", "db": 3.0, "w": 2.0, "h": 0.9, "kind": "furniture"},
    "chair":    {"label": "椅子", "db": 1.5, "w": 0.5, "h": 0.5, "kind": "furniture"},
    "desk":     {"label": "桌子", "db": 2.0, "w": 1.4, "h": 0.7, "kind": "furniture"},
    "toilet":   {"label": "马桶", "db": 2.5, "w": 0.7, "h": 0.6, "kind": "furniture"},
    "fridge":   {"label": "冰箱", "db": 6.0, "w": 0.7, "h": 0.7, "kind": "furniture"},
    "tv":       {"label": "电视/电视柜", "db": 2.0, "w": 1.6, "h": 0.4, "kind": "furniture"},
    "bath":     {"label": "浴缸", "db": 3.5, "w": 1.7, "h": 0.8, "kind": "furniture"},
    "washer":   {"label": "洗衣机", "db": 4.0, "w": 0.6, "h": 0.6, "kind": "furniture"},
    "shelf":    {"label": "书架", "db": 3.5, "w": 1.2, "h": 0.35, "kind": "furniture"},
    "plant":    {"label": "绿植", "db": 1.0, "w": 0.5, "h": 0.5, "kind": "furniture"},
    "door":     {"label": "室内门", "db": 2.0, "w": 0.9, "h": 0.15, "kind": "opening"},
}

# 网络训练超参
BASE_CH = 12
EPOCHS = 60
BATCH = 8
LR = 2e-3
PATIENCE = 14
N_TRAIN_SIM = 480
N_VAL_SIM = 96
N_TEST_SIM = 64

np.random.seed(SEED_TRAIN)
