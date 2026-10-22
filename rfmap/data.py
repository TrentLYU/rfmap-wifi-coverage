"""实测数据读取、坐标对齐与规范化。

数据来源
--------
``data.xlsx``（工作区根目录，Sheet1，A1:D50）：表头为
``'', 'x轴（m）', 'y轴（m）', 'RSSI（dBm）'``，共 **49** 个测点。

坐标对齐
--------
实测点为现场人工布置的标称坐标，x 取值 0/2/4/6/8/10.3/12，y 取值 0/2/4/6/8/10/12，
名义跨度 12 m；而户型图内轮廓为 12.643 x 11.731 m。两者存在
``x: +0.643 m``、``y: -0.269 m`` 的现场量测偏差。

为了把实测点放进户型图坐标系，这里做**各轴分别对齐**：

.. math::

    x_{room} = x_{data} \\times \\frac{12.643}{12}, \\qquad
    y_{room} = y_{data} \\times \\frac{11.731}{12}

该变换保持点与点之间的相对位置关系不变，并保证四角测点恰好落在房间内角上。
所有报告数字都会显式记录这一变换，避免"看起来对齐了、其实悄悄挪了点"。

.. warning::

    最终采用的对齐**不是**两轴等比：见 :data:`X_SCALE` / :data:`Y_SCALE` 的说明。
    x 轴保持现场标称值（X_SCALE = 1），只有 y 轴按内高归一。

为什么 x 轴不缩放
------------------
实测数据在 ``x = 12`` 处**缺少 y = 6 与 y = 8 两点**，改用 ``(11.7, 6)`` 与
``(11.7, 8.9)`` 替代。把 ``x=11.7`` 乘上 12.643/12 = 1.0536 会得到 12.327，
落进结构柱 ``x in [11.813, 12.643], y in [5.761, 6.771]`` 内 —— 与"该点确实测到了
-44 dBm"这一事实矛盾。保持 x 不缩放（11.7 < 11.813）则完全自洽：

* ``(12, 6)`` -> 柱体内部 -> 无法布点 -> 数据缺失；
* ``(11.7, 6)`` -> 柱体外侧 0.113 m -> 可以布点 -> 数据存在。

x 方向保留现场标称值的代价是：房间在 x = 12 ~ 12.643 m 之间留有一段未采样边带，
这一点会在报告中如实标注。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

from . import geometry as G

NOMINAL_SPAN = 12.0

#: x 轴对齐系数：保持现场标称值（理由见模块 docstring）
X_SCALE = 1.0
#: y 轴对齐系数：把实测 12 m 跨度压缩到户型图内高 11.731 m
Y_SCALE = G.INNER_H / NOMINAL_SPAN


@dataclass
class Point:
    x_data: float
    y_data: float
    rssi: float
    idx: int = 0

    # ---- 对齐后的房间坐标 ----
    @property
    def x(self) -> float:
        return self.x_data * X_SCALE

    @property
    def y(self) -> float:
        return self.y_data * Y_SCALE

    def to_row(self) -> dict:
        return {
            'idx': self.idx,
            'x_data': round(self.x_data, 3),
            'y_data': round(self.y_data, 3),
            'rssi_dbm': self.rssi,
            'x_room': round(self.x, 4),
            'y_room': round(self.y, 4),
            'inside_room': G.inside_room(self.x, self.y, tol=1e-3),
        }


# --------------------------------------------------------------------------
# 读取
# --------------------------------------------------------------------------
def read_xlsx(path: str) -> list[Point]:
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    pts: list[Point] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row is None or len(row) < 4:
            continue
        _, xs, ys, rssi = row[:4]
        if xs is None or ys is None or rssi is None:
            continue
        pts.append(Point(float(xs), float(ys), float(rssi)))
    pts.sort(key=lambda p: (p.x_data, p.y_data))
    for k, p in enumerate(pts, 1):
        p.idx = k
    return pts


def read_csv(path: str) -> list[Point]:
    pts: list[Point] = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            pts.append(Point(float(row['x_data']), float(row['y_data']),
                             float(row['rssi_dbm']), int(row['idx'])))
    return pts


def load(path: str | None = None, root: str = '.') -> list[Point]:
    """优先读规范化 CSV，缺失时回退到 data.xlsx。"""
    csv_path = path or os.path.join(root, 'data/rfmap/measured.csv')
    if os.path.exists(csv_path):
        return read_csv(csv_path)
    return read_xlsx(os.path.join(root, 'data.xlsx'))


# --------------------------------------------------------------------------
# 写出
# --------------------------------------------------------------------------
FIELDS = ['idx', 'x_data', 'y_data', 'rssi_dbm', 'x_room', 'y_room', 'inside_room']


def write_csv(pts: list[Point], path: str) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for p in pts:
            w.writerow(p.to_row())


def quality_report(pts: list[Point]) -> dict:
    """数据质检：范围、重复点、缺失点、面积覆盖等。"""
    xs = sorted({p.x_data for p in pts})
    ys = sorted({p.y_data for p in pts})
    rssis = [p.rssi for p in pts]
    grid = {(p.x_data, p.y_data) for p in pts}
    full = {(x, y) for x in xs for y in ys}
    missing = sorted(full - grid)

    # 最近邻间距
    import math
    mind = None
    arr = [(p.x, p.y) for p in pts]
    for a in range(len(arr)):
        for b in range(a + 1, len(arr)):
            d = math.dist(arr[a], arr[b])
            if mind is None or d < mind:
                mind = d

    outside = [p.idx for p in pts if not G.inside_room(p.x, p.y, tol=1e-3)]

    # 任务书 BR-04：正常摆放时 RSSI 应落在 -40 ~ -70 dBm；超出者必须标注、不得删除。
    br04_out = [p.idx for p in pts if not (-70.0 <= p.rssi <= -40.0)]
    br04_strong = [p.idx for p in pts if p.rssi > -40.0]
    br04_weak = [p.idx for p in pts if p.rssi < -70.0]

    return {
        'n_points': len(pts),
        'x_values_data': xs,
        'y_values_data': ys,
        'rssi_min': min(rssis),
        'rssi_max': max(rssis),
        'rssi_mean': round(sum(rssis) / len(rssis), 3),
        'rssi_median': round(sorted(rssis)[len(rssis) // 2], 3),
        'br04_range': [-70.0, -40.0],
        'br04_violations': br04_out,
        'br04_too_strong': br04_strong,
        'br04_too_weak': br04_weak,
        'missing_cells_nominal': missing,
        'min_neighbor_spacing_m': round(mind, 3) if mind else None,
        'outside_room_idx': outside,
        'alignment': {'x_scale': X_SCALE, 'y_scale': round(Y_SCALE, 6)},
        'unsampled_strip_x_m': round(G.INNER_W - max(xs) * X_SCALE, 3),
    }


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pts = load(root=root)
    out = os.path.join(root, 'data/rfmap/measured.csv')
    write_csv(pts, out)
    rep = quality_report(pts)
    print('csv ->', out)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
