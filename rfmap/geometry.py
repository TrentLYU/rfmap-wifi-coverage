"""房间几何定义。

数据来源
--------
尺寸全部取自 ``实验地户型.png``（Homestyler 导出的 3D 户型图）上的实测标注，
未作任何主观放大或美化。标注读数如下（单位 m）：

    外轮廓宽           13.123
    外轮廓高           12.357
    墙厚                0.240
    内部净宽           12.643      = 13.123 - 2 x 0.240
    内部净高           11.731      = 12.357 - 0.240 - 0.240 - 0.146
    底部墙体额外厚度    0.146
    顶部窗宽           10.991
    底部门宽            2.330
    结构柱宽            0.830      柱体伸入房间的深度
    结构柱高            1.010
    柱上沿距内顶        4.960      尺寸链 4.960 + 1.010 + 5.761 = 11.731
    柱下沿距内底        5.761
    左侧窗组(自上而下)  3.156 / 0.045 / 1.380 / 0.440 / 1.029 / 5.686
                                  尺寸链合计 11.736 ~= 11.731（图上四舍五入）
    套内面积          146.21 m^2

坐标系约定
----------
原点 ``(0, 0)`` = 房间**内部右下角**；

* ``x`` 轴正向指向户型图的**左侧**（与实测数据表 ``x轴（m）`` 增大方向一致）；
* ``y`` 轴正向**向上**。

即：站在原点上，x 增大向左、y 增大向上，符合现场采集时的方位约定。
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# 户型图标注尺寸（单位：米）
# --------------------------------------------------------------------------
OUTER_W = 13.123
OUTER_H = 12.357
WALL = 0.240
INNER_W = 12.643
INNER_H = 11.731
BASE_EXTRA = 0.146

TOP_WINDOW = 10.991
DOOR_W = 2.330

COL_W = 0.830
COL_H = 1.010
COL_TOP_OFFSET = 4.960
COL_BOTTOM_OFFSET = 5.761

LEFT_OPENING_CHAIN = (3.156, 0.045, 1.380, 0.440, 1.029, 5.686)

FLOOR_AREA = 146.21

# --------------------------------------------------------------------------
# 结构柱在内部坐标中的位置
# --------------------------------------------------------------------------
# 图上柱体标注在“内顶往下 4.960 起、高 1.010”，换算到 y 向上坐标系：
COL_Y0 = INNER_H - (COL_TOP_OFFSET + COL_H)   # 5.761
COL_Y1 = INNER_H - COL_TOP_OFFSET             # 6.771
# 实测数据中 (12, 6) 与 (12, 8) 两点无法布点、改用 (11.7, 6) / (11.7, 8.9) 替代，
# 说明柱体位于**实测 x 较大**的一侧（户型图左墙内侧）。柱体贴墙，深 COL_W。
COL_X1 = INNER_W
COL_X0 = INNER_W - COL_W                      # 11.813
COL_CENTER = ((COL_X0 + COL_X1) / 2.0, (COL_Y0 + COL_Y1) / 2.0)

DOOR_X0 = (INNER_W - DOOR_W) / 2.0
DOOR_X1 = (INNER_W + DOOR_W) / 2.0

TOP_WINDOW_X0 = (INNER_W - TOP_WINDOW) / 2.0
TOP_WINDOW_X1 = (INNER_W + TOP_WINDOW) / 2.0


def room_polygon() -> list[tuple[float, float]]:
    """返回房间有效区域多边形（逆时针，已挖去贴墙结构柱）。

    多边形 = 内部矩形 [0, INNER_W] x [0, INNER_H] 减去柱体
             [COL_X0, COL_X1] x [COL_Y0, COL_Y1]。
    """
    return [
        (0.0, 0.0),
        (0.0, INNER_H),
        (COL_X1, INNER_H),
        (COL_X1, COL_Y1),
        (COL_X0, COL_Y1),
        (COL_X0, COL_Y0),
        (COL_X1, COL_Y0),
        (COL_X1, 0.0),
    ]


def polygon_area(poly: list[tuple[float, float]]) -> float:
    """鞋带公式（返回正值）。"""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def net_area() -> float:
    """净面积 = 内部矩形 - 结构柱。"""
    return INNER_W * INNER_H - COL_W * COL_H


def inside_room(x: float, y: float, tol: float = 0.0) -> bool:
    """判断点是否落在房间有效区域内（含边界，柱体内部为 False）。"""
    if not (-tol <= x <= INNER_W + tol and -tol <= y <= INNER_H + tol):
        return False
    if COL_X0 - tol < x and COL_Y0 - tol < y < COL_Y1 + tol:
        return False
    return True


def distance_to_walls(x: float, y: float) -> float:
    """点到最近墙面的距离（用于可视化与安全检查）。"""
    d = min(x, INNER_W - x, y, INNER_H - y)
    # 柱体视为障碍
    dx = max(COL_X0 - x, x - COL_X1, 0.0)
    dy = max(COL_Y0 - y, y - COL_Y1, 0.0)
    dcol = math.hypot(dx, dy)
    return min(d, dcol)


def summary() -> dict:
    """可供报告与自检使用的几何摘要。"""
    poly = room_polygon()
    return {
        'outer_w': OUTER_W,
        'outer_h': OUTER_H,
        'inner_w': INNER_W,
        'inner_h': INNER_H,
        'wall': WALL,
        'gross_area': INNER_W * INNER_H,
        'column': {'w': COL_W, 'h': COL_H, 'x0': COL_X0, 'x1': COL_X1,
                   'y0': COL_Y0, 'y1': COL_Y1, 'center': COL_CENTER},
        'polygon_area': polygon_area(poly),
        'net_area': net_area(),
        'floor_area_label': FLOOR_AREA,
        'door': {'w': DOOR_W, 'x0': DOOR_X0, 'x1': DOOR_X1},
        'top_window': {'w': TOP_WINDOW, 'x0': TOP_WINDOW_X0, 'x1': TOP_WINDOW_X1},
        'left_opening_chain': list(LEFT_OPENING_CHAIN),
    }


if __name__ == '__main__':
    import json
    print(json.dumps(summary(), indent=2, ensure_ascii=False))
    print('多边形面积 =', polygon_area(room_polygon()))
