"""规则网格定义。

网络输入 / 输出统一使用 ``ny x nx`` 的二维栅格，栅格为**正方形**（cell=0.2 m），
网格覆盖范围略大于房间内部轮廓，超出部分用无效区掩码标出。

设计取舍
--------
* 取 0.2 m 间距：实测点最近间距约 2 m，0.2 m 栅格足以刻画室内阴影衰落的起伏
  （实测相关长度约 0.8~1.1 m），同时 64 x 64 的尺寸能被 U-Net 的 3 次下采样整除。
* 只保留**一个**网格配置并写入 ``grid.json``，训练 / 推理 / 前端三方共用同一份，
  避免"训练用 A 网格、推理用 B 网格"这类难以察觉的错位。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from . import geometry as G


@dataclass(frozen=True)
class Grid:
    x0: float = 0.0
    y0: float = 0.0
    cell: float = 0.2
    nx: int = 64
    ny: int = 64

    # ---------------- 基本属性 ----------------
    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def x_max(self) -> float:
        return self.x0 + self.nx * self.cell

    @property
    def y_max(self) -> float:
        return self.y0 + self.ny * self.cell

    def x_coords(self) -> np.ndarray:
        """各列中心的 x 坐标（长度 nx）。"""
        return self.x0 + (np.arange(self.nx) + 0.5) * self.cell

    def y_coords(self) -> np.ndarray:
        """各行中心的 y 坐标（长度 ny，索引 0 在下方）。"""
        return self.y0 + (np.arange(self.ny) + 0.5) * self.cell

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """返回 (X, Y) 两个 ny x nx 的坐标矩阵。"""
        xc = self.x_coords()
        yc = self.y_coords()
        return np.meshgrid(xc, yc)

    # ---------------- 坐标 <-> 索引 ----------------
    def xy_to_idx(self, x: float, y: float) -> tuple[int, int]:
        i = int(round((x - self.x0) / self.cell - 0.5))
        j = int(round((y - self.y0) / self.cell - 0.5))
        return j, i

    def idx_to_xy(self, j: int, i: int) -> tuple[float, float]:
        return (self.x0 + (i + 0.5) * self.cell,
                self.y0 + (j + 0.5) * self.cell)

    # ---------------- 掩码 ----------------
    def room_mask(self) -> np.ndarray:
        """房间有效区掩码，True = 房间内部（柱子内部为 False）。"""
        X, Y = self.centers()
        return np.vectorize(G.inside_room, otypes=[bool])(X, Y)

    def wall_distance(self) -> np.ndarray:
        """每格中心到最近墙 / 柱的距离（米）。"""
        X, Y = self.centers()
        return np.vectorize(G.distance_to_walls, otypes=[float])(X, Y)

    # ---------------- 序列化 ----------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({'x_max': self.x_max, 'y_max': self.y_max,
                  'room_w': G.INNER_W, 'room_h': G.INNER_H})
        return d

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'Grid':
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        return cls(x0=d['x0'], y0=d['y0'], cell=d['cell'],
                   nx=d['nx'], ny=d['ny'])


DEFAULT = Grid()


if __name__ == '__main__':
    g = DEFAULT
    m = g.room_mask()
    print(g.to_dict())
    print('mask true =', int(m.sum()), '/', m.size)
