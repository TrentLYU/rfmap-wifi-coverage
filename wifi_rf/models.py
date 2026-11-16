"""五个网络方案（与已有报告完全一致的命名与结构思路）。

方案一  AE          卷积补全自编码器（无跳跃连接）
方案二  UNet        小型 U-Net（3 通道输入）
方案三  ResUNet     IDW 先验 + 残差 U-Net（4 通道）
方案四  UNetScatter 与方案二结构相同，训练数据换成散落布点（配对消融）
方案五  PGMPR       物理引导多先验残差 U-Net（7 通道 + 观测加权损失 + TTA）

不使用 BatchNorm / Dropout：归一化只保留输入/输出各一次线性缩放，
这样权重可以直接导到浏览器做纯 JS 前向。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class ConvAE(nn.Module):
    """方案一：无跳跃连接的编码-解码补全网络。"""

    def __init__(self, cin: int = 3, base: int = C.BASE_CH):
        super().__init__()
        b = base
        self.enc = nn.Sequential(
            conv_block(cin, b), nn.MaxPool2d(2),
            conv_block(b, b * 2), nn.MaxPool2d(2),
            conv_block(b * 2, b * 4), nn.MaxPool2d(2),
            conv_block(b * 4, b * 4),
        )
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"), conv_block(b * 4, b * 2),
            nn.Upsample(scale_factor=2, mode="nearest"), conv_block(b * 2, b),
            nn.Upsample(scale_factor=2, mode="nearest"), conv_block(b, b),
            nn.Conv2d(b, 1, 1),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class UNet(nn.Module):
    """方案二 / 方案三 / 方案四 / 方案五共用的 U-Net 主干。

    prior_idx: 若给出，则该通道是插值先验，网络输出的是相对先验的残差。
    """

    def __init__(self, cin: int = 3, base: int = C.BASE_CH, prior_idx: int | None = None):
        super().__init__()
        b = base
        self.prior_idx = prior_idx
        self.e1 = conv_block(cin, b)
        self.e2 = conv_block(b, b * 2)
        self.e3 = conv_block(b * 2, b * 4)
        self.bott = conv_block(b * 4, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2)
        self.d3 = conv_block(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2)
        self.d2 = conv_block(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2)
        self.d1 = conv_block(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d = self.d3(torch.cat([self.u3(b), e3], 1))
        d = self.d2(torch.cat([self.u2(d), e2], 1))
        d = self.d1(torch.cat([self.u1(d), e1], 1))
        y = self.out(d)
        if self.prior_idx is not None:
            y = y + x[:, self.prior_idx:self.prior_idx + 1]
        return y


def build(name: str) -> nn.Module:
    if name == "ae":
        return ConvAE(3)
    if name in ("unet", "unet_scatter"):
        return UNet(3)
    if name == "resunet":
        return UNet(4, prior_idx=3)
    if name == "pgmpr":
        return UNet(7, prior_idx=6)
    raise ValueError(name)


def inputs_for(name: str, x7) -> torch.Tensor:
    """从 7 通道数组里取出该方案实际使用的通道。"""
    if name in ("ae", "unet", "unet_scatter"):
        return x7[:, :3]
    if name == "resunet":
        return torch.cat([x7[:, :3], x7[:, 6:7]], 1)
    if name == "pgmpr":
        return x7
    raise ValueError(name)
