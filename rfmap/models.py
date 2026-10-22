"""三个网络方案与插值基线（对应任务书「可尝试的三个网络方案」）。

任务书原文要点
--------------
* 统一使用 PyTorch，先跑通基础结构，再比较改进；
* 输入包括**稀疏 RSSI 图、观测位置标记图和房间有效区域图**；
* 未测位置只是占位值，不能被理解为零信号；
   —— 因此代码里所有观测值都会先乘上观测掩码（未观测格点严格置 0），
      并且在损失函数里用 masked loss，绝不让占位值参与梯度。
* 输出为同尺寸的单通道 RSSI 网格。

本文件提供
----------
===================  ============================================
方案                类
===================  ============================================
方案一 补全自编码器    :class:`CompletionAE`
方案二 小型 U-Net     :class:`SmallUNet`
方案三 插值引导残差    :class:`InterpResidualUNet`
===================  ============================================

以及三条**非神经网络**基线（用于回答"神经网络是否值得"）：
:func:`idw_predict` / :func:`nearest_predict` / :func:`rbf_predict`。

可移植性取舍
------------
为了让训练好的权重能在浏览器里用纯 JS 前向（本项目附带 Web 演示软件），
网络**不使用 BatchNorm / Dropout**：前者推理时是固定仿射变换、却要求权重与
统计量分开搬运，后者在推理时本来就要关掉。全部归一化只保留"输入 / 输出
各一次线性缩放"，由 :class:`Normalizer` 统一负责并随权重一起导出。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 输入通道：0=稀疏观测值, 1=观测掩码, 2=房间有效区掩码
N_CH_IN = 3
# 方案三额外拼接一路插值先验图
N_CH_IN_RESIDUAL = 4


# ==========================================================================
# 标准化
# ==========================================================================
class Normalizer:
    """只从训练集拟合、随权重一起保存的线性标准化参数。"""

    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    @classmethod
    def fit(cls, target: np.ndarray, room: np.ndarray) -> 'Normalizer':
        m = room > 0.5
        vals = target[m]
        return cls(float(vals.mean()), float(vals.std() + 1e-8))

    def encode(self, rssi_dbm):
        return (rssi_dbm - self.mean) / self.std

    def decode(self, z):
        return z * self.std + self.mean

    def to_dict(self) -> dict:
        return {'mean': self.mean, 'std': self.std}

    @classmethod
    def from_dict(cls, d: dict) -> 'Normalizer':
        return cls(d['mean'], d['std'])


# ==========================================================================
# 基础模块
# ==========================================================================
def conv3x3(cin: int, cout: int) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, padding=1, bias=True)


class DoubleConv(nn.Module):
    """Conv-ReLU 两次。刻意不含归一化层，见模块 docstring。"""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(cin, cout), nn.LeakyReLU(0.2, inplace=True),
            conv3x3(cout, cout), nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ==========================================================================
# 方案一：卷积补全自编码器
# ==========================================================================
class CompletionAE(nn.Module):
    """方案一 卷积补全自编码器。

    编码器把稀疏观测压缩到 1/8 分辨率的潜变量，解码器直接恢复整张地图。
    没有跳跃连接 —— 这正是它与 U-Net 的关键差别：空间细节只能靠潜变量带回来，
    因此对"观测点稀疏"更敏感。任务书要求它作为**可训练、可预测的基线**存在。

    Parameters
    ----------
    width : int
        基础通道数，逐级乘 2。
    """

    def __init__(self, width: int = 24, in_ch: int = N_CH_IN):
        super().__init__()
        w = width
        self.enc1 = DoubleConv(in_ch, w)
        self.enc2 = DoubleConv(w, w * 2)
        self.enc3 = DoubleConv(w * 2, w * 4)
        self.pool = nn.AvgPool2d(2)
        self.latent = DoubleConv(w * 4, w * 4)

        self.up3 = nn.Conv2d(w * 4, w * 4, 1)
        self.dec3 = DoubleConv(w * 4, w * 2)
        self.up2 = nn.Conv2d(w * 2, w * 2, 1)
        self.dec2 = DoubleConv(w * 2, w)
        self.up1 = nn.Conv2d(w, w, 1)
        self.dec1 = DoubleConv(w, w)
        self.out = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        z = self.latent(self.pool(e3))

        d = F.interpolate(self.up3(z), scale_factor=2, mode='bilinear',
                          align_corners=False)
        d = self.dec3(d)
        d = F.interpolate(self.up2(d), scale_factor=2, mode='bilinear',
                          align_corners=False)
        d = self.dec2(d)
        d = F.interpolate(self.up1(d), scale_factor=2, mode='bilinear',
                          align_corners=False)
        d = self.dec1(d)
        return self.out(d)


# ==========================================================================
# 方案二：小型 U-Net
# ==========================================================================
class SmallUNet(nn.Module):
    """方案二 小型 U-Net（任务书建议的主模型）。

    编码 / 解码之间用跳跃连接保留空间信息：室内 RSSI 的起伏主要来自
    墙体遮挡与多径，这些是**局部**结构，靠潜变量上下采样容易糊掉。
    """

    def __init__(self, width: int = 24, in_ch: int = N_CH_IN, out_ch: int = 1):
        super().__init__()
        w = width
        self.enc1 = DoubleConv(in_ch, w)
        self.enc2 = DoubleConv(w, w * 2)
        self.enc3 = DoubleConv(w * 2, w * 4)
        self.bottleneck = DoubleConv(w * 4, w * 8)
        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, 2, stride=2)
        self.dec3 = DoubleConv(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, stride=2)
        self.dec2 = DoubleConv(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, stride=2)
        self.dec1 = DoubleConv(w * 2, w)
        self.out = nn.Conv2d(w, out_ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d = self.up3(b)
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.up2(d)
        d = self.dec2(torch.cat([d, e2], dim=1))
        d = self.up1(d)
        d = self.dec1(torch.cat([d, e1], dim=1))
        return self.out(d)


# ==========================================================================
# 方案三：插值引导残差 U-Net
# ==========================================================================
class InterpResidualUNet(nn.Module):
    """方案三 插值引导残差 U-Net（U-Net 的工程变体，非某篇论文的原样复现）。

    结构：先用反距离加权给出一张**插值先验图** ``prior``，把它作为第 4 个输入
    通道；U-Net 不直接预测 RSSI，而是预测 ``真值 - prior`` 的残差，
    最终输出 ``prior + residual``。

    动机：插值擅长恢复大尺度趋势，网络擅长补局部结构。让网络只学"插值差在
    哪里"，比让它从零学会整个路径损耗场容易得多 —— 这也是任务书所谓
    "插值引导"的含义。
    """

    def __init__(self, width: int = 24):
        super().__init__()
        self.net = SmallUNet(width=width, in_ch=N_CH_IN_RESIDUAL, out_ch=1)

    def forward(self, x, prior):
        z = self.net(torch.cat([x, prior], dim=1))
        return prior + z  # 残差相加，输出仍是"标准化后的 RSSI"


# ==========================================================================
# 方案五：物理引导多先验残差 U-Net
# ==========================================================================
# 方案五的基础输入通道（先验图由 forward 单独拼接）
N_CH_IN_V5 = 6
# 到最近观测点距离的归一化尺度（m），与 IDW 的 max_radius 保持一致
V5_DIST_SCALE = 6.0
# 观测密度图的模糊尺度（格），1 格 = 0.2 m
V5_DENSITY_SIGMA = 2.0


def log_distance_trend(pts_xy: np.ndarray, pts_v: np.ndarray, grid_x, grid_y,
                       room_mask: np.ndarray, n_range=(1.2, 5.0)) -> np.ndarray:
    """用观测点拟合对数距离路损模型，返回**整图**的物理趋势（dBm）。

    与 :func:`gp_residual_predict` 共用同一套标定口径：AP 位置取观测最强点作代理，
    ``P0`` 与路损指数 ``n`` 由最小二乘拟合，``n`` 截断在 ``n_range`` 内。

    它和 IDW 先验是**互补**的：

    * IDW 在观测点处精确、远端趋于均值，是"局部可信"的先验；
    * 对数趋势是**全局**的单调衰减场，外推到没有观测点的区域时反而是更可靠的形状。

    方案五把两者同时交给网络，让网络自己决定在哪儿相信谁。
    """
    out = np.full(grid_x.shape, np.nan, dtype=np.float64)
    if len(pts_xy) == 0:
        return np.where(room_mask, -60.0, np.nan)

    k = int(np.argmax(pts_v))
    ax, ay = float(pts_xy[k, 0]), float(pts_xy[k, 1])
    dp = np.maximum(np.hypot(pts_xy[:, 0] - ax, pts_xy[:, 1] - ay), 1.0)
    L = np.log10(dp)
    A = np.stack([np.ones_like(L), -10.0 * L], axis=1)
    coef, *_ = np.linalg.lstsq(A, pts_v, rcond=None)
    P0 = float(coef[0])
    n = float(np.clip(coef[1], n_range[0], n_range[1]))

    d = np.maximum(np.hypot(grid_x - ax, grid_y - ay), 1.0)
    out = P0 - 10.0 * n * np.log10(d)
    out = np.clip(out, -100.0, -20.0)
    return np.where(room_mask, out, np.nan)


def nearest_obs_distance(obs_mask: np.ndarray, cell: float) -> np.ndarray:
    """每个格点到**最近观测点**的距离（m）；无观测点时返回 +inf 的有限替代。"""
    from scipy.ndimage import distance_transform_edt
    o = np.asarray(obs_mask) > 0.5
    if not o.any():
        return np.full(o.shape, 99.0, dtype=np.float64)
    return distance_transform_edt(~o) * float(cell)


def obs_density(obs_mask: np.ndarray, sigma: float = V5_DENSITY_SIGMA) -> np.ndarray:
    """观测点的空间密度（高斯模糊后的观测掩码，峰值归一到 1）。

    等价于告诉网络"这一小片区域有多少个已知点"，用来区分
    "孤零零一个点"和"好几米的密集测量区" —— 后者的局部趋势显然更可信。
    """
    from scipy.ndimage import gaussian_filter
    d = gaussian_filter(np.asarray(obs_mask, dtype=np.float64), sigma=sigma,
                        mode='constant')
    mx = float(d.max())
    return d / mx if mx > 1e-9 else d


def build_input_tensor_v5(obs_value: np.ndarray, obs_mask: np.ndarray,
                          room: np.ndarray, norm: 'Normalizer',
                          trend: np.ndarray, dist: np.ndarray,
                          density: np.ndarray) -> np.ndarray:
    """方案五的 6 通道输入张量。

    ======  ==========================================
    通道    含义
    ======  ==========================================
    0       标准化后的稀疏观测值（未观测严格为 0）
    1       观测掩码
    2       房间有效区掩码
    3       对数距离物理趋势（标准化）
    4       到最近观测点距离（归一化到 0~1）
    5       观测密度（高斯模糊后的掩码）
    ======  ==========================================

    第 7 个通道（IDW 先验）不在这里拼 —— 仍由模型 forward 内部拼接，
    与方案三保持同一约定，避免重复拼接导致通道数错位。
    """
    v = norm.encode(obs_value) * (obs_mask > 0.5)
    r = (room > 0.5).astype(np.float32)
    tr = norm.encode(np.where(np.isfinite(trend), trend, norm.mean)) * r
    dn = np.clip(dist / V5_DIST_SCALE, 0.0, 1.0).astype(np.float32)
    return np.stack([v, obs_mask, room, tr, dn, density], axis=0).astype(np.float32)


class PGMPRUNet(nn.Module):
    """方案五 · 物理引导多先验残差 U-Net（Physics-Guided Multi-Prior Residual UNet）。

    为什么这样设计 —— 每条改动都对应前面方案暴露出的一个具体短板
    -----------------------------------------------------------
    1. **残差基准仍用 IDW**：方案三已证明"学插值残差"比从零学整场容易得多
       （仿真 4.13 dB，优于 U-Net 的 4.25 dB），保留这个优势。
    2. **补进全局物理趋势**（通道 3）：方案三只有 IDW 一路先验，而 IDW 在没有
       观测点的角落会退化到全局均值 —— 方案二在角块外推上明显吃亏（5.0+ dB）
       正是这个原因。对数距离趋势虽然粗糙，但外推时形状是对的。
    3. **补进"到最近观测点距离"**（通道 4）：这是外推难度的**显式度量**。
       网络据此可以学到"离已知点越远越该相信物理趋势、越近越该相信观测"。
       实测数据显示散落布点会把该距离的 P95 从 2.71 m 拉到 3.90 m，
       也就是外推区显著扩大 —— 这个通道正是针对它。
    4. **补进观测密度**（通道 5）：区分孤立点与密集测量区，前者不该被过度信任。
    5. **观测加权损失**：见 :func:`rfmap.train.weighted_masked_mse`。

    结构上仍是小型 U-Net（与方案二同宽），参数量只增加输入层的那几个通道，
    这样"提升来自信息与损失设计，而不是堆参数"这一点是可验证的。
    """

    def __init__(self, width: int = 24):
        super().__init__()
        self.net = SmallUNet(width=width, in_ch=N_CH_IN_V5 + 1, out_ch=1)

    def forward(self, x, prior):
        return prior + self.net(torch.cat([x, prior], dim=1))


# ==========================================================================
# 工厂
# ==========================================================================
MODEL_ZOO = {
    'ae': CompletionAE,
    'unet': SmallUNet,
    'resunet': InterpResidualUNet,
    'unet_scatter': SmallUNet,
    'pgmpr': PGMPRUNet,
}

MODEL_LABELS = {
    'ae': '方案一 · 卷积补全自编码器',
    'unet': '方案二 · 小型 U-Net',
    'resunet': '方案三 · 插值引导残差 U-Net',
    'unet_scatter': '方案四 · 散落布点训练的 U-Net',
    'pgmpr': '方案五 · 物理引导多先验残差 U-Net',
}


def build_model(name: str, width: int = 24) -> nn.Module:
    if name not in MODEL_ZOO:
        raise KeyError(f'未知模型 {name}，可选 {sorted(MODEL_ZOO)}')
    return MODEL_ZOO[name](width=width)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ==========================================================================
# 非神经网络基线
# ==========================================================================
def idw_predict(pts_xy: np.ndarray, pts_v: np.ndarray, grid_x, grid_y,
                room_mask: np.ndarray, power: float = 2.0,
                max_radius: float = 6.0) -> np.ndarray:
    """反距离加权插值。

    每个格点取所有观测点的 ``1/d^power`` 加权平均；距离超过 ``max_radius``
    的观测点不再计入（否则远处只剩数值噪声）。完全没有可用观测时回退全局均值。
    """
    out = np.full(grid_x.shape, np.nan, dtype=np.float64)
    if len(pts_xy) == 0:
        return np.where(room_mask, -60.0, np.nan)

    d = np.sqrt((grid_x[..., None] - pts_xy[None, None, :, 0]) ** 2 +
                (grid_y[..., None] - pts_xy[None, None, :, 1]) ** 2)
    w = 1.0 / np.maximum(d, 1e-3) ** power
    w = np.where(d <= max_radius, w, 0.0)

    wsum = w.sum(axis=-1)
    ok = wsum > 0
    out = np.where(ok, (w * pts_v[None, None, :]).sum(axis=-1) /
                   np.maximum(wsum, 1e-12), np.nan)

    # 无可用观测的格点用全局均值兜底
    if not np.all(ok):
        out = np.where(ok, out, float(np.mean(pts_v)))
    out = np.where(room_mask, out, np.nan)
    return out


def nearest_predict(pts_xy: np.ndarray, pts_v: np.ndarray, grid_x, grid_y,
                    room_mask: np.ndarray) -> np.ndarray:
    """最近邻插值（最朴素的对照）。"""
    if len(pts_xy) == 0:
        return np.where(room_mask, -60.0, np.nan)
    d = np.sqrt((grid_x[..., None] - pts_xy[None, None, :, 0]) ** 2 +
                (grid_y[..., None] - pts_xy[None, None, :, 1]) ** 2)
    idx = d.argmin(axis=-1)
    out = pts_v[idx]
    return np.where(room_mask, out, np.nan)


def rbf_predict(pts_xy: np.ndarray, pts_v: np.ndarray, grid_x, grid_y,
                room_mask: np.ndarray, length_scale: float | None = None,
                reg: float = 1e-3) -> np.ndarray:
    """RBF（高斯核）插值，等价于零均值高斯过程的均值预测。

    核宽 ``length_scale`` 缺省用观测点的平均最近邻距离估计；核矩阵加对角正则
    保证数值稳定。
    """
    if len(pts_xy) == 0:
        return np.where(room_mask, -60.0, np.nan)

    n = len(pts_xy)
    if length_scale is None:
        d = np.sqrt(((pts_xy[:, None, :] - pts_xy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        length_scale = float(np.median(d.min(axis=1))) or 1.0

    K = np.exp(-((pts_xy[:, None, :] - pts_xy[None, :, :]) ** 2).sum(-1) /
               (2 * length_scale ** 2))
    K = K + reg * np.eye(n)
    try:
        alpha = np.linalg.solve(K, pts_v)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K, pts_v, rcond=None)[0]

    Ks = np.exp(-((grid_x[..., None] - pts_xy[None, None, :, 0]) ** 2 +
                  (grid_y[..., None] - pts_xy[None, None, :, 1]) ** 2) /
                (2 * length_scale ** 2))
    out = Ks @ alpha
    return np.where(room_mask, out, np.nan)


def gp_residual_predict(pts_xy: np.ndarray, pts_v: np.ndarray, grid_x, grid_y,
                        room_mask: np.ndarray, length_scale: float | None = None,
                        reg: float = 1e-3, n_range=(1.2, 5.0)) -> np.ndarray:
    """对数距离路损 + RBF 残差插值（本项目里最强的非神经网络基线）。

    公式：

    .. math::

        \\mathrm{PL}(d) = P_0 - 10\\,n\\log_{10}(d), \\quad d_0 = 1\\,\\mathrm{m}

    其中 :math:`P_0, n` 由输入点用最小二乘拟合，AP 位置取**观测最强点**做代理。
    残差 :math:`r = \\mathrm{RSSI} - \\mathrm{PL}(d)` 在空间上平滑，用 RBF 插值，
    最终预测 :math:`\\hat{y} = \\mathrm{PL}(d) + \\hat{r}`。

    .. note::

        为什么不直接用零均值 RBF（:func:`rbf_predict`）？
        因为它没有"距离越远越弱"的先验，在实测点之外会迅速衰减回 0，
        在本数据上 RMSE 高达 23~26 dB，作为基线并不公平。
        加上对数距离趋势后它才是真正有竞争力的对照方法 —— 这也正好回应
        任务书"与传统插值比较"的要求。
    """
    if len(pts_xy) == 0:
        return np.where(room_mask, -60.0, np.nan)

    k = int(np.argmax(pts_v))
    ax, ay = float(pts_xy[k, 0]), float(pts_xy[k, 1])

    dp = np.maximum(np.hypot(pts_xy[:, 0] - ax, pts_xy[:, 1] - ay), 1.0)
    L = np.log10(dp)
    A = np.stack([np.ones_like(L), -10.0 * L], axis=1)
    coef, *_ = np.linalg.lstsq(A, pts_v, rcond=None)
    P0 = float(coef[0])
    n = float(np.clip(coef[1], n_range[0], n_range[1]))

    trend_pts = P0 - 10.0 * n * L
    resid = pts_v - trend_pts

    ones = np.ones_like(room_mask, dtype=bool)
    r_hat = rbf_predict(pts_xy, resid, grid_x, grid_y, ones,
                        length_scale=length_scale, reg=reg)
    d = np.maximum(np.hypot(grid_x - ax, grid_y - ay), 1.0)
    trend = P0 - 10.0 * n * np.log10(d)
    out = trend + r_hat
    return np.where(room_mask, out, np.nan)


def fit_log_distance(pts_xy: np.ndarray, pts_v: np.ndarray,
                     n_range=(1.2, 5.0)) -> dict:
    """只做对数距离拟合，返回 P0 / n / AP 代理位置（供报告记录标定结果）。"""
    k = int(np.argmax(pts_v))
    ax, ay = float(pts_xy[k, 0]), float(pts_xy[k, 1])
    dp = np.maximum(np.hypot(pts_xy[:, 0] - ax, pts_xy[:, 1] - ay), 1.0)
    L = np.log10(dp)
    A = np.stack([np.ones_like(L), -10.0 * L], axis=1)
    coef, res, *_ = np.linalg.lstsq(A, pts_v, rcond=None)
    pred = A @ coef
    return {
        'ap_proxy': [round(ax, 3), round(ay, 3)],
        'P0_dbm': round(float(coef[0]), 3),
        'path_loss_exponent': round(float(np.clip(coef[1], *n_range)), 3),
        'residual_std_db': round(float((pts_v - pred).std()), 3),
    }


BASELINE_ZOO = {
    'nearest': nearest_predict,
    'idw': idw_predict,
    'rbf': rbf_predict,
    'gp': gp_residual_predict,
}

BASELINE_LABELS = {
    'nearest': '基线 · 最近邻',
    'idw': '基线 · 反距离加权 (IDW, p=2)',
    'rbf': '基线 · 零均值 RBF（对照，无趋势项）',
    'gp': '基线 · 对数距离趋势 + RBF 残差',
}


# ==========================================================================
# 打包成网络输入张量
# ==========================================================================
def prepare_prior(prior_dbm: np.ndarray, room: np.ndarray,
                  norm: 'Normalizer') -> np.ndarray:
    """把插值先验整理成可安全进网络的引导图（标准化后的 float32）。

    为什么要专门做这一步
    --------------------
    :func:`idw_predict` 对房间外的格点返回 ``NaN``。如果直接送去标准化，
    ``NaN`` 会一路传播成整张图 NaN，训练立刻发散（本项目实际踩过这个坑：
    resunet 前 30 个 epoch 的 loss 全是 NaN）。
    这里把非法值和房间外格点统一填成训练集均值 —— 标准化之后正好是 0，
    与"该位置无信息"的语义一致。
    """
    p = np.where(np.isfinite(prior_dbm), prior_dbm, norm.mean)
    p = np.where(room > 0.5, p, norm.mean)
    return norm.encode(p).astype(np.float32)


def build_input_tensor(obs_value: np.ndarray, obs_mask: np.ndarray,
                       room: np.ndarray, norm: Normalizer) -> np.ndarray:
    """按统一约定组网网络输入，返回 (3, ny, nx) float32。

    约定
    ----
    通道 0：标准化后的观测值 ``(v - mean)/std``，**未观测格点严格为 0**；
    通道 1：观测掩码；
    通道 2：房间有效区掩码。

    .. note::

        插值先验**不在这里拼接**。方案三需要的第 4 个通道由
        :class:`InterpResidualUNet` 内部拼接 —— 早期版本在两边各拼了一次，
        导致输入变成 5 通道、网络第一层报维度错误。把"谁负责拼 prior"
        固定为**模型内部**一个地方，避免这类重复拼接。
    """
    v = norm.encode(obs_value) * (obs_mask > 0.5)
    return np.stack([v, obs_mask, room], axis=0).astype(np.float32)


if __name__ == '__main__':
    # 注意：prior 通道一律由模型 forward 内部拼接，所以这里传给模型的 x
    # 只是"基础通道"（方案五 6 通道、其余 3 通道），不要提前把 prior 算进去。
    for name in MODEL_ZOO:
        m = build_model(name)
        n_in = N_CH_IN_V5 if name == 'pgmpr' else N_CH_IN
        x = torch.zeros(2, n_in, 64, 64)
        if name in ('pgmpr', 'resunet'):
            y = m(x, torch.zeros(2, 1, 64, 64))
        else:
            y = m(x)
        print(f'{name:14s} in_ch={n_in}+prior  params={count_params(m):7d}  out={tuple(y.shape)}')
