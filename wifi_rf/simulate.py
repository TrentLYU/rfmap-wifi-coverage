"""物理仿真数据：随机户型 + 随机 AP + 对数距离损耗 + 障碍穿透 + 空间相关阴影场。

仿真只用于"预训练"：真实房间只有一张图，无法靠它学出跨户型的结构先验。
"""
from __future__ import annotations

import os

import numpy as np

from . import config as C
from .physics import cell_centers, obstacle_field, obs_density, dist_to_obs, idw_prior

CELL = C.CELL
N = C.GRID_N


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    o = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="reflect"), k, "valid"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode="reflect"), k, "valid"), 1, o)


def make_scene(rng: np.random.Generator, obs_mode: str = "mix") -> dict:
    w = float(rng.uniform(8.0, 14.0))
    h = float(rng.uniform(8.0, 12.0))
    X, Y = cell_centers(N, CELL)
    room = ((X <= w) & (Y <= h)).astype(np.float32)

    obstacles = []
    for _ in range(int(rng.integers(0, 4))):
        ow = float(rng.uniform(0.4, 2.6))
        oh = float(rng.uniform(0.3, 1.8))
        cx = float(rng.uniform(0.6, max(0.7, w - 0.6)))
        cy = float(rng.uniform(0.6, max(0.7, h - 0.6)))
        obstacles.append({"x": cx, "y": cy, "w": ow, "h": oh,
                          "db": float(rng.uniform(4.0, 14.0))})

    ap = (float(rng.uniform(0.8, max(0.9, w - 0.8))),
          float(rng.uniform(0.8, max(0.9, h - 0.8))))

    pl0 = float(rng.uniform(32.0, 46.0))
    n_exp = float(rng.uniform(1.8, 3.2))
    d = np.maximum(np.hypot(X - ap[0], Y - ap[1]), 1.0)
    base = pl0 + 10.0 * n_exp * np.log10(d)
    att = obstacle_field(ap, obstacles, N, CELL)

    sigma_cells = float(rng.uniform(1.5, 3.5))
    shadow_std = float(rng.uniform(1.5, 4.5))
    raw = rng.normal(0, 1, (N, N))
    sh = _blur(raw, sigma_cells)
    sh = sh / max(sh.std(), 1e-6) * shadow_std

    noise = rng.uniform(0.3, 1.2) * rng.normal(0, 1, (N, N))
    rssi = -(base + att) + sh + noise
    rssi = np.clip(rssi, -95.0, -25.0).astype(np.float32)
    rssi = np.where(room > 0, rssi, -120.0).astype(np.float32)

    # 观测点
    k = int(rng.integers(6, 21))
    pts = []
    if obs_mode in ("grid", "mix"):
        gw = max(2, int(round(np.sqrt(k * w / max(h, 1e-6)))))
        gh = max(2, int(np.ceil(k / gw)))
        gx = np.linspace(0.6, w - 0.6, gw)
        gy = np.linspace(0.6, h - 0.6, gh)
        for yy in gy:
            for xx in gx:
                pts.append((xx + rng.normal(0, 0.25), yy + rng.normal(0, 0.25)))
        rng.shuffle(pts)
        pts = pts[:k]
    if obs_mode == "scatter" or (obs_mode == "mix" and rng.random() < 0.6):
        pts = []
        for _ in range(k):
            if rng.random() < 0.35:  # 簇状聚集
                cx = float(rng.uniform(1.0, w - 1.0))
                cy = float(rng.uniform(1.0, h - 1.0))
                pts.append((cx + rng.normal(0, 0.5), cy + rng.normal(0, 0.5)))
            else:
                pts.append((float(rng.uniform(0.4, w - 0.4)), float(rng.uniform(0.4, h - 0.4))))

    obs = np.array(pts, dtype=np.float64)
    obs[:, 0] = np.clip(obs[:, 0], 0.15, max(0.2, w - 0.15))
    obs[:, 1] = np.clip(obs[:, 1], 0.15, max(0.2, h - 0.15))
    # 网格吸附（同一格只保留一个点）
    ix = np.clip((obs[:, 0] / CELL).astype(int), 0, N - 1)
    iy = np.clip((obs[:, 1] / CELL).astype(int), 0, N - 1)
    key = iy * N + ix
    _, uniq = np.unique(key, return_index=True)
    uniq = np.sort(uniq)
    ix, iy, obs = ix[uniq], iy[uniq], obs[uniq]

    obs_mask = np.zeros((N, N), dtype=np.float32)
    obs_mask[iy, ix] = 1.0
    obs_val = rssi[iy, ix].copy()
    obs_xy = np.stack([(ix + 0.5) * CELL, (iy + 0.5) * CELL], axis=1)

    # 物理先验通道：含障碍衰减的路径损耗场（与前端家具衰减同源）
    phys = (-(base + att)).astype(np.float32)
    phys = np.where(room > 0, phys, -120.0).astype(np.float32)

    return {"room": room, "target": rssi, "phys": phys, "obs_mask": obs_mask,
            "obs_val": obs_val, "obs_xy": obs_xy, "ap": ap}


def build_arrays(scenes: list[dict], mean: float, std: float):
    """把场景列表堆成 npz 数组。"""
    room = np.stack([s["room"] for s in scenes])
    target = np.stack([s["target"] for s in scenes])
    phys = np.stack([s["phys"] for s in scenes])
    om = np.stack([s["obs_mask"] for s in scenes])
    ov = np.zeros_like(om)
    for i, s in enumerate(scenes):
        ix = np.clip((s["obs_xy"][:, 0] / CELL).astype(int), 0, N - 1)
        iy = np.clip((s["obs_xy"][:, 1] / CELL).astype(int), 0, N - 1)
        ov[i, iy, ix] = s["obs_val"]
    x = np.zeros((len(scenes), 7, N, N), dtype=np.float32)
    x[:, 0] = np.where(om > 0, (ov - mean) / std, 0.0)
    x[:, 1] = om
    x[:, 2] = room
    x[:, 3] = np.where(room > 0, (phys - mean) / std, 0.0)
    x[:, 4] = np.stack([dist_to_obs(s["obs_xy"], N, CELL) for s in scenes])
    x[:, 5] = np.stack([obs_density(om[i], 2.0) for i in range(len(scenes))])
    x[:, 6] = np.stack([np.where(room[i] > 0, (idw_prior(s["obs_xy"], s["obs_val"], room[i], N, CELL) - mean) / std, 0.0)
                        for i, s in enumerate(scenes)])
    y = np.where(room > 0, (target - mean) / std, 0.0).astype(np.float32)
    return {"x": x, "y": y, "room": room, "target": target, "obs_mask": om, "obs_val": ov}


def make_dataset(n: int, seed: int, obs_mode: str = "mix") -> list[dict]:
    rng = np.random.default_rng(seed)
    return [make_scene(rng, obs_mode) for _ in range(n)]


def generate_all() -> dict:
    tr = make_dataset(C.N_TRAIN_SIM, C.SEED_SIM_TRAIN, "mix")
    va = make_dataset(C.N_VAL_SIM, C.SEED_SIM_VAL, "mix")
    te = make_dataset(C.N_TEST_SIM, C.SEED_SIM_TEST, "mix")

    # 统计量只从训练集真值（房间内）估计
    vals = np.concatenate([s["target"][s["room"] > 0] for s in tr])
    mean, std = float(vals.mean()), float(vals.std())
    meta = {"mean": mean, "std": std, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}

    packs = {"train": tr, "val": va, "test": te}
    for name, scenes in packs.items():
        arr = build_arrays(scenes, mean, std)
        np.savez(os.path.join(C.DATA_DIR, f"sim_{name}.npz"), **arr)

    # 散落布点版本：复用同一批户型与真值，只重抽观测点（方案四的配对消融）
    rng = np.random.default_rng(C.SEED_SIM_TRAIN + 500)
    sc_tr = make_dataset(C.N_TRAIN_SIM, C.SEED_SIM_TRAIN, "scatter")
    sc_va = make_dataset(C.N_VAL_SIM, C.SEED_SIM_VAL, "scatter")
    sc_te = make_dataset(C.N_TEST_SIM, C.SEED_SIM_TEST, "scatter")
    for name, scenes in (("train", sc_tr), ("val", sc_va), ("test", sc_te)):
        arr = build_arrays(scenes, mean, std)
        np.savez(os.path.join(C.DATA_DIR, f"sim_scatter_{name}.npz"), **arr)

    import json
    with open(os.path.join(C.DATA_DIR, "sim_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    m = generate_all()
    print(m)
    for name in ("train", "val", "test"):
        z = np.load(os.path.join(C.DATA_DIR, f"sim_{name}.npz"))
        print(name, {k: z[k].shape for k in z.files},
              "obs/场景=%.1f" % z["obs_mask"].sum((1, 2)).mean(),
              "真值域=[%.1f, %.1f]" % (z["target"][z["room"] > 0].min(), z["target"][z["room"] > 0].max()))
