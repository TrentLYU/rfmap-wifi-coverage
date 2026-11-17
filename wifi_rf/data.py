"""实测数据加载、质检、坐标规范化、70/30 划分与五级分档。"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config as C

# 质检阈值：单点 RSSI 若强于 -20 dBm，认为不是 Wi-Fi 客户端可读值（近场饱和/录入错误）
RSSI_VALID_MIN = -20.0


def load_measured() -> dict:
    """读取上传的 xlsx。

    表格首格 "11,12" 是 AP 坐标；表头其余为 x 坐标，首列为 y 坐标。
    单元格数值为 RSSI 的绝对值（单位 dBm），建模时取负号：RSSI = -value。
    空单元格表示该位置未测（x = 12.9 一整列在房间之外）。
    """
    import openpyxl

    wb = openpyxl.load_workbook(C.UPLOAD_XLSX, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))

    header = rows[0]
    xs = np.array([float(v) for v in header[1:] if v is not None], dtype=np.float64)
    pts = []
    for r in rows[1:]:
        if r[0] is None:
            continue
        y = float(r[0])
        for j, x in enumerate(xs):
            v = r[j + 1] if j + 1 < len(r) else None
            if v is None:
                continue
            pts.append((float(x), y, -float(v)))

    arr = np.array(pts, dtype=np.float64)
    return {"xy": arr[:, :2], "rssi": arr[:, 2],
            "ap": np.array(C.AP_POS, dtype=np.float64), "xs": xs}


def quintile_thresholds(rssi: np.ndarray) -> list[float]:
    """按 20/40/60/80 分位数给出 5 级边界（升序）。"""
    return [float(np.percentile(rssi, q)) for q in (20, 40, 60, 80)]


def level_of(rssi, thresholds: list[float]) -> np.ndarray:
    """RSSI → 等级 5(极好) … 1(很弱)；thresholds 升序，RSSI 越大等级越高。"""
    rssi = np.asarray(rssi, dtype=np.float64)
    lv = np.ones(rssi.shape, dtype=np.int32)
    for t in thresholds:
        lv += (rssi > t).astype(np.int32)
    return lv


def split(xy: np.ndarray, rssi: np.ndarray, valid: np.ndarray) -> dict:
    """70/30 划分：只在有效点上做，按强弱中位数分层，固定种子。"""
    from sklearn.model_selection import train_test_split

    idx = np.where(valid)[0]
    med = float(np.median(rssi[idx]))
    coarse = (rssi[idx] > med).astype(int)
    n_test = int(round(len(idx) * 0.30))
    tr, te = train_test_split(idx, test_size=n_test, random_state=C.SEED_SPLIT,
                              stratify=coarse)
    tr, te = np.sort(tr), np.sort(te)
    thr = quintile_thresholds(rssi[tr])
    return {"train_idx": tr, "test_idx": te, "thresholds": thr}


def save_all() -> dict:
    d = load_measured()
    xy, rssi = d["xy"], d["rssi"]
    valid = rssi <= RSSI_VALID_MIN  # 强于 -20 dBm 视为异常
    s = split(xy, rssi, valid)
    tr, te, thr = s["train_idx"], s["test_idx"], s["thresholds"]
    levels = level_of(rssi, thr)

    # 物理参数：对训练点做 log-distance 最小二乘拟合 RSSI = -(PL0 + 10·n·log10 d)
    dist = np.maximum(np.hypot(xy[tr, 0] - d["ap"][0], xy[tr, 1] - d["ap"][1]), 1.0)
    A = np.stack([np.ones_like(dist), 10 * np.log10(dist)], axis=1)
    coef, *_ = np.linalg.lstsq(A, rssi[tr], rcond=None)
    pl0, n_exp = -float(coef[0]), -float(coef[1])
    fit_rmse = float(np.sqrt(np.mean((rssi[tr] - (-(pl0 + 10 * n_exp * np.log10(dist)))) ** 2)))

    out = {
        "ap": [float(d["ap"][0]), float(d["ap"][1])],
        "room": [C.ROOM_W, C.ROOM_H],
        "n_points": int(len(rssi)),
        "n_valid": int(valid.sum()),
        "invalid_points": [int(i) for i in np.where(~valid)[0]],
        "thresholds": thr,
        "level_counts_valid": {str(k): int(((levels == k) & valid).sum()) for k in range(1, 6)},
        "split": {"train_idx": tr.tolist(), "test_idx": te.tolist()},
        "physical_fit": {"pl0": pl0, "n": n_exp, "rmse_db": fit_rmse},
        "rssi_stats_valid": {"min": float(rssi[valid].min()), "max": float(rssi[valid].max()),
                             "mean": float(rssi[valid].mean()), "std": float(rssi[valid].std())},
    }

    np.savez(os.path.join(C.DATA_DIR, "measured.npz"),
             xy=xy, rssi=rssi, ap=d["ap"], valid=valid,
             train_idx=tr, test_idx=te, thresholds=np.array(thr))
    with open(os.path.join(C.DATA_DIR, "measured.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with open(os.path.join(C.OUT_DIR, "measured.csv"), "w", encoding="utf-8-sig") as f:
        f.write("idx,x_m,y_m,rssi_dbm,quality,split,level\n")
        trs, tes = set(tr.tolist()), set(te.tolist())
        for i in range(len(rssi)):
            part = "train" if i in trs else ("test" if i in tes else "invalid")
            q = "ok" if valid[i] else "outlier"
            f.write(f"{i},{xy[i,0]:.2f},{xy[i,1]:.2f},{rssi[i]:.0f},{q},{part},{levels[i]}\n")

    return {**out, "data": {**d, "valid": valid, "levels": levels}}


if __name__ == "__main__":
    r = save_all()
    print(json.dumps({k: v for k, v in r.items() if k != "data"}, ensure_ascii=False, indent=2))
    d = r["data"]
    for i, (p, v) in enumerate(zip(d["xy"], d["rssi"])):
        tag = "train" if i in r["split"]["train_idx"] else ("test " if i in r["split"]["test_idx"] else "INVALID")
        print(f"{i:2d}  x={p[0]:5.1f} y={p[1]:5.1f}  rssi={v:6.1f}  {tag}  L{d['levels'][i]}")
