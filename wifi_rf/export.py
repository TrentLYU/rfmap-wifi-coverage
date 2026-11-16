"""导出前端资源：权重（float16 → base64）、实测点、阈值、指标。"""
from __future__ import annotations

import base64
import json
import os

import numpy as np
import torch

from . import config as C
from . import models as M
from .realdata import load_real

OUT = os.path.join(C.OUT_DIR, "app_data.json")


def f16_b64(a: np.ndarray) -> str:
    """float32 → little-endian float16 → base64（浏览器端用 DataView 解）。"""
    return base64.b64encode(a.astype("<f2").tobytes()).decode("ascii")


def export_weights(name: str = "pgmpr") -> dict:
    model = M.build(name)
    model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, name, "weights.pt"), map_location="cpu"))
    model.eval()
    out = {}
    for k, v in model.state_dict().items():
        arr = v.detach().cpu().numpy().astype(np.float32)
        out[k] = {"shape": list(arr.shape), "b64": f16_b64(arr)}
    return out


def main():
    real = load_real()
    valid = real["valid"]
    xy = real["xy"][valid]
    rssi = real["rssi"][valid]
    idx_valid = np.where(valid)[0]

    metrics = json.load(open(os.path.join(C.OUT_DIR, "metrics.json"), encoding="utf-8"))
    by_key = {r["key"]: r for r in metrics["results"]}

    payload = {
        "grid": {"n": C.GRID_N, "cell": C.CELL, "domain": C.DOMAIN},
        "room": [C.ROOM_W, C.ROOM_H],
        "ap": [float(v) for v in real["ap"]],
        "physics": {"pl0": C.PL0, "n": C.NEXP, "power": 2.0, "eps": 0.05, "obsCap": 6.0},
        "norm": {"mean": -56.98366928100586, "std": 9.989950180053711},
        "thresholds": [float(t) for t in real["thresholds"]],
        "abs_thresholds": [-80.0, -70.0, -60.0, -50.0],
        "furniture": C.FURNITURE,
        "points": [{"x": float(p[0]), "y": float(p[1]), "rssi": float(v),
                    "test": bool(i in set(real["test_idx"].tolist())),
                    "idx": int(i)} for p, v, i in zip(xy, rssi, idx_valid)],
        "split": {"train": real["train_idx"].tolist(), "test": real["test_idx"].tolist()},
        "scorecard": {
            "deploy": {k: by_key["deploy"][k] for k in ("rmse", "mae", "level_acc", "level_acc_pm1", "binary_acc")},
            "pgmpr": {k: by_key["pgmpr"][k] for k in ("rmse", "mae", "level_acc", "level_acc_pm1", "binary_acc")},
            "pgmpr_tta": {k: by_key["pgmpr_tta"][k] for k in ("rmse", "mae", "level_acc", "level_acc_pm1", "binary_acc")},
            "n_test": len(real["test_idx"]),
        },
        "weights": export_weights("pgmpr"),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print("->", OUT, round(os.path.getsize(OUT) / 1024), "KB")
    print("tensors:", {k: v["shape"] for k, v in payload["weights"].items()})


if __name__ == "__main__":
    main()
