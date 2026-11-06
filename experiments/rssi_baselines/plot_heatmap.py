"""Plot source points and two fitted-model fields using the same color scale."""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=root/"examples"/"synthetic_rssi.csv")
args = parser.parse_args()
out = root/"optimized"
s = json.loads((out/"summary.json").read_text(encoding="utf-8"))
if hashlib.sha256(args.input.read_bytes()).hexdigest() != s["source_sha256"]:
    parser.error("This input does not match the fitted results. Run optimize_wifi.py first.")
points = pd.read_csv(args.input)
grid = pd.read_csv(out/"heatmap_grid.csv")
x, y, support = (grid[k].to_numpy().reshape(s["grid_shape"])
                 for k in ("x_m", "y_m", "inside_sample_hull"))
keys = ["formula", "physics_mlp"]
actual = points.rssi_dbm
scale = dict(cmap="RdYlBu_r", vmin=actual.min()-1, vmax=actual.max()+1)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
im = axes[0].scatter(points.y_m, points.x_m, c=actual, marker="s", s=60, **scale)
demo = args.input.resolve() == (root/"examples"/"synthetic_rssi.csv").resolve()
axes[0].set_title("Synthetic input points" if demo else "Input points")
for ax, key, title in zip(axes[1:], keys, ["Propagation model", "Propagation + residual MLP"]):
    field = grid[key+"_dbm"].to_numpy().reshape(s["grid_shape"])
    ax.pcolormesh(y, x, np.where(support, field, np.nan), shading="auto", **scale)
    ax.set_title(title)
for ax in axes:
    rx, ry = s["router_xy"]
    ax.scatter(ry, rx, marker="*", color="#162334", edgecolors="white", s=120)
    ax.set(xlim=(x.max()+.2, -.2), ylim=(-.2, y.max()+.2), xlabel="y (m)", ylabel="x (m)", aspect="equal")
fig.colorbar(im, ax=axes, label="RSSI (dBm)")
fig.suptitle("SIMULATED DATA | Pipeline demonstration" if demo else "Local input | Fitted heatmaps")
fig.savefig(out/"heatmap_demo.png", dpi=160)
