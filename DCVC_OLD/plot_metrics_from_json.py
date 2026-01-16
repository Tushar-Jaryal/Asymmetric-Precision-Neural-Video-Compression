import os
import json
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# ----------- CONFIG ----------
JSON_DIR = "./out_bin/HEVC_B/"
PLOT_DIR = "./plots/"
METRIC = "psnr"
# -----------------------------

os.makedirs(PLOT_DIR, exist_ok=True)
pattern = re.compile(r"(.*)_q(\d+)\.json")
data = defaultdict(list)
for path in glob.glob(os.path.join(JSON_DIR, "*.json")):
    fname = os.path.basename(path)
    m = pattern.match(fname)
    if not m:
        continue

    seq_name, qp = m.group(1), int(m.group(2))
    with open(path, "r") as f:
        js = json.load(f)

    bpp = js["ave_all_frame_bpp"]
    if METRIC == "psnr":
        val = js["ave_all_frame_psnr"]
    elif METRIC == "psnr_y":
        val = js["ave_all_frame_psnr_y"]
    elif METRIC == "psnr_u":
        val = js["ave_all_frame_psnr_u"]
    elif METRIC == "psnr_v":
        val = js["ave_all_frame_psnr_v"]
    else:
        raise ValueError("Unknown metric")

    data[seq_name].append((bpp, val, qp))

# ---------- PLOTTING ----------
for seq, points in data.items():
    points = sorted(points, key=lambda x: x[2])
    bpps = [p[0] for p in points]
    vals = [p[1] for p in points]
    qps = [p[2] for p in points]

    plt.figure(figsize=(6, 4))
    plt.plot(bpps, vals, marker="o")
    for x, y, qp in points:
        plt.text(x, y, f"QP{qp}", fontsize=8)

    plt.xlabel("Bits per Pixel (bpp)")
    plt.ylabel(METRIC.upper())
    plt.title(seq)
    plt.grid(True)
    out_path = os.path.join(PLOT_DIR, f"{seq}_{METRIC}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")