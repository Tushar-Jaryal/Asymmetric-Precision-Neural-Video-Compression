import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER CONFIG
# =========================
OLD_DIR = "./DCVC_OLD/out_bin/HEVC_B"
NEW_DIR = "./DCVC_NEW/DCVC/out_bin/HEVC_B"

FPS_TABLE = {
    "BasketballDrive": 50,
    "BQTerrace": 60,
    "Cactus": 50,
    "Kimono": 24,
    "ParkScene": 24,
}

FRAME_COUNT = {
    "BasketballDrive": 8320,
    "BQTerrace": 8320,
    "Cactus": 8320,
    "Kimono": 8320,
    "ParkScene": 8320,
}

QP_PATTERN = re.compile(r"_q(\d+)\.bin")


# =========================
# HELPERS
# =========================
def parse_filename(fname):
    qp = int(QP_PATTERN.search(fname).group(1))
    seq = fname.split("_")[0]
    return seq, qp


def bitrate(bytes_, seq):
    fps = FPS_TABLE[seq]
    frames = FRAME_COUNT[seq]
    return bytes_ * 8 * fps / frames / 1000  # kbps


# =========================
# COLLECT DATA
# =========================
rows = []

for fname in os.listdir(OLD_DIR):
    if not fname.endswith(".bin"):
        continue

    old_path = os.path.join(OLD_DIR, fname)
    new_path = os.path.join(NEW_DIR, fname)

    if not os.path.exists(new_path):
        print(f"⚠ Missing in NEW: {fname}")
        continue

    seq, qp = parse_filename(fname)

    old_size = os.path.getsize(old_path)
    new_size = os.path.getsize(new_path)

    old_kbps = bitrate(old_size, seq)
    new_kbps = bitrate(new_size, seq)
    delta_pct = 100 * (new_kbps - old_kbps) / old_kbps

    rows.append([
        seq, qp,
        old_kbps, new_kbps,
        delta_pct
    ])

df = pd.DataFrame(
    rows,
    columns=["Sequence", "QP", "OLD_kbps", "LSQ_kbps", "Delta_%"]
)

print("\n=== Bitrate Comparison Table ===")
print(df.to_string(index=False))


# =========================
# PLOTTING
# =========================
plt.figure(figsize=(10, 6))

for seq in df["Sequence"].unique():
    sub = df[df["Sequence"] == seq].sort_values("QP")
    plt.plot(
        sub["QP"],
        sub["Delta_%"],
        marker="o",
        label=seq
    )

plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("QP")
plt.ylabel("Bitrate Change (%)  (LSQ vs OLD)")
plt.title("DCVC-LSQ Bitrate Difference")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Bitrate Comparison LSQ DCVC.png')
plt.close()