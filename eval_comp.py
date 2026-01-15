import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt

QP_RE = re.compile(r"_q(\d+)\.json$")


def load_results(folder):
    data = defaultdict(dict)

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue

        m = QP_RE.search(fname)
        if not m:
            continue

        qp = int(m.group(1))
        seq = fname.replace(m.group(0), "")

        with open(os.path.join(folder, fname)) as f:
            data[seq][qp] = json.load(f)

    return data


def plot_rd(seq, dcvc, lsq):
    qps = sorted(dcvc[seq].keys())

    bpp_dcvc = [dcvc[seq][q]["ave_all_frame_bpp"] for q in qps]
    psnr_dcvc = [dcvc[seq][q]["ave_all_frame_psnr"] for q in qps]

    bpp_lsq = [lsq[seq][q]["ave_all_frame_bpp"] for q in qps]
    psnr_lsq = [lsq[seq][q]["ave_all_frame_psnr"] for q in qps]

    plt.figure(figsize=(7,5))
    plt.plot(bpp_dcvc, psnr_dcvc, "o-", label="DCVC")
    plt.plot(bpp_lsq, psnr_lsq, "s--", label="DCVC + LSQ")

    for q, x, y in zip(qps, bpp_dcvc, psnr_dcvc):
        plt.text(x, y, f"QP{q}", fontsize=9)

    plt.xlabel("Bits per Pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title(seq)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{seq}_RD.png", dpi=300)
    plt.close()


def plot_iframe_pframe(seq, data):
    qps = sorted(data[seq].keys())
    bpp = [data[seq][q]["ave_all_frame_bpp"] for q in qps]

    plt.figure(figsize=(7,5))
    plt.plot(bpp, [data[seq][q]["ave_i_frame_psnr"] for q in qps],
             "o-", label="I-frame")
    plt.plot(bpp, [data[seq][q]["ave_p_frame_psnr"] for q in qps],
             "s--", label="P-frame")

    plt.xlabel("Bits per Pixel (bpp)")
    plt.ylabel("PSNR (dB)")
    plt.title(seq + " (I vs P)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{seq}_I_vs_P.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    dcvc = load_results("./results_dcvc")
    lsq = load_results("./results_lsq")

    for seq in dcvc:
        print(f"Plotting {seq}")
        plot_rd(seq, dcvc, lsq)
        #plot_iframe_pframe(seq, dcvc)
        plot_iframe_pframe(seq, lsq)