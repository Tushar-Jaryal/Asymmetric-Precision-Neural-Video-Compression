import os
import torch
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from skimage.metrics import structural_similarity as ssim

def read_yuv420(path, w, h, frames):
    y_size = w * h
    uv_size = (w//2) * (h//2)
    frame_size = y_size + 2 * uv_size

    y_list, u_list, v_list = [], [], []

    with open(path, "rb") as f:
        for _ in range(frames):
            y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape(h, w)
            u = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape(h//2, w//2)
            v = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape(h//2, w//2)
            
            y_list.append(y)
            u_list.append(u)
            v_list.append(v)
    return np.array(y_list), np.array(u_list), np.array(v_list)

def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 10 * math.log10(255 * 255 / mse)

def ssim_y(a, b):
    return ssim(a, b, data_range=255)

def ms_ssim_y(a,b):
    a = torch.tensor(a).unsqueeze(0).unsqueeze(0).float()
    b = torch.tensor(b).unsqueeze(0).unsqueeze(0).float()
    return ms_ssim(a, b, data_range=255).item()

def yuv420_to_rgb(y,u,v):
    h, w = y.shape
    u_up = cv2.resize(u, (w, h), interpolation=cv2.INTER_LINEAR)
    v_up = cv2.resize(v, (w, h), interpolation=cv2.INTER_LINEAR)

    yuv = np.stack([y, u_up, v_up], axis=-1).astype(np.unit8)
    return cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)

def save_comparison(orig, rec, idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    diff = np.abs(orig.astype(np.int16) - rec.astype(np.int16))
    
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(orig, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Reconstructed")
    plt.imshow(rec, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Difference")
    plt.imshow(diff, cmap="hot")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/frame_{idx:04d}.png")
    plt.close()

# Example
W, H, F = 1920, 1080, 50
orig_y, orig_u, orig_v = read_yuv420("./data/HEVC_B/BasketballDrive_1920x1080_50.yuv", W, H, F)
rec_y, rec_u, rec_v = read_yuv420("./out_bin/HEVC_B/HEVC_B/BasketballDrive_1920x1080_50_4764kbps.yuv_q63_4764kbps.yuv", W, H, F)

psnr_y, psnr_u, psnr_v =[], [], []
ssim_vals, mssim_vals =[], []

for i in range(F):
    psnr_y.append(psnr(orig_y[i], rec_y[i]))
    psnr_u.append(psnr(orig_u[i], rec_u[i]))
    psnr_v.append(psnr(orig_v[i], rec_v[i]))

    ssim_vals.append(ssim_y(orig_y[i], rec_y[i]))
    mssim_vals.append(ms_ssim_y(orig_y[i], rec_y[i]))

    if i < 5:
        save_comparison(orig_y[i], rec_y[i], i, "./vis")

print("===== DCVC-RT Evaluation =====")
print(f"Avg PSNR-Y : {np.mean(psnr_y):.3f} dB")
print(f"Avg PSNR-U : {np.mean(psnr_u):.3f} dB")
print(f"Avg PSNR-V : {np.mean(psnr_v):.3f} dB")
print(f"Avg SSIM-Y : {np.mean(ssim_vals):.4f}")
print(f"Avg MS-SSIM-Y : {np.mean(mssim_vals):.4f}")