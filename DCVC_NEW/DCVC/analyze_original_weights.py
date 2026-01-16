import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def quantize_naive(weight, n_bits=8):
    """
    Simulate Naive INT8 Quantization using Min-Max scaling.
    s = max(abs(weight)) / (2^(n-1) - 1)
    """
    # 1. Calculate Naive Scale (Min-Max)
    max_val = torch.max(torch.abs(weight))
    s_naive = max_val / (2**(n_bits-1) - 1)
    
    # Avoid divide by zero for empty/zero layers
    if s_naive == 0: s_naive = 1.0

    Qn = -(2**(n_bits-1))
    Qp = 2**(n_bits-1) - 1
    
    # 2. Scale
    w_scaled = weight / s_naive
    # 3. Clamp and Round
    w_int = torch.clamp(torch.round(w_scaled), Qn, Qp)
    # 4. Dequantize
    w_recon = w_int * s_naive
    
    return w_recon, s_naive.item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to Original FP32 .pth")
    parser.add_argument('--output_dir', type=str, default="original_analysis_plots")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading original model: {args.model_path}...")
    ckpt = torch.load(args.model_path, map_location='cpu')
    if 'state_dict' in ckpt: ckpt = ckpt['state_dict']

    s_values = []
    layer_names = []
    quant_errors = []

    print("Analyzing layers...")
    
    # Iterate through all weights in the checkpoint
    for key, val in ckpt.items():
        # We only care about Conv2d or Linear weights (ignore biases/batchnorm)
        if ".weight" in key and val.dim() >= 2:
            # Filter out tiny scalars or non-weight parameters
            if val.numel() < 100: continue
            
            w_fp32 = val.float()
            
            # Perform Naive Quantization simulation
            w_recon, s_naive = quantize_naive(w_fp32)
            
            # Calculate MSE Error
            mse = torch.mean((w_fp32 - w_recon) ** 2).item()
            
            s_values.append(s_naive)
            layer_names.append(key.replace(".weight", ""))
            quant_errors.append(mse)

    # --- PLOT 1: Dynamic Range (The "Naive" Step Sizes) ---
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(s_values)), s_values, color='gray', alpha=0.7)
    plt.yscale('log')
    plt.title("Weight Dynamic Range per Layer (Naive Step Size)")
    plt.xlabel("Layer Index")
    plt.ylabel("Step Size (s = Max/127)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(args.output_dir, "1_original_dynamic_range.png"))
    plt.close()
    print(f"Generated Plot 1: Analyzed {len(s_values)} layers.")

    # --- PLOT 2: Naive Quantization Error ---
    plt.figure(figsize=(12, 6))
    plt.plot(quant_errors, 'k--o', linewidth=1, markersize=3, label="Naive Quantization")
    plt.title("Naive Quantization Error (Standard PTQ)")
    plt.xlabel("Layer Index")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "2_original_quant_error.png"))
    plt.close()
    print("Generated Plot 2.")

    # --- PLOT 3: Visualization of One Representative Layer ---
    # Pick the median layer again for fair comparison
    mid_idx = np.argsort(s_values)[len(s_values)//2]
    target_layer = layer_names[mid_idx]
    s_target = s_values[mid_idx]
    w_target = ckpt[target_layer + ".weight"].flatten().numpy()
    
    # Naive Grid: Uniformly spaced based on max value
    grid_points = np.arange(-128, 127) * s_target
    
    plt.figure(figsize=(10, 6))
    plt.hist(w_target, bins=100, color='gray', alpha=0.5, label='Original FP32 Weights', density=True)
    
    # Plot vertical lines for the Naive grid
    for i, gp in enumerate(grid_points):
        if i % 10 == 0: 
            plt.axvline(gp, color='black', alpha=0.3, linewidth=0.5, linestyle='--')
            
    plt.xlim(np.percentile(w_target, 1), np.percentile(w_target, 99))
    plt.title(f"Weight Distribution vs Naive Grid (No Training)\nLayer: {target_layer}")
    plt.xlabel("Weight Value")
    plt.legend(["Naive Grid (Max/127)", "FP32 Distribution"])
    plt.savefig(os.path.join(args.output_dir, "3_original_layer_vis.png"))
    plt.close()
    print(f"Generated Plot 3 for layer: {target_layer}")

if __name__ == "__main__":
    main()