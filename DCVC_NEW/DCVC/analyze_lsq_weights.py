import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def quantize_lsq(weight, s_w, n_bits=8):
    """Simulate LSQ Quantization: x_int = round(clamp(x/s, min, max))"""
    Qn = -(2**(n_bits-1))
    Qp = 2**(n_bits-1) - 1
    
    # 1. Scale
    w_scaled = weight / s_w
    # 2. Clamp and Round
    w_int = torch.clamp(torch.round(w_scaled), Qn, Qp)
    # 3. Dequantize (Reconstruct)
    w_recon = w_int * s_w
    return w_recon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True, help="Path to Original FP32 .pth")
    parser.add_argument('--lsq_model', type=str, required=True, help="Path to LSQ Epoch 100 .pth")
    parser.add_argument('--output_dir', type=str, default="lsq_analysis_plots")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading models...")
    # Load checkpoints directly (CPU is fine for analysis)
    base_ckpt = torch.load(args.base_model, map_location='cpu')
    lsq_ckpt = torch.load(args.lsq_model, map_location='cpu')
    
    # Handle state_dict wrappers if present
    if 'state_dict' in base_ckpt: base_ckpt = base_ckpt['state_dict']
    if 'state_dict' in lsq_ckpt: lsq_ckpt = lsq_ckpt['state_dict']

    s_w_values = []
    layer_names = []
    quant_errors = []

    print("Analyzing layers...")
    
    # Iterate through LSQ checkpoint to find learned scales
    for key, val in lsq_ckpt.items():
        if "s_w" in key:
            # Found a learned weight scale!
            layer_name = key.replace(".s_w", ".weight")
            base_key = layer_name # Usually matches, or adjust if needed
            
            if base_key in base_ckpt:
                s_w = val.item()
                w_fp32 = base_ckpt[base_key]
                
                # Calculate Reconstruction Error
                w_recon = quantize_lsq(w_fp32, s_w)
                mse = torch.mean((w_fp32 - w_recon) ** 2).item()
                
                s_w_values.append(s_w)
                layer_names.append(key.replace(".s_w", ""))
                quant_errors.append(mse)

    # --- PLOT 1: Learned Step Sizes (The "Sensitivity" of Layers) ---
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(s_w_values)), s_w_values, color='teal', alpha=0.7)
    plt.yscale('log')
    plt.title("Learned LSQ Step Sizes per Layer (Log Scale)")
    plt.xlabel("Layer Index")
    plt.ylabel("Step Size (s_w)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(args.output_dir, "1_learned_step_sizes.png"))
    plt.close()
    print(f"Generated Plot 1: {len(s_w_values)} layers analyzed.")

    # --- PLOT 2: Quantization Error per Layer ---
    plt.figure(figsize=(12, 6))
    plt.plot(quant_errors, 'r-o', linewidth=1, markersize=3)
    plt.title("Mean Squared Quantization Error per Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("MSE (FP32 vs INT8)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "2_quantization_error.png"))
    plt.close()
    print("Generated Plot 2.")

    # --- PLOT 3: Deep Dive into One Representative Layer ---
    # We pick the layer with the Median step size (typical layer)
    mid_idx = np.argsort(s_w_values)[len(s_w_values)//2]
    target_layer = layer_names[mid_idx]
    s_w_target = s_w_values[mid_idx]
    w_fp32_target = base_ckpt[target_layer + ".weight"].flatten().numpy()
    
    # Calculate INT8 Grid points
    grid_points = np.arange(-128, 127) * s_w_target
    
    plt.figure(figsize=(10, 6))
    # Plot histogram of weights
    plt.hist(w_fp32_target, bins=100, color='blue', alpha=0.5, label='Original FP32 Weights', density=True)
    # Plot vertical lines for the learned quantization grid
    for i, gp in enumerate(grid_points):
        if i % 10 == 0: # Only plot every 10th line to avoid clutter
            plt.axvline(gp, color='red', alpha=0.3, linewidth=0.5)
            
    plt.xlim(np.percentile(w_fp32_target, 1), np.percentile(w_fp32_target, 99))
    plt.title(f"Weight Distribution vs Learned INT8 Grid\nLayer: {target_layer}")
    plt.xlabel("Weight Value")
    plt.legend(["Learned INT8 Steps (Every 10th)", "FP32 Distribution"])
    plt.savefig(os.path.join(args.output_dir, "3_layer_visualization.png"))
    plt.close()
    print(f"Generated Plot 3 for layer: {target_layer}")

if __name__ == "__main__":
    main()