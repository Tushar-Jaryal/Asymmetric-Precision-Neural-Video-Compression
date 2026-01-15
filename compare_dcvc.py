import argparse
import json
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip_interpolate

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Compare DCVC-RT vs LSQ Models")
    parser.add_argument('--baseline_dir', type=str, required=True, help="Folder containing Original JSONs")
    parser.add_argument('--lsq_dir', type=str, required=True, help="Folder containing LSQ JSONs")
    parser.add_argument('--metric', type=str, default='psnr', choices=['psnr', 'msssim'], help="Metric to plot")
    parser.add_argument('--output', type=str, default='comparison_plots', help="Folder to save plots")
    return parser.parse_args()

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_fps_from_filename(filename):
    """
    Tries to extract FPS from filenames like 'Sequence_1920x1080_60.yuv_q22.json'
    Returns 30.0 if not found.
    """
    # Look for patterns like '_60.yuv', '_50fps', etc.
    match = re.search(r'_(\d+)\.yuv', filename)
    if match:
        return float(match.group(1))
    
    # Fallback checks
    if '50' in filename: return 50.0
    if '60' in filename: return 60.0
    if '24' in filename: return 24.0
    if '30' in filename: return 30.0
    
    return 30.0 # Default

def calculate_kbps(entry, filename):
    """
    Convert BPP -> kbps
    Formula: BPP * NumPixels * FPS / 1000
    """
    # 1. Try direct bitrate
    if entry.get('bit_rate'): return float(entry['bit_rate'])

    # 2. Calculate from BPP (Your JSON format)
    bpp = entry.get('ave_all_frame_bpp')
    pixels = entry.get('frame_pixel_num')
    
    if bpp is not None and pixels is not None:
        fps = get_fps_from_filename(filename)
        kbps = (bpp * pixels * fps) / 1000.0
        return kbps

    return 0.0

def get_quality(entry, metric_name):
    """
    Maps 'psnr' -> 'ave_all_frame_psnr' or 'ave_all_frame_psnr_y'
    """
    key_map = {
        'psnr': ['ave_all_frame_psnr', 'ave_all_frame_psnr_y', 'psnr'],
        'msssim': ['ave_all_frame_msssim', 'ave_all_frame_msssim_y', 'msssim']
    }
    
    potential_keys = key_map.get(metric_name, [metric_name])
    
    for k in potential_keys:
        val = entry.get(k)
        if val is not None and val > 0:
            if isinstance(val, list): return float(val[0])
            return float(val)
            
    return 0.0

def aggregate_folder(folder_path, metric_name):
    files = glob.glob(os.path.join(folder_path, "*.json"))
    grouped_data = {}
    print(f"Scanning {len(files)} files in {folder_path}...")

    for fpath in files:
        try:
            entry = load_json(fpath)
        except:
            continue

        filename = os.path.basename(fpath)
        
        # Identify Sequence Name (Remove _qXX suffix)
        # Expecting: SequenceName_WxH_FPS.yuv_qXX.json
        seq_name = filename.split('_q')[0]
        # Remove extension if present in seq_name
        if seq_name.endswith('.json'): seq_name = seq_name[:-5]

        if seq_name not in grouped_data:
            grouped_data[seq_name] = {'pairs': []}

        # Extract Data
        br = calculate_kbps(entry, filename)
        qual = get_quality(entry, metric_name)
        
        if br > 0 and qual > 0:
            grouped_data[seq_name]['pairs'].append((br, qual))

    # Convert to Numpy
    final_sequences = {}
    for seq, data in grouped_data.items():
        sorted_pairs = sorted(data['pairs'], key=lambda x: x[0])
        
        if len(sorted_pairs) < 2:
            print(f"  [Warn] Sequence '{seq}' has {len(sorted_pairs)} valid points (needs >=2). Skipping.")
            continue
            
        final_sequences[seq] = {
            'bitrate': np.array([x[0] for x in sorted_pairs]),
            'quality': np.array([x[1] for x in sorted_pairs])
        }
        
    return final_sequences

def calculate_bd_psnr(R1, PSNR1, R2, PSNR2):
    try:
        lR1 = np.log(R1)
        lR2 = np.log(R2)
        min_int = max(min(lR1), min(lR2))
        max_int = min(max(lR1), max(lR2))
        
        if min_int >= max_int: return 0.0

        x = np.linspace(min_int, max_int, num=100)
        y1 = pchip_interpolate(lR1, PSNR1, x)
        y2 = pchip_interpolate(lR2, PSNR2, x)
        
        avg1 = np.trapz(y1, x) / (max_int - min_int)
        avg2 = np.trapz(y2, x) / (max_int - min_int)
        return avg2 - avg1
    except:
        return 0.0

def main():
    args = parse_args()
    if not os.path.exists(args.output): os.makedirs(args.output)

    print("--- Loading Baseline ---")
    base_seqs = aggregate_folder(args.baseline_dir, args.metric)
    
    print("\n--- Loading LSQ ---")
    lsq_seqs = aggregate_folder(args.lsq_dir, args.metric)

    all_diffs = []
    print("\n--- Generating Plots ---")

    for seq_name, b_data in base_seqs.items():
        if seq_name not in lsq_seqs: continue

        l_data = lsq_seqs[seq_name]
        bd_diff = calculate_bd_psnr(b_data['bitrate'], b_data['quality'], l_data['bitrate'], l_data['quality'])
        all_diffs.append(bd_diff)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(b_data['bitrate'], b_data['quality'], 'o-', label='Original (FP32)')
        plt.plot(l_data['bitrate'], l_data['quality'], 'x--', label='LSQ (INT8)')
        
        plt.title(f'{seq_name}\nBD-PSNR Diff: {bd_diff:.4f} dB')
        plt.xlabel('Bitrate (kbps)')
        plt.ylabel(f'{args.metric.upper()} (dB)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        
        safe_name = seq_name.replace(' ', '_').replace('/', '_')
        plt.savefig(os.path.join(args.output, f"{safe_name}.png"))
        plt.close()
        print(f"Processed {seq_name}: {bd_diff:.4f} dB")

    if all_diffs:
        print("\n" + "="*40)
        print(f"Global Average BD-PSNR Change: {np.mean(all_diffs):.4f} dB")
        print("="*40)
    else:
        print("No valid overlapping sequences found.")

if __name__ == '__main__':
    main()