import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def rgb_to_y(rgb):
    """
    rgb: Tensor [1,3,H,W] in [0,1]
    returns: Tensor [H,W] in [0,255]
    """
    r, g, b = rgb[0]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return (y * 255.0).cpu().numpy()

def psnr_y(y_true, y_pred):
    return peak_signal_noise_ratio(y_true, y_pred, data_range=255)

def ssim_y(y_true, y_pred):
    return structural_similarity(
        y_true, y_pred, data_range=255
    )