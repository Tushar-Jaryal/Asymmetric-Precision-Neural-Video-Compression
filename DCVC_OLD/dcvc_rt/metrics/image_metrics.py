import torch
import math
import torch.nn.functional as F

@torch.no_grad()
def psnr(x, y, data_range=1.0):
    """
    PSNR for RGB images.
    Args:
        x, y: [1, 3, H, W] tensors
    """
    mse = F.mse_loss(x, y)
    if mse == 0:
        return float("inf")
    return 10 * math.log10((data_range ** 2) / mse.item())

@torch.no_grad()
def bpp(bits, height, width):
    return bits / (height * width)