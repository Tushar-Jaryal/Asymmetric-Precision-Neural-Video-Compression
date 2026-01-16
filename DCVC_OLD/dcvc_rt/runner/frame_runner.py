import torch
import numpy as np
from dcvc_rt.utils.image import crop_to_shape
from dcvc_rt.utils.metrics import rgb_to_y, psnr_y, ssim_y

class FrameRunner:
    def __init__(self, codec, width, height):
        self.codec = codec
        self.W = width
        self.H = height
        self.results = []

    def run(self, frames):
        self.results = []
        for idx, frame in enumerate(frames):
            bits, recon = self.codec.encode_decode(frame)
            bpp = bits / (self.W * self.H)

            y_true = rgb_to_y(frame)
            y_rec = rgb_to_y(recon)

            y_rec = crop_to_shape(y_rec, self.H, self.W)
            
            psnr_Y = psnr_y(y_true, y_rec)
            ssim_Y = ssim_y(y_true, y_rec)
            
            self.results.append({
                "frame": idx,
                "bits": bits,
                "bpp": bpp,
                "psnr_y": psnr_Y,
                "ssim_y": ssim_Y
            })
        return self.results