import torch
from compressai.zoo import cheng2020_attn

class ELICCodec:
    def __init__(self, quality=4, device="cuda"):
        self.device = device
        self.model = cheng2020_attn(
            quality = quality,
            pretrained = True
        ).to(device).eval()

    @torch.no_grad()
    def encode_decode(self, frame):
        """
        frame: Tensor [1, 3, H, W] in [0, 1]
        """
        out = self.model(frame)
        bits = 0.0
        for likelihood in out["likelihoods"].values():
            bits += torch.sum(-torch.log2(likelihood)).item()

        recon = out["x_hat"].clamp(0, 1)
        return bits, recon