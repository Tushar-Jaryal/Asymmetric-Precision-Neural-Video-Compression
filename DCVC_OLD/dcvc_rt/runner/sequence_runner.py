import torch
import torch.nn.functional as F
from dcvc_rt.metrics.image_metrics import psnr, bpp

def pad_frame(x, multiple=64):
    """
    Pads frame to H,W multiple of `multiple`.
    Returns padded frame and original size.
    """

    _, _, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x_pad, (H, W)

def crop_frame(x, size):
    H, W = size
    return x[:, :, :H, :W]

class SequenceRunner:
    """
    Runs DCVC codec on a single video sequence.
    """

    def __init__(
        self,
        codec,
        gop_size=32,
        force_intra=False,
    ):
        self.codec = codec
        self.gop_size = gop_size
        self.force_intra = force_intra
        self.frame_idx = 0
        self.total_bits = 0
        self.recon_frames = []

    def run(self, sequence):
        """
        Args:
            sequence: YUVSequence
        Returns:
            dict with bits, recon frames
        """
        self.frame_idx = 0
        self.total_bits = 0
        self.recon_frames.clear()

        for _ in range(len(sequence)):
            frame = sequence.read_frame()
            is_intra = (
                self.force_intra
                or self.frame_idx == 0
                or self.frame_idx % self.gop_size == 0
            )
            frame_pad, orig_size = pad_frame(frame)
            bits, recon = self.codec.encode_decode(
                frame_pad,
                force_intra=is_intra,
            )
            recon = crop_frame(recon, orig_size)

            frame_bpp = bpp(bits, orig_size[0], orig_size[1])
            frame_psnr = psnr(frame, recon)

            self.total_bits += bits
            self.recon_frames.append(recon)

            if not hasattr(self, "metrics"):
                self.metrics = {
                    "bpp": [],
                    "psnr": [],
                }

            self.metrics["bpp"].append(frame_bpp)
            self.metrics["psnr"].append(frame_psnr)
            self.frame_idx += 1

        return {
            "total_bits": self.total_bits,
            "num_frames": self.frame_idx,
            "avg_bpp": sum(self.metrics["bpp"]) / len(self.metrics["bpp"]),
            "avg_psnr": sum(self.metrics["psnr"]) / len(self.metrics["psnr"]),
            "metrics": self.metrics,
        }