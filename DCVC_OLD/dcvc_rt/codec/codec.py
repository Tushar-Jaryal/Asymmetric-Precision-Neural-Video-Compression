import torch

class DCVCCodec:
    """
    Clean single-process wrapper around DCVC's DMC model.

    Internally uses:
    - compress()
    - decompress()
    """

    def __init__(self, model, qp=32, force_zero_thres=None, device="cuda"):
        self.model = model
        self.qp = qp
        self.device = device
        self.force_zero_thres = force_zero_thres
        self._entropy_initialized = False
        self.is_first_frame = True
        self.reset()

    def reset(self):
        self.model.clear_dpb()
        self.model.set_curr_poc(0)
        if not self._entropy_initialized:
            self.model.update(force_zero_thres=self.force_zero_thres)
            self.model.set_use_two_entropy_coders(False)
            self._entropy_initialized = True
        #self.model.update(force_zero_thres=None)
        self.is_first_frame = True

    @torch.no_grad()
    def encode_decode(self, x, force_intra=False):
        """
        Encode + decode a single frame.

        Args:
            frame: torch.Tensor [1, 3, H, W] in [0,1]

        Returns:
            bits: int
            recon: torch.tensor
        """

        device = x.device
        qp = self.qp

        if force_intra or self.is_first_frame:
            self.model.clear_dpb()
            self.model.set_curr_poc(0)
            device = next(self.model.parameters()).device
            x = x.to(device)
            self.model.add_ref_frame(frame=x, feature=None, increase_poc=False)
            self.is_first_frame = False

        enc_out = self.model.compress(x, qp)
        bit_stream = enc_out["bit_stream"]
        H, W = x.shape[-2:]
        sps = {
            "height": H,
            "width": W,
            "ec_part": 1,
        }
        dec_out = self.model.decompress(bit_stream, sps, qp)
        x_hat = dec_out["x_hat"]
        bits = len(bit_stream) * 8
        return bits, x_hat