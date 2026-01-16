import torch
from dcvc_rt.core.models import load_models
from dcvc_rt.codec.codec import DCVCCodec

def main():
    model = load_models(
        model_path_p = "cvpr2025_video.pth.tar",
        device = "cuda"
    )
    codec = DCVCCodec(model, qp=32)
    dummy = torch.rand(1, 3, 256, 256)
    bits, recon = codec.encode_decode(dummy)

    print("Bits:", bits)
    print("Recon shape:", recon.shape)

if __name__ == "__main__":
    main()