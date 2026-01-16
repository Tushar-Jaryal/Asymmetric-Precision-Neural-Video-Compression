import torch
import torch.multiprocessing as mp
from dcvc_rt.codec.elic_codec import ELICCodec
from dcvc_rt.core.models import load_models
from dcvc_rt.codec.codec import DCVCCodec
from dcvc_rt.runner.frame_runner import FrameRunner
from dcvc_rt.utils.video import load_yuv420_as_tensor

device = "cuda"
mp.set_start_method("fork", force=True)

# Load video frames as [N,1,3,H,W]
frames = load_yuv420_as_tensor(
    "./dcvc_rt/experiments/Kimono_1920x1080_24.yuv",
    width=1920,
    height=1080,
    frames=30,
    device=device
)

# ---- ELIC ----
elic = ELICCodec(quality=4, device=device)
elic_runner = FrameRunner(elic, 1920, 1080)
elic_results = elic_runner.run(frames)

# ---- DCVC ----
dcvc_model = load_models(
        model_path_i = "cvpr2025_image.pth.tar",
        model_path_p = "cvpr2025_video.pth.tar",
        device = "cuda"
    )
dcvc = DCVCCodec(dcvc_model, qp=32)
dcvc_runner = FrameRunner(dcvc, 1920, 1080)
dcvc_results = dcvc_runner.run(frames)

# Print comparison
for e, d in zip(elic_results, dcvc_results):
    print(
        f"Frame {e['frame']:02d} | "
        f"ELIC bpp={e['bpp']:.4f} PSNR={e['psnr']:.2f} | "
        f"DCVC bpp={d['bpp']:.4f} PSNR={d['psnr']:.2f}"
    )