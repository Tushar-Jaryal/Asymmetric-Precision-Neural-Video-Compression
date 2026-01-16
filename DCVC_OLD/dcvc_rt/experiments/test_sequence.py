from dcvc_rt.data.yuv_sequence import YUVSequence
from dcvc_rt.codec.codec import DCVCCodec
from dcvc_rt.core.models import load_model
from dcvc_rt.runner.sequence_runner import SequenceRunner

def main():
    model = load_model(
        "./cvpr2025_video.pth.tar",
        device="cuda",
    )
    codec = DCVCCodec(model, qp=32)
    seq = YUVSequence(
        path="/home/ollama/ollama_hdd/VP/DCVC/dcvc_rt/experiments/Kimono_1920x1080_24.yuv",
        width=256,
        height=256,
        num_frames=10,
        device="cuda",
    )

    runner = SequenceRunner(codec, gop_size=8)
    result = runner.run(seq)

    print("Frames:", result["num_frames"])
    print("Avg bpp:", result["avg_bpp"])
    print("Avg psnr:", result["avg_psnr"])

    seq.close()

if __name__ == "__main__":
    main()