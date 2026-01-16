from dcvc_rt.runner.mp_runner import MultiprocessRunner

def main():
    sequences = [
        {
            "path": "/home/ollama/ollama_hdd/VP/DCVC/dcvc_rt/experiments/Kimono_1920x1080_24.yuv",
            "width": 256,
            "height": 256,
            "num_frames": 30,
        },
        {
            "path": "/home/ollama/ollama_hdd/VP/DCVC/dcvc_rt/experiments/Kimono_1920x1080_24.yuv",
            "width": 256,
            "height": 256,
            "num_frames": 30,
        }
    ]
    runner = MultiprocessRunner(
        ckpt_path = "./cvpr2025_video.pth.tar",
        qp=32,
        gop_size=8,
        device="cuda:0",
        num_workers=2,
    )
    result = runner.run(sequences)
    for r in result:
        print(r)

if __name__ == "__main__":
    main()