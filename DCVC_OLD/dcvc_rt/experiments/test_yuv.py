from dcvc_rt.data.yuv_sequence import YUVSequence

def main():
    seq = YUVSequence(
        path = "/home/ollama/ollama_hdd/VP/DCVC/dcvc_rt/experiments/Kimono_1920x1080_24.yuv",
        width=1920,
        height=1080,
        num_frames=240,
        device="cuda",
    )

    for i in range(len(seq)):
        frame = seq.read_frame()
        print(i, frame.shape, frame.device)

    seq.close()

if __name__ == "__main__":
    main()