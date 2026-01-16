import torch
import multiprocessing as mp
from dcvc_rt.core.models import load_model
from dcvc_rt.codec.codec import DCVCCodec
from dcvc_rt.runner.sequence_runner import SequenceRunner
from dcvc_rt.data.yuv_sequence import YUVSequence

def _run_one_sequence(args):
    """
    Worker function (runs in a separate process).
    """
    (
        seq_path,
        width,
        height,
        num_frames,
        ckpt_path,
        qp,
        gop_size,
        device,
    ) = args
    torch.cuda.set_device(device)
    model = load_model(ckpt_path, device=device)
    codec = DCVCCodec(model, qp=qp)
    sequence = YUVSequence(
        path=seq_path,
        width=width,
        height=height,
        num_frames=num_frames,
        device=device,
    )
    runner = SequenceRunner(codec, gop_size=gop_size)
    result = runner.run(sequence)
    sequence.close()

    return {
        "sequence": seq_path,
        "frames": result["num_frames"],
        "avg_bpp": result["avg_bpp"],
        "avg_psnr": result["avg_psnr"],
    }

class MultiprocessRunner:
    """
    Runs multiple sequences in parallel (process-level).
    """

    def __init__(
        self,
        ckpt_path,
        qp=32,
        gop_size=32,
        device="cuda:0",
        num_workers=1,
    ):
        self.ckpt_path = ckpt_path
        self.qp = qp
        self.gop_size = gop_size
        self.device = device
        self.num_workers = num_workers

    def run(self, sequences):
        """
        Args:
            sequences: list of dicts with keys:
                path, width, height, num_frames
        """
        ctx = mp.get_context("spawn")

        tasks = []
        for seq in sequences:
            tasks.append(
                (
                    seq["path"],
                    seq["width"],
                    seq["height"],
                    seq.get("num_frames"),
                    self.ckpt_path,
                    self.qp,
                    self.gop_size,
                    self.device,
                )
            )
        with ctx.Pool(self.num_workers) as pool:
            results = pool.map(_run_one_sequence, tasks)

        return results