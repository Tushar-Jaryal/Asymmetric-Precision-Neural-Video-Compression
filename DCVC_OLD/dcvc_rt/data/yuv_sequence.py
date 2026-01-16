import os
import torch
import numpy as np

class YUVSequence:
    """
    Simple YUV420 sequence reader (8-bit).
    Provides frame-by-frame access as torch tensors.
    """

    def __init__(
        self,
        path,
        width,
        height,
        num_frames=None,
        device="cuda",
    ):
        self.path = path
        self.width = width
        self.height = height
        self.device = device
        self.frame_size = width * height
        self.uv_size = (width // 2) * (height // 2)
        self.bytes_per_frame = self.frame_size + 2 * self.uv_size
        self.file = open(path, "rb")
        self.file_size = os.path.getsize(path)
        max_frames = self.file_size // self.bytes_per_frame
        self.num_frames = num_frames or max_frames
        assert self.num_frames <= max_frames, "Not enough data in YUV file"
        self.index = 0

    def __len__(self):
        return self.num_frames

    def reset(self):
        self.file.seek(0)
        self.index = 0

    def close(self):
        self.file.close()

    def read_frame(self):
        """
        Returns:
            frame: torch.Tensor [1, 3, H, W] on device
        """

        if self.index >= self.num_frames:
            raise StopIteration

        y = np.frombuffer(
            self.file.read(self.frame_size),
            dtype=np.uint8,
        ).reshape(self.height, self.width)

        u = np.frombuffer(
            self.file.read(self.uv_size),
            dtype=np.uint8,
        ).reshape(self.height // 2, self.width // 2)

        v = np.frombuffer(
            self.file.read(self.uv_size),
            dtype=np.uint8,
        ).reshape(self.height // 2, self.width // 2)

        self.index += 1

        u = u.repeat(2, axis=0).repeat(2, axis=1)
        v = v.repeat(2, axis =0).repeat(2, axis=1)

        y = y.astype(np.float32)
        u = u.astype(np.float32) - 128.0
        v = v.astype(np.float32) - 128.0

        r = y + 1.402 * v
        g = y - 0.344136 * u - 0.714136 * v
        b = y + 1.772 * u

        rgb = np.stack([r, g, b], axis=0)
        rgb = np.clip(rgb / 255.0, 0.0, 1.0)

        frame = torch.from_numpy(rgb).unsqueeze(0)
        return frame.to(self.device, non_blocking=True)