import torch
import numpy as np

def load_yuv420_as_tensor(
    path,
    width,
    height,
    frames,
    device="cuda"
):
    """
    Returns:
        List[Tensor] of shape [1,3,H,W] in [0,1]
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size

    tensors = []

    with open(path, "rb") as f:
        for _ in range(frames):
            y = np.frombuffer(f.read(y_size), dtype=np.uint8).reshape(height, width)
            u = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape(height // 2, width // 2)
            v = np.frombuffer(f.read(uv_size), dtype=np.uint8).reshape(height // 2, width // 2)

            # Upsample UV → Y size
            u = u.repeat(2, axis=0).repeat(2, axis=1)
            v = v.repeat(2, axis=0).repeat(2, axis=1)

            rgb = np.stack([y, u, v], axis=0) / 255.0
            tensor = torch.from_numpy(rgb).float().unsqueeze(0).to(device)

            tensors.append(tensor)

    return tensors