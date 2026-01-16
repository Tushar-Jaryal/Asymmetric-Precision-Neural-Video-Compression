import torch
from src.utils.common import get_state_dict
from src.models.image_model import DMCI
from src.models.video_model import DMC

def load_models(
    model_path_i=None,
    model_path_p=None,
    device="cuda",
    force_zero_thres=None,
):
    """
    Load DCVC-RT I-frame and P-frame models.
    """
    assert model_path_p is not None, "DCVC-RT requires video model checkpoint"
    print("[DCVC] Loading unified DCVC-RT model (DMC)")
    state = get_state_dict(model_path_p)
    model = DMC()
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    return model