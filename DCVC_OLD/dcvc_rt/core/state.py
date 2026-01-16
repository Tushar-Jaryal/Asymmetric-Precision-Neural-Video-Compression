class TemporalState:
    """
    Holds temporal information between frames.
    This is intentionally minimal and explicit.
    """

    def __init__(self):
        self.reset()
    def reset(self):
        self.prev_frame = None
        self.prev_rec = None
        self.is_intra = True