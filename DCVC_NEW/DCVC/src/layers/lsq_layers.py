import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSQPlusQuantizer(nn.Module):
    def __init__(self, bit_width=8, is_weight=False, out_channels=1):
        super().__init__()
        self.bit_width = bit_width
        self.is_weight = is_weight
        self.per_channel = per_channel and is_weight

        if self.per_channel:
            self.s = nn.Paramter(torch.ones(out_channels, 1, 1, 1))
        else:
            self.s = nn.Parameter(torch.ones(1))
        self.is_initialized = False
        self.Qn = -(2 ** (bit_width - 1))
        self.Qp = 2 ** (bit_width - 1) - 1

    def _init_step_size(self, x):
        detached_x = x.detach()
        if self.per_channel:
            flat_x = detached_x.view(x.shape[0], -1)
            mean_val = flat_x.abs().mean(dim=1).view(-1, 1, 1, 1)
            init_val = 2 * mean_val / math.sqrt(self.Qp)
            self.s.data.copy_(init_val)
        else:
            mean_val = detached_x.abs().mean()
            init_val = 2 * mean_val / math.sqrt(self.Qp)
            self.s.data.copy_(init_val)
        self.is_initialized = True

    def forward(self, x):
        if self.training and not self.is_initialized:
            self._init_step_size(x)
        if self.per_channel:
            numel = x.numel() // x.shape[0]
        else:
            numel = x.numel()
        g = 1.0 / math.sqrt(numel * self.Qp)
        s_scale = (self.s - self.s.detach()) * g + self.s.detach()
        x_div = x / s_scale
        x_int = torch.clamp(torch.round(x_div), self.Qn, self.Qp)
        x_dequant = x_int * s_scale
        if not self.is_weight:
            x_dequant = torch.clamp(x_dequant, -1000.0, 1000.0)
        return (x_dequant - x).detach() + x

class LSQConv2dPlus(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bit_width=8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.act_quantizer = LSQPlusQuantizer(bit_width=bit_width, is_weight=False)
        self.weight_quantizer = LSQPlusQuantizer(
            bit_width=bit_width, is_weight=True,
            per_channel=True, out_channels=out_channels
        )
    def forward(self, x):
        x_q = self.act_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.conv2d(x_q, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)