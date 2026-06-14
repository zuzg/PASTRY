import torch.nn.functional as F
from torch import Tensor, nn


class BicubicSR(nn.Module):
    def __init__(self, scale_factor: int):
        super(BicubicSR, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        sr = F.interpolate(
            x, scale_factor=self.scale_factor, mode="bicubic", align_corners=False
        )
        return sr
