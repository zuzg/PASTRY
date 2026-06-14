import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvSR(nn.Module):
    def __init__(self, scale_factor: int, num_bands: int):
        super(ConvSR, self).__init__()
        self.conv1 = nn.Conv2d(num_bands, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv_up = nn.Conv2d(64, num_bands * (scale_factor**2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.conv_out = nn.Conv2d(num_bands, num_bands, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv_up(x))
        x = self.conv_out(x)
        return torch.sigmoid(x)
