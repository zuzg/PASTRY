import numpy as np
import torch
from loguru import logger
from torch import Tensor, nn

from src.architectures.constraint import AbundanceConstraint

def get_bilinear_kernel(kernel_size: int, channels: int) -> Tensor:
    """
    Creates a bilinear upsampling kernel.
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((channels, channels, kernel_size, kernel_size), dtype=np.float32)

    for i in range(channels):
        weight[i, i, :, :] = filt

    return torch.from_numpy(weight)


class ResidualBlock(nn.Module):
    """Standard ResBlock"""

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)  # TODO leaky?
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class NeuroSymbolicSR(nn.Module):
    def __init__(
        self,
        scale_factor: int,
        num_bands: int,
        num_symbols: int,
        pretrained_library: Tensor | None = None,
        constraint_mode: str = "both",
    ):
        super(NeuroSymbolicSR, self).__init__()
        self.scale_factor = scale_factor

        self.head = nn.Conv2d(num_bands, 64, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])
        self.abundance_conv = nn.Conv2d(64, num_symbols, 3, padding=1)

        kernel_size = 2 * scale_factor
        padding = scale_factor // 2
        self.up_conv = nn.ConvTranspose2d(
            in_channels=num_symbols,
            out_channels=num_symbols,
            kernel_size=kernel_size,
            stride=scale_factor,
            padding=padding,
            bias=False,
        )
        self._init_upsample_weights(kernel_size, num_symbols)

        self.abundance_act = AbundanceConstraint(mode=constraint_mode)

        self.renderer = nn.Conv2d(num_symbols, num_bands, 1, bias=False)

        if pretrained_library is not None:
            self.load_library(pretrained_library)

    def _init_upsample_weights(self, kernel_size: int, channels: int):
        """Helper to apply the bilinear weights to the layer."""
        initial_weight = get_bilinear_kernel(kernel_size, channels)
        with torch.no_grad():
            self.up_conv.weight.copy_(initial_weight)
        logger.info(f"Upsample layer initialized with Bilinear weights (k={kernel_size}).")

    def load_library(self, library_tensor: Tensor):
        with torch.no_grad():
            self.renderer.weight.data = library_tensor.T.unsqueeze(2).unsqueeze(2)
            self.renderer.weight.requires_grad = False
            logger.info("Spectral Library Loaded & Frozen.")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.head(x)
        features = self.body(features) + features
        logits_lr = self.abundance_conv(features)
        logits_hr = self.up_conv(logits_lr)
        hr_abundances = self.abundance_act(logits_hr)
        base_image = self.renderer(hr_abundances)
        return base_image, hr_abundances
