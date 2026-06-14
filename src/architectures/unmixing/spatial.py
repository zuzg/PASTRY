from torch import Tensor, nn

from src.architectures.constraint import AbundanceConstraint


class SpatialSymbolLearner(nn.Module):
    def __init__(self, num_bands: int, num_symbols: int, constraint_mode: str = "both"):
        super(SpatialSymbolLearner, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_bands, 128, 3, padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, 3, padding=1, padding_mode="zeros"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, num_symbols, 3, padding=1, padding_mode="zeros"),
            # nn.BatchNorm1d(num_symbols),
            AbundanceConstraint(mode=constraint_mode),
        )
        self.decoder = nn.Conv2d(num_symbols, num_bands, 1, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        abundances = self.encoder(x)
        reconstruction = self.decoder(abundances)
        return reconstruction, abundances
