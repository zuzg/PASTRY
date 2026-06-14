import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional import structural_similarity_index_measure as ssim


class HSIMetrics:
    @staticmethod
    def calculate_psnr(sr: Tensor, hr: Tensor, max_val: float = 1.0) -> float:
        mse_per_band = torch.mean((sr - hr) ** 2, dim=(-2, -1))

        mse_per_band = torch.clamp(mse_per_band, min=1e-12)

        # Calculate PSNR per band
        psnr_per_band = 20 * torch.log10(max_val / torch.sqrt(mse_per_band))

        return psnr_per_band.mean().item()

    @staticmethod
    def calculate_rmse(sr: Tensor, hr: Tensor) -> float:
        mse = F.mse_loss(sr, hr)
        return torch.sqrt(mse).item()

    @staticmethod
    def calculate_sam(sr: Tensor, hr: Tensor) -> float:
        sr = sr.permute(0, 2, 3, 1)
        hr = hr.permute(0, 2, 3, 1)

        dot_product = (sr * hr).sum(dim=-1)
        sr_norm = sr.norm(dim=-1)
        hr_norm = hr.norm(dim=-1)

        cos_sim = dot_product / (sr_norm * hr_norm + 1e-8)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        sam_map = torch.acos(cos_sim) * (180.0 / np.pi)

        return sam_map.mean().item()

    @staticmethod
    def calculate_ssim_bandwise(sr: Tensor, hr: Tensor) -> float:
        return ssim(sr, hr, data_range=1.0).item()
