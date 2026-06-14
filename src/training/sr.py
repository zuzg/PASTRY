from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from loguru import logger
from torch import Tensor, nn
from torch.utils.data import DataLoader

import wandb
from src.config import TrainerConfig
from src.metrics import HSIMetrics
from src.viz.visualize import get_hsi_plot


class HybridLoss(nn.Module):
    def __init__(self, sam_weight: float = 0.1, tv_weight: float = 0.0):
        super(HybridLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.sam_weight = sam_weight
        self.tv_weight = tv_weight

    def forward(self, sr: Tensor, hr: Tensor) -> float:
        loss_l1 = self.l1(sr, hr)

        dot = torch.sum(sr * hr, dim=1)
        norm_sr = torch.norm(sr, dim=1)
        norm_hr = torch.norm(hr, dim=1)
        cos_sim = dot / (norm_sr * norm_hr + 1e-8)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        loss_sam = torch.acos(cos_sim).mean()

        h_tv = torch.abs(sr[:, :, 1:, :] - sr[:, :, :-1, :]).mean()
        w_tv = torch.abs(sr[:, :, :, 1:] - sr[:, :, :, :-1]).mean()
        loss_tv = h_tv + w_tv

        return loss_l1 + (self.sam_weight * loss_sam) + (self.tv_weight * loss_tv)


class AbundanceLoss(nn.Module):
    """
    Encourages physical plausibility in abundance maps.

    Args:
        sparsity_weight (float): Penalty strength for non-sparse solutions (L1 norm).
        smoothness_weight (float): Penalty for noisy transitions between pixels (Total Variation).
    """

    def __init__(self, sparsity_weight: float = 1e-3, smoothness_weight: float = 1e-5):
        super(AbundanceLoss, self).__init__()
        self.sparsity_weight = sparsity_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, abundances: Tensor) -> Tensor:

        epsilon = 1e-8
        entropy = -torch.sum(abundances * torch.log(abundances + epsilon), dim=1)
        sparsity_loss = torch.mean(entropy)

        diff_h = torch.abs(abundances[..., :-1, :] - abundances[..., 1:, :])
        diff_w = torch.abs(abundances[..., :, :-1] - abundances[..., :, 1:])
        smoothness_loss = torch.mean(diff_h) + torch.mean(diff_w)
        total_loss = (self.sparsity_weight * sparsity_loss) + (self.smoothness_weight * smoothness_loss)
        return total_loss


def validate_hsi(model: nn.Module, dataloader: DataLoader, device: str) -> tuple[dict, list]:
    model.eval()
    metrics = {"psnr": 0.0, "sam": 0.0, "rmse": 0.0, "ssim": 0.0}

    vis_samples = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            lr_imgs = batch["lr"].to(device)
            hr_imgs = batch["hr"].to(device)

            sr_imgs = model(lr_imgs)
            if isinstance(sr_imgs, tuple):
                sr_imgs, abundances = sr_imgs
            else:
                abundances = None
            sr_imgs = torch.clamp(sr_imgs, 0.0, 1.0)

            metrics["psnr"] += HSIMetrics.calculate_psnr(sr_imgs, hr_imgs)
            metrics["sam"] += HSIMetrics.calculate_sam(sr_imgs, hr_imgs)
            metrics["rmse"] += HSIMetrics.calculate_rmse(sr_imgs, hr_imgs)
            metrics["ssim"] += HSIMetrics.calculate_ssim_bandwise(sr_imgs, hr_imgs)

            if i in (0, 3):
                vis_samples.append(
                    (
                        hr_imgs[0],
                        sr_imgs[0],
                        lr_imgs[0],
                        abundances[0] if abundances is not None else None,
                    )
                )

    avg_metrics = {k: v / len(dataloader) for k, v in metrics.items()}
    return avg_metrics, vis_samples


def visualize_hsi_samples(vis_samples: list[Tensor], epoch: int, log_dict: dict[str, Any]):
    if (epoch + 1) % 10 != 0:
        return

    for idx, (hr, sr, lr, ab) in enumerate(vis_samples):
        fig = get_hsi_plot(
            hr,
            sr,
            lr,
            idx,
            band_indices=(58, 38, 23),
        )
        log_dict[f"visual_results_{idx}"] = wandb.Image(fig)
        plt.close(fig)


def train_hsi(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainerConfig,
):
    model = model.to(cfg.device)
    criterion = HybridLoss().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # wandb.watch(model, criterion, log="all", log_freq=10)
    logger.info(f"🚀 Training on {cfg.device}...")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            lr_imgs = batch["lr"].to(cfg.device)
            hr_imgs = batch["hr"].to(cfg.device)

            optimizer.zero_grad()
            sr_imgs = model(lr_imgs)
            if isinstance(sr_imgs, tuple):
                sr_imgs, abundances = sr_imgs
            loss = criterion(sr_imgs, hr_imgs)  # + criterion_ab(abundances)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        avg_metrics, vis_samples = validate_hsi(model, test_loader, cfg.device)

        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_psnr": avg_metrics["psnr"],
            "val_sam": avg_metrics["sam"],
            "val_rmse": avg_metrics["rmse"],
            "val_ssim": avg_metrics["ssim"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        visualize_hsi_samples(vis_samples, epoch, log_dict)

        wandb.log(log_dict)
        logger.info(
            f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | "
            f"PSNR: {avg_metrics['psnr']:.2f} | SAM: {avg_metrics['sam']:.2f}"
        )

    logger.info("Run Complete.")
