import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch.utils.data import DataLoader

import wandb
from src.config import TrainerConfig
from src.data.vca import extract_endmembers_vca
from src.metrics import HSIMetrics


def train_stage1_spatial(
    model: nn.Module,
    train_loader: DataLoader,
    num_symbols: int,
    init_image: np.ndarray,
    cfg: TrainerConfig,
) -> Tensor:
    model = model.to(cfg.device)
    logger.info(f"Running {cfg.init_method} for symbol initialization...")

    if cfg.init_method == "kmeans":
        b = init_image.shape[2]
        flat_data = init_image.reshape(-1, b)
        flat_data = np.delete(flat_data, np.where(flat_data == -1), axis=0)
        kmeans = KMeans(n_clusters=num_symbols, n_init=10).fit(flat_data)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    elif cfg.init_method == "vca":
        pixels = init_image.reshape(-1, init_image.shape[-1])
        valid_mask = np.all((pixels > 0.0) & (pixels < 1.0), axis=1)
        clean_pixels = pixels[valid_mask]
        centers = extract_endmembers_vca(clean_pixels.T, num_symbols)
        centers = torch.tensor(centers, dtype=torch.float32)

    with torch.no_grad():
        model.decoder.weight.data = centers.T.to(cfg.device).unsqueeze(2).unsqueeze(2)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.L1Loss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    logger.info("Stage-1: Training Spatial-Symbol Learner...")

    model.train()

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0

        for batch in train_loader:
            patches = batch["hr"].to(cfg.device)

            optimizer.zero_grad()
            recon, ab = model(patches)
            loss = criterion(recon, patches)

            loss.backward()
            optimizer.step()

            model.decoder.weight.data.clamp_(min=0.0)

            epoch_loss += loss.item()

        scheduler.step()

        logger.info(f"[Stage-1] Epoch {epoch+1}/{cfg.epochs} | " f"Loss: {epoch_loss / len(train_loader):.5f}")

    return model.decoder.weight.data.squeeze().T.detach()


def check_library_quality(learner: nn.Module, valid_loader: DataLoader, device="cuda"):
    learner.eval()
    total_psnr = 0
    with torch.no_grad():
        for batch in valid_loader:
            img = batch["hr"].to(device)
            recon, ab = learner(img)
            total_psnr += HSIMetrics.calculate_psnr(recon, img)

    avg_psnr = total_psnr / len(valid_loader)
    logger.info(f"Stage 1 Library Quality: {avg_psnr:.2f} dB")
    wandb.log({"lib_psnr": avg_psnr})
