import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

import wandb
from src.analysis.ablations import (
    run_constraint_ablation,
    run_k_sensitivity_analysis,
    run_kmeans_analysis,
    run_stability_analysis,
)
from src.architectures.unmixing import SpatialSymbolLearner
from src.config import ExperimentConfig
from src.consts import MODELS_DICT
from src.data.dataset import HSIDataset
from src.data.vca import extract_endmembers_vca
from src.training.library import check_library_quality, train_stage1_spatial
from src.training.sr import train_hsi, validate_hsi, visualize_hsi_samples
from src.validation.endmembers import validate_endmembers
from src.viz.visualize import visualize_endmember_dashboard, visualize_symbols


class Experiment:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self._initialize_wandb()
        self._init_dirs()

    def _initialize_wandb(self) -> None:
        wandb.init(
            project="hyperspectral-superresolution",
            name=f"{self.cfg.net.name}",
            config=vars(self.cfg),
            tags=self.cfg.tags,
            mode=self.cfg.wandb_mode,
        )

    def prepare_data(self) -> tuple[Dataset, Dataset]:
        d_cfg = self.cfg.data
        scale_factor = self.cfg.net.params["scale_factor"]

        train_ds = HSIDataset(d_cfg.path, d_cfg.name, "train", scale_factor)
        test_ds = HSIDataset(d_cfg.path, d_cfg.name, "test", scale_factor)
        return train_ds, test_ds

    def get_wavelengths(self, dataset_name: str, num_bands: int) -> np.ndarray:
        name = dataset_name.lower()
        if "cuprite" in name:
            # Original 224 bands from 370 nm to 2480 nm
            full_wv = np.linspace(370, 2480, 224)
            # Apply the same band removal mask used in the dataset
            remove_indices = list(range(0, 2)) + list(range(103, 113)) + list(range(147, 167)) + list(range(220, 224))
            keep_indices = [i for i in range(224) if i not in remove_indices]

            return full_wv[keep_indices]
        elif "houston" in name:
            return np.linspace(380, 1050, num_bands)
        elif "chikusei" in name:
            return np.linspace(363, 1018, num_bands)
        elif "pavia" in name:
            return np.linspace(430, 860, num_bands)
        else:
            return np.linspace(500, 800, num_bands)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _init_dirs(self) -> None:
        output_path = Path(self.cfg.net.ckpt_path).parents[1]
        # Path(self.cfg.net.ckpt_path).parents[0].mkdir(parents=True, exist_ok=True)
        (output_path / "models").mkdir(parents=True, exist_ok=True)
        (output_path / "viz").mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        train_ds, test_ds = self.prepare_data()
        train_dl = DataLoader(train_ds, batch_size=self.cfg.data.batch_size, shuffle=True)
        test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
        num_bands = train_ds.full_image.shape[2]
        init_image = train_ds.image
        num_symbols = self.cfg.net.params["num_symbols"]
        wv = self.get_wavelengths(self.cfg.data.name, num_bands)

        if self.cfg.net.name == "Nesy":
            symbol_learner = SpatialSymbolLearner(num_bands, num_symbols)
            if Path(self.cfg.net.ckpt_path).exists():
                symbol_learner.load_state_dict(torch.load(self.cfg.net.ckpt_path, weights_only=True))
                symbol_learner = symbol_learner.to(self.cfg.trainer.device)
                library = symbol_learner.decoder.weight.data.squeeze().T.detach()
                logger.info(f"Loaded library from {self.cfg.net.ckpt_path}")
            else:
                library = train_stage1_spatial(symbol_learner, train_dl, num_symbols, init_image, self.cfg.trainer)
                torch.save(symbol_learner.state_dict(), self.cfg.net.ckpt_path)
                logger.info(f"Saved library to {self.cfg.net.ckpt_path}")

            symbols = visualize_symbols(symbol_learner, library, init_image, wv, 2, device=self.cfg.trainer.device)
            wandb.log({"symbols_train": wandb.Image(symbols)})
            self.cfg.net.params["pretrained_library"] = library
            check_library_quality(symbol_learner, test_dl, device=self.cfg.trainer.device)

        elif self.cfg.net.name == "VCA":
            pixels = init_image.reshape(-1, init_image.shape[-1])
            valid_mask = np.all((pixels > 0.0) & (pixels < 1.0), axis=1)
            clean_pixels = pixels[valid_mask]
            vca_endmembers = extract_endmembers_vca(clean_pixels.T, num_endmembers=num_symbols)
            library = torch.from_numpy(vca_endmembers).float().to(self.cfg.trainer.device)
            self.cfg.net.params["pretrained_library"] = library
            self.cfg.net.name = "Nesy"

        self.cfg.net.params["num_bands"] = num_bands

        model_class = MODELS_DICT[self.cfg.net.name]
        model = model_class(**self.cfg.net.params)

        if self.cfg.net.name == "Bicubic":
            avg_metrics, vis_samples = validate_hsi(model, test_dl, self.cfg.trainer.device)
            log_dict = {
                "epoch": self.cfg.trainer.epochs,
                "val_psnr": avg_metrics["psnr"],
                "val_sam": avg_metrics["sam"],
                "val_rmse": avg_metrics["rmse"],
            }
            visualize_hsi_samples(vis_samples, self.cfg.trainer.epochs - 1, log_dict)
            wandb.log(log_dict)
        else:
            sr_path = "output/models/sr_TOP_x.pth"
            if Path(sr_path).exists():
                model.load_state_dict(torch.load(sr_path))
                model = model.to(self.cfg.trainer.device)
            else:
                train_hsi(model, train_dl, test_dl, self.cfg.trainer)
                # torch.save(model.state_dict(), sr_path)
            symbols = visualize_symbols(model, library, test_ds.image, wv, 2, mode="test", device=self.cfg.trainer.device)
            if self.cfg.data.name == "paviac":
                img = test_ds.image[:220, ...]  # paviac
                validate_endmembers(model, train_ds, "data/benchmark_full/Pavia_gt.mat")
            else:
                img = test_ds.image[:, :128, ...]  # chikusei
            visualize_endmember_dashboard(model, library, img, wv, device=self.cfg.trainer.device)
            wandb.log({"symbols_test": wandb.Image(symbols)})

        wandb.finish()

    def stability_analysis(self, num_runs: int = 5) -> None:
        run_stability_analysis(self, num_runs)

    def k_sensitivity_analysis(self, k_values: list[int]) -> None:
        run_k_sensitivity_analysis(self, k_values)

    def constraint_ablation(self) -> None:
        run_constraint_ablation(self)

    def elbow_analysis(self, max_k: int = 30) -> None:
        run_kmeans_analysis(self, k_range=range(2, max_k + 1, 2))
