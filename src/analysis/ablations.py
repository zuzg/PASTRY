from collections import defaultdict

import numpy as np
import pysptools.material_count as ns
import torch
from loguru import logger
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from src.architectures.unmixing.spatial import SpatialSymbolLearner
from src.consts import MODELS_DICT
from src.training.library import train_stage1_spatial
from src.training.sr import train_hsi, validate_hsi
from src.viz.visualize import (
    analyze_endmember_utilization,
    plot_constraint_comparison,
    plot_k_sensitivity,
    plot_kmeans_metrics,
)


def run_stability_analysis(exp, num_runs: int = 5) -> None:
    train_ds, test_ds = exp.prepare_data()
    train_dl = DataLoader(train_ds, batch_size=exp.cfg.data.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    num_bands = train_ds.full_image.shape[2]
    init_image = train_ds.image
    num_symbols = exp.cfg.net.params["num_symbols"]

    all_libraries = []
    all_metrics = defaultdict(list)

    for run_idx in range(num_runs):
        seed = 42 + run_idx
        exp._set_seed(seed)
        logger.info(f"=== Starting Stability Run {run_idx + 1}/{num_runs} (Seed: {seed}) ===")

        # Stage 1
        symbol_learner = SpatialSymbolLearner(num_bands, num_symbols)
        library = train_stage1_spatial(symbol_learner, train_dl, num_symbols, init_image, exp.cfg.trainer)
        all_libraries.append(library.detach().cpu().numpy())

        exp.cfg.net.params["pretrained_library"] = library
        exp.cfg.net.params["num_bands"] = num_bands

        # Stage 2
        model_class = MODELS_DICT[exp.cfg.net.name]
        model = model_class(**exp.cfg.net.params)

        logger.info("Training Stage 2: Super-Resolution...")
        train_hsi(model, train_dl, test_dl, exp.cfg.trainer)

        # Evaluate
        avg_metrics, _ = validate_hsi(model, test_dl, exp.cfg.trainer.device)
        for k, v in avg_metrics.items():
            all_metrics[k].append(v)

        logger.info(f"Run {run_idx + 1} | PSNR: {avg_metrics['psnr']:.2f} | SAM: {avg_metrics['sam']:.4f}")

    # Hungarian Alignment
    base_lib = all_libraries[0]
    aligned_libraries = [base_lib]

    for i in range(1, num_runs):
        curr_lib = all_libraries[i]
        cost_matrix = np.zeros((num_symbols, num_symbols))
        for r in range(num_symbols):
            for c in range(num_symbols):
                dot = np.dot(base_lib[r], curr_lib[c])
                norm = np.linalg.norm(base_lib[r]) * np.linalg.norm(curr_lib[c])
                cos_sim = np.clip(dot / (norm + 1e-8), -1.0, 1.0)
                cost_matrix[r, c] = np.arccos(cos_sim)
        _, col_ind = linear_sum_assignment(cost_matrix)
        aligned_libraries.append(curr_lib[col_ind])

    aligned_libraries = np.array(aligned_libraries)
    mean_lib = np.mean(aligned_libraries, axis=0)

    endmember_variances = []
    for s in range(num_symbols):
        sams = []
        for r in range(num_runs):
            dot = np.dot(mean_lib[s], aligned_libraries[r, s])
            norm = np.linalg.norm(mean_lib[s]) * np.linalg.norm(aligned_libraries[r, s])
            cos_sim = np.clip(dot / (norm + 1e-8), -1.0, 1.0)
            sams.append(np.arccos(cos_sim) * (180.0 / np.pi))
        endmember_variances.append(np.std(sams))

    avg_lib_std = np.mean(endmember_variances)

    logger.info(f"\nLibrary Stability (Mean SAM Std Dev): ± {avg_lib_std:.2f}°")
    for k, v in all_metrics.items():
        logger.info(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")


def run_k_sensitivity_analysis(exp, k_values: list[int]) -> None:
    train_ds, test_ds = exp.prepare_data()
    train_dl = DataLoader(train_ds, batch_size=exp.cfg.data.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    num_bands = train_ds.full_image.shape[2]
    init_image = train_ds.image

    sensitivity_results = defaultdict(list)

    for k in k_values:
        logger.info(f"\n{'='*40}\nTesting K = {k}\n{'='*40}")
        exp._set_seed(42)
        exp.cfg.net.params["num_symbols"] = k

        symbol_learner = SpatialSymbolLearner(num_bands, k)
        library = train_stage1_spatial(symbol_learner, train_dl, k, init_image, exp.cfg.trainer)

        exp.cfg.net.params["pretrained_library"] = library
        exp.cfg.net.params["num_bands"] = num_bands

        model_class = MODELS_DICT[exp.cfg.net.name]
        model = model_class(**exp.cfg.net.params)

        train_hsi(model, train_dl, test_dl, exp.cfg.trainer)

        avg_metrics, _ = validate_hsi(model, test_dl, exp.cfg.trainer.device)

        sensitivity_results["k"].append(k)
        for metric, value in avg_metrics.items():
            sensitivity_results[metric].append(value)

        if k == max(k_values):
            analyze_endmember_utilization(
                model=model,
                hsi_image=test_ds.image,
                k=k,
                save_path=f"output/viz/utilization_K{k}.pdf",
                device=exp.cfg.trainer.device,
            )

    plot_k_sensitivity(sensitivity_results)


def run_constraint_ablation(exp) -> None:
    train_ds, test_ds = exp.prepare_data()
    train_dl = DataLoader(train_ds, batch_size=exp.cfg.data.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)
    num_bands = train_ds.full_image.shape[2]
    init_image = train_ds.image
    num_symbols = exp.cfg.net.params["num_symbols"]
    
    modes = ["both", "anc_only", "asc_only", "none"]
    results = {}
    sample_abundance_maps = {} 
    test_img_tensor = test_ds[0]["lr"].unsqueeze(0).to(exp.cfg.trainer.device)

    for mode in modes:
        logger.info(f"\n{'='*40}\nTesting Constraint Mode: {mode.upper()}\n{'='*40}")
        exp._set_seed(42) # fixed so ONLY the constraint changes
        
        exp.cfg.net.params["constraint_mode"] = mode
        
        symbol_learner = SpatialSymbolLearner(num_bands, num_symbols, constraint_mode=mode)
        library = train_stage1_spatial(symbol_learner, train_dl, num_symbols, init_image, exp.cfg.trainer)
        
        exp.cfg.net.params["pretrained_library"] = library
        exp.cfg.net.params["num_bands"] = num_bands

        model_class = MODELS_DICT[exp.cfg.net.name]
        model = model_class(**exp.cfg.net.params)
        
        train_hsi(model, train_dl, test_dl, exp.cfg.trainer)

        avg_metrics, _ = validate_hsi(model, test_dl, exp.cfg.trainer.device)
        results[mode] = avg_metrics
        logger.info(f"Mode {mode.upper()} | PSNR: {avg_metrics['psnr']:.2f} | SAM: {avg_metrics['sam']:.4f}")

        model.eval()
        with torch.no_grad():
            _, abundances = model(test_img_tensor)
            # Store as numpy array: Shape [K, H, W]
            sample_abundance_maps[mode] = abundances.squeeze(0).cpu().numpy()

    logger.info("\n" + "="*50)
    logger.info("PHYSICS CONSTRAINT ABLATION REPORT")
    logger.info("="*50)
    logger.info(f"{'Constraint':>12} | {'PSNR (dB)':>10} | {'SAM (deg)':>10} | {'RMSE':>8}")
    logger.info("-" * 50)
    
    for mode in modes:
        psnr = results[mode]["psnr"]
        sam = results[mode]["sam"]
        rmse = results[mode]["rmse"]
        logger.info(f"{mode.upper():>12} | {psnr:10.3f} | {sam:10.3f} | {rmse:8.4f}")

    logger.info("Generating visual constraint comparison...")
    plot_constraint_comparison(sample_abundance_maps, endmember_idx=2)


def run_kmeans_analysis(exp, k_range: range = range(2, 21)) -> None:
    train_ds, _ = exp.prepare_data()
    init_image = train_ds.image
    b = init_image.shape[2]
    
    logger.info("Running HySime to estimate intrinsic Virtual Dimensionality (this may take a minute)...")
    hysime = ns.HySime()
    kf, Rn = hysime.count(init_image)
    hysime_k = int(kf)
    logger.info(f">>> HySime estimated Virtual Dimensionality (K) = {hysime_k} <<<")
    
    logger.info("Extracting valid pixels for K-Means analysis...")
    flat_data = init_image.reshape(-1, b)
    valid_mask = np.all((flat_data > 0.0) & (flat_data <= 1.0), axis=1)
    clean_pixels = flat_data[valid_mask]
    
    k_values = list(k_range)
    inertias = []
    silhouette_scores = []
    
    for k in k_values:
        logger.info(f"Fitting K-Means for K={k}...")
        exp._set_seed(42) 
        
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(clean_pixels)
        inertias.append(kmeans.inertia_)
        
        score = silhouette_score(
            clean_pixels, 
            kmeans.labels_, 
            sample_size=20000, 
            random_state=42
        )
        silhouette_scores.append(score)
        
        logger.info(f"K={k} | Inertia: {kmeans.inertia_:.2f} | Silhouette: {score:.4f}")

    plot_kmeans_metrics(k_values, inertias, silhouette_scores, hysime_k=hysime_k)
