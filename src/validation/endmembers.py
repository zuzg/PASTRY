from typing import Any

import mat73
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import spectral.io.envi as envi
import torch.nn as nn
from loguru import logger
from scipy.optimize import linear_sum_assignment

from src.consts import HOUSTON_CLASSES, PAVIA_CLASSES


def calculate_sam_numpy(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.dot(v1_norm, v2_norm)
    dot = np.clip(dot, -1, 1)
    return float(np.degrees(np.arccos(dot)))


def get_reference_endmembers(image_np: np.ndarray, gt_path: str) -> tuple[np.ndarray | None, list[str] | None]:
    if ".mat" in gt_path:
        try:
            gt_data = sio.loadmat(gt_path)
        except Exception:
            gt_data = mat73.loadmat(gt_path)
        try:
            key = [k for k in gt_data.keys() if "gt" in k][0]
            gt_map: np.ndarray = gt_data[key]
        except Exception as e:
            logger.error(f"Could not load GT file at {gt_path}: {e}")
            return None, None
    else:
        gt_map = envi.open(gt_path).asarray()[..., 0]

    if image_np.shape[:2] != gt_map.shape:
        logger.warning(f"Shape mismatch: Img {image_np.shape} vs GT {gt_map.shape}. Attempting crop...")
        h, w = image_np.shape[:2]
        gh, gw = gt_map.shape
        sh = (gh - h) // 2
        sw = (gw - w) // 2
        gt_map = gt_map[sh : sh + h, sw : sw + w]

    unique_classes = np.unique(gt_map)
    unique_classes = unique_classes[unique_classes != 0]

    ref_endmembers: list[np.ndarray] = []
    class_names: list[str] = []
    if "Pavia" in gt_path:
        classes = PAVIA_CLASSES
    else:
        classes = HOUSTON_CLASSES

    for c in unique_classes:
        mask = gt_map == c
        mean_spectrum = image_np[mask].mean(axis=0)
        ref_endmembers.append(mean_spectrum)
        class_names.append(classes[c])

    return np.array(ref_endmembers), class_names


def compute_matches(model: nn.Module, dataset: Any, gt_path: str) -> dict[str, Any] | None:
    model.eval()
    if hasattr(model, "renderer") and hasattr(model.renderer, "weight"):
        learned_weights: np.ndarray = model.renderer.weight.data.squeeze().cpu().numpy()
    else:
        logger.error("Model structure unknown: could not find renderer.weight")
        return None

    ref_endmembers, ref_names = get_reference_endmembers(dataset.full_image, gt_path)
    if ref_endmembers is None or ref_names is None:
        return None

    num_learned = learned_weights.shape[1]
    num_ref = len(ref_endmembers)
    logger.info(f"Matching {num_learned} learned symbols against {num_ref} GT classes...")

    cost_matrix = np.zeros((num_ref, num_learned))
    for r in range(num_ref):
        for l in range(num_learned):
            cost_matrix[r, l] = calculate_sam_numpy(ref_endmembers[r], learned_weights[:, l])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    wv = np.linspace(500, 800, dataset.full_image.shape[-1])

    return {
        "row_ind": row_ind,
        "col_ind": col_ind,
        "ref_endmembers": ref_endmembers,
        "learned_weights": learned_weights,
        "ref_names": ref_names,
        "cost_matrix": cost_matrix,
        "wv": wv,
    }


def normalize_spectrum(spec: np.ndarray) -> np.ndarray:
    return (spec - spec.min()) / (spec.max() - spec.min())


def plot_full_grid(data: dict[str, Any], save_path: str = "output/viz/validation.pdf") -> None:
    row_ind: np.ndarray = data["row_ind"]
    col_ind: np.ndarray = data["col_ind"]
    ref_endmembers: np.ndarray = data["ref_endmembers"]
    learned_weights: np.ndarray = data["learned_weights"]
    ref_names: list[str] = data["ref_names"]
    cost_matrix: np.ndarray = data["cost_matrix"]
    wv: np.ndarray = data["wv"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 15,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
        }
    )

    fig, axes_array = plt.subplots(3, 3, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes_array.flatten()

    total_sam: float = 0.0

    for i, (r, c) in enumerate(zip(row_ind, col_ind)):
        if i >= len(axes):
            break

        ref_spec = normalize_spectrum(ref_endmembers[r])
        learned_spec = normalize_spectrum(learned_weights[:, c])
        sam_score = cost_matrix[r, c]
        total_sam += sam_score

        ax = axes[i]
        ax.plot(wv, ref_spec, label="GT", color="black", linestyle="--")
        ax.plot(wv, learned_spec, label=f"E. {c}", color="red")
        ax.set_title(f"{ref_names[r]} (SAM: {sam_score:.2f}°)")
        ax.legend(fontsize="small", loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.margins(x=0)

    plt.subplots_adjust(wspace=0.1, hspace=0.25)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    logger.info(f"Saved full grid plot to {save_path}")


def plot_chosen_subset(data: dict[str, Any], save_path: str = "output/viz/endm-match-chosen.pdf") -> None:
    row_ind: np.ndarray = data["row_ind"]
    col_ind: np.ndarray = data["col_ind"]
    ref_endmembers: np.ndarray = data["ref_endmembers"]
    learned_weights: np.ndarray = data["learned_weights"]
    ref_names: list[str] = data["ref_names"]

    target_classes = ["Water", "Trees", "S-B Bricks", "Shadows"]

    fig, axes_array = plt.subplots(1, len(target_classes), figsize=(16, 2), sharey=True)
    axes = axes_array.flatten()

    plot_idx = 0

    for r, c in zip(row_ind, col_ind):
        if ref_names[r] in target_classes:
            if plot_idx >= len(axes):
                break

            ref_spec = normalize_spectrum(ref_endmembers[r])
            learned_spec = normalize_spectrum(learned_weights[:, c])

            ax = axes[plot_idx]
            ax.plot(ref_spec, label="GT", color="black", linestyle="--")
            ax.plot(learned_spec, label=f"E.{c}", color="red")
            ax.set_title(f"{ref_names[r]}")
            ax.legend(fontsize="small", loc="upper right")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis="both", which="both", length=0)
            ax.margins(x=0)

            plot_idx += 1

    plt.subplots_adjust(wspace=0.05, hspace=0, left=0.01, right=0.99, bottom=0.01, top=0.90)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()
    logger.info(f"Saved chosen subset plot to {save_path}")


def validate_endmembers(
    model: nn.Module,
    dataset: Any,
    gt_path: str,
    mode: str = "all",
) -> None:
    data = compute_matches(model, dataset, gt_path)
    if data is None:
        return

    if mode in ["all", "both"]:
        plot_full_grid(data)

    if mode in ["chosen", "both"]:
        plot_chosen_subset(data)
