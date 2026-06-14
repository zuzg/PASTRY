import cmcrameri.cm as cmc
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from torch import Tensor, nn


def create_rgb_image(hsi_tensor: np.ndarray, bands_rgb: list[int] = [50, 27, 10]) -> np.ndarray:
    rgb_image = hsi_tensor[:, :, bands_rgb]
    p2, p98 = np.percentile(rgb_image, (2, 98))
    rgb_image_stretched = np.clip((rgb_image - p2) / (p98 - p2), 0, 1)
    return rgb_image_stretched


def visualize_symbols(
    model: nn.Module,
    library_weights: Tensor,
    hsi_image: np.ndarray | Tensor,
    wavelengths: np.ndarray | None = None,
    num_cols: int = 3,
    mode: str = "train",
    device: str = "cuda",
) -> plt.Figure:
    model.eval()
    h, w, b = hsi_image.shape

    if isinstance(hsi_image, np.ndarray):
        hsi_image = torch.from_numpy(hsi_image).float().permute(2, 0, 1).unsqueeze(0)
    else:
        hsi_image = hsi_image.unsqueeze(0)

    hsi_image = hsi_image.to(device)

    with torch.no_grad():
        if mode == "train":
            _, abundances = model(hsi_image)
        else:
            _, abundances = model(hsi_image)

    num_symbols = library_weights.shape[0]
    if wavelengths is None:
        wavelengths = np.arange(b)

    num_rows = (num_symbols + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(6 * num_cols, 3 * num_rows))
    axes = axes.flatten()

    cmap = plt.get_cmap("viridis")
    cmap.set_bad(color="white")

    for i in range(num_symbols):
        ax_spec = axes[2 * i]  # Even index: Spectrum
        ax_map = axes[2 * i + 1]  # Odd index: Map

        lw = library_weights[i].cpu().numpy()

        ax_spec.plot(wavelengths, lw, color="tab:blue", linewidth=2)
        ax_spec.set_title(f"Symbol {i} Signature", fontsize=10, fontweight="bold")
        ax_spec.set_xlabel("Wavelength")
        ax_spec.set_ylabel("Reflectance")
        ax_spec.grid(True, alpha=0.3)

        ax_spec.fill_between(wavelengths, lw, alpha=0.1, color="tab:blue")

        im = ax_map.imshow(abundances[0, i].cpu().numpy(), cmap=cmap, vmin=0, vmax=1)
        ax_map.set_title(f"Symbol {i} Location", fontsize=10, fontweight="bold")
        ax_map.axis("off")
        plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)

    total_plots = num_cols * 2 * num_rows
    for j in range(num_symbols * 2, total_plots):
        axes[j].axis("off")

    plt.tight_layout()
    return fig


def visualize_endmember_dashboard(
    model: nn.Module,
    library_weights: Tensor,
    hsi_image: np.ndarray | Tensor,
    wavelengths: np.ndarray | None = None,
    save_path: str = "output/viz/endmember_dashboard.pdf",
    device: str = "cuda",
) -> plt.Figure:
    model.eval()
    if isinstance(hsi_image, np.ndarray):
        hsi_image = torch.from_numpy(hsi_image).float().permute(2, 0, 1).unsqueeze(0)
    else:
        hsi_image = hsi_image.unsqueeze(0)
    hsi_image = hsi_image.to(device)
    with torch.no_grad():
        _, abundances = model(hsi_image)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 13,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
        }
    )

    weights_np = library_weights.detach().cpu().numpy()
    ab_np = abundances.detach().cpu().numpy()[0]

    num_symbols, num_bands = weights_np.shape
    if wavelengths is None:
        wavelengths = np.arange(num_bands)

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.4, 0.6], wspace=0.01)

    # Left Panel: Spectral Signatures
    ax_curves = fig.add_subplot(gs[0])

    # Right Panel: Abundance Maps
    if num_symbols % 5 == 0:
        n_cols_map = 5
    else:
        n_cols_map = 4
    n_rows_map = (num_symbols + n_cols_map - 1) // n_cols_map
    gs_maps = gs[1].subgridspec(n_rows_map, n_cols_map, hspace=0.11, wspace=0.05)

    id_cmap = plt.get_cmap("Paired")
    map_vmin, map_vmax = 0, ab_np.max()
    map_axes_list = []

    for i in range(num_symbols):
        color = id_cmap(i / (num_symbols - 1)) if num_symbols > 12 else id_cmap(i)

        ax_curves.plot(wavelengths, weights_np[i], color=color, linewidth=2.5, alpha=0.9, label=f"E. {i}")
        ax_map = fig.add_subplot(gs_maps[i // n_cols_map, i % n_cols_map])
        map_axes_list.append(ax_map)
        im = ax_map.imshow(ab_np[i], cmap="turbo", vmin=map_vmin, vmax=map_vmax)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_title(f"E. {i}", color=color, fontweight="bold", fontsize=15, pad=3)

        for spine in ax_map.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)

    # ax_curves.set_title("Spectral Signatures", fontweight="bold", fontsize=16)
    ax_curves.set_xlabel("Wavelength")
    ax_curves.set_ylabel("Reflectance")
    ax_curves.grid(True, which="major", linestyle="--", alpha=0.5)

    cax = fig.add_axes([0.91, 0.13, 0.005, 0.74])  # [left, bottom, width, height]
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Abundance Fraction", rotation=270, labelpad=15)

    if num_symbols <= 15:
        ax_curves.legend(loc="upper left", ncol=2, fontsize=14, framealpha=0.9)

    fig.canvas.draw()
    map_bboxes = [ax.get_position() for ax in map_axes_list]
    min_y0 = min([b.y0 for b in map_bboxes])  # Bottom-most edge
    max_y1 = max([b.y1 for b in map_bboxes])  # Top-most edge
    curve_pos = ax_curves.get_position()
    ax_curves.set_position([curve_pos.x0, min_y0, curve_pos.width, max_y1 - min_y0])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Dashboard saved to {save_path}")

    return fig


def get_hsi_plot(
    hr_tensor: Tensor,
    sr_tensor: Tensor,
    lr_tensor: Tensor | None = None,
    idx: int = 0,
    band_indices: tuple[int] = (50, 27, 10),
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 25,
        }
    )

    def to_numpy(t):
        return t.detach().cpu().permute(1, 2, 0).numpy()

    hr_np = to_numpy(hr_tensor)
    sr_np = to_numpy(sr_tensor)

    if lr_tensor is not None:
        lr_resized = torch.nn.functional.interpolate(
            lr_tensor.unsqueeze(0), size=hr_tensor.shape[1:], mode="nearest"
        ).squeeze(0)
        lr_np = to_numpy(lr_resized)
    else:
        lr_np = np.zeros_like(hr_np)

    r, g, b = band_indices

    def make_rgb(img_np):
        rgb = img_np[:, :, [r, g, b]]
        p2, p98 = np.percentile(rgb, (2, 98))
        return np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)

    img_hr_rgb = make_rgb(hr_np)
    img_sr_rgb = make_rgb(sr_np)
    img_lr_rgb = make_rgb(lr_np)
    spatial_error = np.mean(np.abs(sr_np - hr_np), axis=2)

    fig = plt.figure(figsize=(20, 6))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(1, 4),
        axes_pad=0.05,
        cbar_mode="each",
        cbar_location="right",
        cbar_size="5%",
        cbar_pad=0.1,
    )
    titles = ["LR", "SR", "GT", "Error"]
    images = [img_lr_rgb, img_sr_rgb, img_hr_rgb, spatial_error]

    for i, ax in enumerate(grid):
        if i == 3:
            im = ax.imshow(images[i], cmap=cmc.lajolla)
            cbar = plt.colorbar(im, cax=grid.cbar_axes[i])
            cbar.ax.tick_params(labelsize=20)
            cbar.set_label("MAE", rotation=270, labelpad=17)
        else:
            ax.imshow(images[i])
            grid.cbar_axes[i].axis("off")

        ax.set_title(titles[i], fontsize=25)
        ax.axis("off")
    plt.savefig(f"output/viz/pc-comp_{idx}.pdf", dpi=300, bbox_inches="tight")

    return fig


def visualize_stage2_results(lr_image: np.ndarray, gt_image: np.ndarray, model: nn.Module, num_symbols: int):
    model.eval()

    # 1. Get the LR Ground Truth (resized) for comparison
    H_lr, W_lr, _ = lr_image.shape
    gt_tensor = torch.from_numpy(gt_image).float().unsqueeze(0).unsqueeze(0)
    gt_lr = nn.functional.interpolate(gt_tensor, size=(H_lr, W_lr), mode="nearest").squeeze().numpy()

    # 2. Get the Predicted Abundance Maps
    lr_pixels_flat = lr_image.view(-1, lr_image.shape[2])
    with torch.no_grad():
        abundances_flat = model.encoder(lr_pixels_flat)

    abundance_maps = abundances_flat.view(H_lr, W_lr, num_symbols).cpu().numpy()
    num_rows = (num_symbols + 2) // 3
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), squeeze=False)
    fig.suptitle("Stage 2: Predicted Abundance Maps (Unmixed HR Image)", fontsize=16)

    for i in range(num_symbols):
        row, col = divmod(i, 3)
        ax = axes[row, col]
        im = ax.imshow(abundance_maps[:, :, i], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Symbol {i} Abundance")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

    ax = axes[num_rows - 1, -1]  # Put in the last available slot
    ax.imshow(gt_lr, cmap="jet")
    ax.set_title("Ground Truth (Resized)")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_single_lr_pixel(
    s2_model: nn.Module,
    lr_image: np.ndarray,
    hr_image: np.ndarray,
    hr_image_gt: np.ndarray,
    spatial_factor: int,
    lr_pixel_coords: tuple[int, int],
    wavelengths: np.ndarray | None = None,
    c: bool = False,
):
    s2_model.eval()
    symbolic_library = s2_model.decoder.weight.data.T.cpu().numpy()
    num_symbols, num_bands = symbolic_library.shape

    if wavelengths is None:
        wavelengths = np.arange(num_bands)

    lr_row, lr_col = lr_pixel_coords
    lr_pixel_spectrum = lr_image[lr_row, lr_col, :].unsqueeze(0)

    with torch.no_grad():
        reconstructed_spectrum, abundances = s2_model(lr_pixel_spectrum)

    reconstructed_spectrum = reconstructed_spectrum.squeeze().cpu().numpy()
    abundances = abundances.squeeze().cpu().numpy()
    lr_pixel_spectrum = lr_pixel_spectrum.squeeze().cpu().numpy()

    hr_row_start = lr_row * spatial_factor
    hr_row_end = (lr_row + 1) * spatial_factor
    hr_col_start = lr_col * spatial_factor
    hr_col_end = (lr_col + 1) * spatial_factor

    hr_image_rgb = create_rgb_image(hr_image, bands_rgb=[50, 27, 10])
    hr_gt_patch = hr_image_gt[hr_row_start:hr_row_end, hr_col_start:hr_col_end]
    hr_rgb_patch = hr_image_rgb[hr_row_start:hr_row_end, hr_col_start:hr_col_end, :]

    if c:
        gt_names = [
            "Bkg",
            "Water",
            "Trees",
            "Meadows",
            "Bricks",
            "Soil",
            "Asphalt",
            "Bitumen",
            "Tiles",
            "Shadows",
        ]
    else:
        gt_names = [
            "Bkg",
            "Asphalt",
            "Meadows",
            "Gravel",
            "Trees",
            "Metal",
            "Soil",
            "Bitumen",
            "Bricks",
            "Shadows",
        ]
    true_counts, _ = np.histogram(hr_gt_patch.flatten(), bins=range(11))

    pie_labels = gt_names
    pie_counts = true_counts
    non_zero_mask = pie_counts > 0
    pie_labels = np.array(pie_labels)[non_zero_mask]
    pie_counts = pie_counts[non_zero_mask]

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"Analysis of LR Pixel at ({lr_row}, {lr_col})", fontsize=20, y=1.03)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.5, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(hr_image_rgb)
    ax1.set_title("A: Full HR Image (Context)")
    rect = patches.Rectangle(
        (hr_col_start, hr_row_start),
        spatial_factor,
        spatial_factor,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax1.add_patch(rect)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    lr_image_rgb = create_rgb_image(lr_image)
    ax2.imshow(lr_image_rgb, interpolation="nearest")
    ax2.set_title("B: Full LR Image (Context)")
    rect_lr = patches.Rectangle((lr_col - 0.5, lr_row - 0.5), 1, 1, linewidth=2, edgecolor="r", facecolor="none")
    ax2.add_patch(rect_lr)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(hr_rgb_patch)
    ax3.set_title(f"C: HR Visual Patch ({spatial_factor}x{spatial_factor})")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[0, 3])
    im_gt = ax4.imshow(hr_gt_patch, cmap="jet", vmin=0, vmax=10)
    ax4.set_title("D: HR Ground Truth Labels")
    ax4.axis("off")
    patches_legend = [patches.Patch(color=plt.cm.jet(i / 10.0), label=gt_names[i]) for i in range(10)]
    ax4.legend(
        handles=patches_legend,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
    )

    ax5 = fig.add_subplot(gs[0, 4])
    if pie_counts.sum() > 0:
        pie_colors = [plt.cm.jet(gt_names.index(l) / 10.0) for l in pie_labels]
        ax5.pie(
            pie_counts,
            labels=pie_labels,
            autopct="%1.1f%%",
            colors=pie_colors,
            startangle=90,
        )
        ax5.set_title("E: True Composition (from GT)")
    else:
        ax5.set_title("E: True Composition (All Background)")
        ax5.text(
            0.5,
            0.5,
            "Background",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
        )
    ax5.axis("equal")

    ax6 = fig.add_subplot(gs[1, 0:2])
    symbol_indices = np.arange(num_symbols)
    colors = plt.cm.viridis(abundances / (abundances.max() + 1e-6))
    ax6.bar(symbol_indices, abundances, color=colors)
    ax6.set_title("F: Predicted Abundance (weights)")
    ax6.set_xlabel("Symbol Index")
    ax6.set_ylabel("Abundance (0.0 to 1.0)")
    ax6.set_xticks(symbol_indices)
    ax6.set_ylim(0, 1)

    ax7 = fig.add_subplot(gs[1, 2:5])
    ax7.set_title("G: Spectral Reconstruction")
    ax7.set_xlabel(f"Wavelength ({'nm' if wavelengths is not None else 'bands'})")
    ax7.set_ylabel("Normalized Reflectance")
    for i in range(num_symbols):
        if abundances[i] > 0.01:
            weighted_symbol = symbolic_library[i] * abundances[i]
            ax7.plot(
                wavelengths,
                weighted_symbol,
                label=f"Symbol {i} (Weight: {abundances[i]:.2f})",
                alpha=0.8,
            )
    ax7.plot(
        wavelengths,
        lr_pixel_spectrum,
        "k-",
        linewidth=2.5,
        label="Original LR Pixel (Input)",
    )
    ax7.plot(
        wavelengths,
        reconstructed_spectrum,
        "r--",
        linewidth=2.5,
        label="Reconstructed Pixel (Output)",
    )
    ax7.legend(loc="upper left", fontsize="small")
    ax7.grid(True, alpha=0.5)
    ax7.set_ylim(bottom=0)
    ax7.set_xlim(wavelengths.min(), wavelengths.max())

    plt.tight_layout(pad=0.5, rect=[0, 0, 1, 1])
    plt.show()


def plot_k_sensitivity(results: dict, save_path: str = "output/viz/k_sensitivity.pdf") -> None:
    """Generates a dual-axis plot of PSNR and SAM vs. K."""
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 14})

    k_vals = results["k"]
    psnr_vals = results["psnr"]
    sam_vals = results["sam"]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Number of Endmembers (K)", fontweight="bold")
    ax1.set_ylabel("PSNR (dB)", color=color, fontweight="bold")
    ax1.plot(k_vals, psnr_vals, marker="o", color=color, linewidth=2, markersize=8, label="PSNR")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.set_xticks(k_vals)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("SAM (Degrees)", color=color, fontweight="bold")
    ax2.plot(k_vals, sam_vals, marker="s", color=color, linewidth=2, markersize=8, label="SAM")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Impact of Spectral Library Size (K) on SR Performance")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Sensitivity plot saved to {save_path}")

    if wandb.run is not None:
        wandb.log({"k_sensitivity_plot": wandb.Image(fig)})
    plt.close(fig)


def analyze_endmember_utilization(
    model: nn.Module,
    hsi_image: np.ndarray | Tensor,
    k: int,
    save_path: str = "output/viz/endmember_utilization.pdf",
    device: str = "cuda",
) -> None:
    """Plots the mean abundance of each endmember to show pruning behavior."""
    model.eval()

    if isinstance(hsi_image, np.ndarray):
        hsi_image = torch.from_numpy(hsi_image).float().permute(2, 0, 1).unsqueeze(0)
    elif hsi_image.dim() == 3:
        hsi_image = hsi_image.unsqueeze(0)

    hsi_image = hsi_image.to(device)

    with torch.no_grad():
        _, abundances = model(hsi_image)

    ab_np = abundances.squeeze(0).cpu().numpy()
    mean_activations = np.mean(ab_np, axis=(1, 2))
    sorted_indices = np.argsort(mean_activations)[::-1]
    sorted_activations = mean_activations[sorted_indices]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 14})
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(k), sorted_activations, color="tab:blue", alpha=0.8, edgecolor="black")

    threshold = 0.01
    for i, bar in enumerate(bars):
        if sorted_activations[i] < threshold:
            bar.set_color("tab:red")
            bar.set_alpha(0.6)
            bar.set_edgecolor("black")

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"E.{idx}" for idx in sorted_indices], rotation=45)
    ax.set_ylabel("Mean Spatial Abundance")
    ax.set_xlabel("Endmembers (Sorted by Utilization)")
    ax.set_title(f"Endmember Utilization Distribution (K={k})")
    ax.axhline(y=threshold, color="black", linestyle="--", linewidth=1.5, label="1% Activation Threshold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Utilization plot saved to {save_path}")

    if wandb.run is not None:
        wandb.log({f"utilization_k{k}": wandb.Image(fig)})
    plt.close(fig)


def plot_constraint_comparison(
    maps_dict: dict[str, np.ndarray], 
    endmember_idx: int = 2, 
    save_path: str = "output/viz/constraint_ablation.pdf"
) -> None:
    """
    Plots a specific endmember's abundance map across the 4 constraint modes.
    """
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 14})
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    modes = ["both", "anc_only", "asc_only", "none"]
    titles = ["Softmax (ANC + ASC)", "ReLU (ANC Only)", "Hyperplane (ASC Only)", "Identity (None)"]
    
    for i, mode in enumerate(modes):
        ax = axes[i]
        # Extract the 2D spatial map for the chosen endmember
        ab_map = maps_dict[mode][endmember_idx]
        
        # Calculate the actual min and max for this specific map to show the severity
        min_val, max_val = ab_map.min(), ab_map.max()
        
        # Force the colormap to center on 0 so negative values are distinctly colored
        limit = max(abs(min_val), abs(max_val), 1.0)
        
        # Use RdBu_r: Red is positive (high abundance), White is zero, Blue is negative (impossible)
        im = ax.imshow(ab_map, cmap="RdBu_r", vmin=-limit, vmax=limit)
        
        ax.set_title(titles[i], fontweight="bold")
        ax.axis("off")
        
        # Add a text box showing the min/max range
        text_str = f"Range: [{min_val:.2f}, {max_val:.2f}]"
        if min_val < -0.01: # Flag negative violations
            text_str += "\n(Negative Violation)"
            text_color = "red"
        elif max_val > 1.01: # Flag ASC violations
            text_str += "\n(Sum Violation)"
            text_color = "darkorange"
        else:
            text_color = "black"
            
        ax.text(0.05, 0.05, text_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                color=text_color)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.suptitle(f"Impact of Physical Constraints on Endmember {endmember_idx} Spatial Distribution", y=1.05, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Constraint visualization saved to {save_path}")
    
    if wandb.run is not None:
        wandb.log({"constraint_visual_comparison": wandb.Image(fig)})
    plt.close(fig)


def plot_kmeans_metrics(
    k_values: list[int], 
    inertias: list[float], 
    silhouette_scores: list[float],
    hysime_k: int | None = None,
    save_path: str = "output/viz/kmeans_metrics.pdf"
) -> None:
    """Plots K-Means inertia and Silhouette Score, overlaid with the HySime estimate."""
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 14
    })
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Endmembers / Clusters (K)', fontweight='bold')
    ax1.set_ylabel('Inertia (Sum of Squared Errors)', color=color1, fontweight='bold')
    line1 = ax1.plot(k_values, inertias, marker='o', color=color1, linewidth=2, markersize=8, label="Inertia")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xticks(k_values)
    
    ax2 = ax1.twinx()
    color2 = 'tab:green'
    ax2.set_ylabel('Silhouette Score', color=color2, fontweight='bold')
    line2 = ax2.plot(k_values, silhouette_scores, marker='s', color=color2, linewidth=2, markersize=8, label="Silhouette Score")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = line1 + line2
    if hysime_k is not None:
        line3 = ax1.axvline(x=hysime_k, color='black', linestyle='-.', linewidth=2, label=f"HySime Estimate (K={hysime_k})")
        lines.append(line3)
        
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    
    plt.title("Intrinsic Dimensionality: K-Means vs. HySime")
    fig.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"K-Means + HySime metrics plot saved to {save_path}")
    
    if wandb.run is not None:
        wandb.log({"kmeans_hysime_plot": wandb.Image(fig)})
    plt.close(fig)
