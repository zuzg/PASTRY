import mat73
import numpy as np
import scipy.io as sio
import spectral.io.envi as envi
import torch
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset


def center_crop_3d(data: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Center crops a 3D array to the target shape (H, W, C).
    """
    d_h, d_w, d_c = data.shape
    t_h, t_w = target_shape

    if d_h < t_h or d_w < t_w:
        raise ValueError(f"Image size ({d_h}, {d_w}) smaller than target ({t_h}, {t_w})")

    start_h = (d_h - t_h) // 2
    start_w = (d_w - t_w) // 2

    return data[start_h : start_h + t_h, start_w : start_w + t_w, :]


class HSIDataset(Dataset):
    def __init__(
        self,
        data_path,
        dataset_name="pavia",
        mode="train",
        scale_factor=4,
        patch_size=32,
        stride=None,
        augment=True,
    ):
        super(HSIDataset, self).__init__()
        self.mode = mode
        self.dataset_name = dataset_name.lower()
        self.scale_factor = scale_factor

        if mode == "test":
            if "chikusei" in self.dataset_name:
                self.patch_size = 128
            elif "paviac" in self.dataset_name:
                self.patch_size = 220
            elif "pavia" in self.dataset_name:
                self.patch_size = 144
            elif "cuprite" in self.dataset_name:
                self.patch_size = 128  # Custom default for Cuprite
            elif "houston" in self.dataset_name:
                self.patch_size = 128  # Custom default for Houston
            else:
                self.patch_size = patch_size
        else:
            self.patch_size = patch_size
        self.augment = augment and (mode == "train")

        full_image = self._load_data(data_path)
        full_image = full_image.astype(np.float32)

        max_val = np.max(full_image)
        if max_val > 0:
            full_image = full_image / max_val

        self.full_image = full_image
        self.image = self._apply_split(full_image)

        if stride is None:
            if "chikusei" in self.dataset_name and mode == "train":
                self.stride = 32
            elif "paviac" in self.dataset_name and mode == "train":
                self.stride = 6
            else:
                self.stride = self.patch_size // 2 if mode == "train" else self.patch_size
        else:
            self.stride = stride

        self.patches = self._extract_patch_coords()

        logger.info(f"[{dataset_name.upper()}] {mode} set loaded.")
        logger.info(f"Split Shape: {self.image.shape}")
        logger.info(f"Num Patches: {len(self.patches)}")

    def _extract_mat_data(self, path, possible_keys):
        """Robust utility to extract 3D hyperspectral arrays from .mat files."""
        try:
            data = mat73.loadmat(path)
        except Exception:
            data = sio.loadmat(path)

        for key in possible_keys:
            if key in data:
                return data[key]

        # Fallback: automatically find the largest 3D numpy array
        max_size = 0
        best_arr = None
        for key, val in data.items():
            if isinstance(val, np.ndarray) and len(val.shape) == 3:
                if val.size > max_size:
                    max_size = val.size
                    best_arr = val

        if best_arr is not None:
            logger.info(f"Used fallback extraction for .mat file. Found 3D array.")
            return best_arr

        raise ValueError(f"Could not find valid 3D data in {path}. Keys: {list(data.keys())}")

    def _load_data(self, path):
        """Loads data and handles dataset-specific pre-processing."""
        name = self.dataset_name

        if "chikusei" in name:
            img = self._extract_mat_data(path, ["chikusei", "data"])
            img = center_crop_3d(img, (512, 512))
            return img[:, :, :128]

        elif "paviac" in name:
            return self._extract_mat_data(path, ["pavia", "paviaC"])

        elif "pavia" in name:
            return self._extract_mat_data(path, ["paviaU", "pavia"])

        elif "cuprite" in name:
            img = self._extract_mat_data(path, ["cuprite", "x", "X", "Y", "data"])

            # 1. Handle negative values (clip non-physical reflectance to 0)
            img = np.clip(img, a_min=0, a_max=None)

            # 2. Remove noisy and water absorption channels
            # Original bands (1-indexed): 1-2, 104-113, 148-167, 221-224
            # Python indices (0-indexed): 0-1, 103-112, 147-166, 220-223
            remove_indices = list(range(0, 2)) + list(range(103, 113)) + list(range(147, 167)) + list(range(220, 224))

            # Keep only the valid 188 channels
            keep_indices = [i for i in range(img.shape[2]) if i not in remove_indices]
            img = img[:, :, keep_indices]

            return img

        elif "houston" in name:
            if str(path).endswith(".hdr"):
                img_obj = envi.open(path, image=str(path).replace(".hdr", ".pix"))
                img = img_obj.load()
                img = img[:, :, :48]
                img = np.clip(img, a_min=0, a_max=None)
                return img
            else:
                return self._extract_mat_data(path, ["houston", "Houston18", "houston18", "ori_data", "data"])

        else:
            raise ValueError(f"Unknown dataset name: {name}")

    def _apply_split(self, full_image):
        """Implements the standard Train/Test splits."""
        H, W, C = full_image.shape

        # --- PAVIA CENTER ---
        if "paviac" in self.dataset_name:
            split_col = 223
            if self.mode == "train":
                return full_image[:, split_col:, :]
            else:
                return full_image[:, :split_col, :]

        # --- CHIKUSEI ---
        elif "chikusei" in self.dataset_name:
            split_row = 128
            if self.mode == "train":
                return full_image[split_row:, :, :]
            else:
                return full_image[:split_row, :256, :]

        # --- PAVIA C ---
        elif "pavia" in self.dataset_name:
            split_row = 300
            if self.mode == "train":
                return full_image[:split_row, :, :]
            else:
                return full_image[split_row:, :, :]

        # --- AVIRIS CUPRITE ---
        elif "cuprite" in self.dataset_name:
            # Cuprite lacks a standardized SR split. Using a 50/50 vertical split.
            split_row = H // 2
            if self.mode == "train":
                return full_image[:split_row, :, :]
            else:
                return full_image[split_row:, :, :]

        # --- HOUSTON 2018 ---
        elif "houston" in self.dataset_name:
            split_col = 4044
            if self.mode == "train":
                return full_image[:, :split_col, :]
            else:
                return full_image[:, split_col:, :]

        return full_image

    def _extract_patch_coords(self):
        """Standard grid patching."""
        h, w, _ = self.image.shape
        patches = []

        if self.stride <= 0:
            raise ValueError("Stride must be > 0")

        h_steps = (h - self.patch_size) // self.stride + 1
        w_steps = (w - self.patch_size) // self.stride + 1

        for i in range(h_steps):
            for j in range(w_steps):
                r = i * self.stride
                c = j * self.stride
                patches.append((r, c))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        r, c = self.patches[index]

        hr_patch_np = self.image[r : r + self.patch_size, c : c + self.patch_size, :]
        hr_tensor = torch.from_numpy(hr_patch_np).permute(2, 0, 1).float()  # [C, H, W]

        if self.augment:
            # Random Horizontal Flip
            if torch.rand(1) < 0.5:
                hr_tensor = torch.flip(hr_tensor, [2])
            # Random Vertical Flip
            if torch.rand(1) < 0.5:
                hr_tensor = torch.flip(hr_tensor, [1])
            # Random Rotation (0, 90, 180, 270)
            k = torch.randint(0, 4, (1,)).item()
            hr_tensor = torch.rot90(hr_tensor, k, [1, 2])

        hr_batch = hr_tensor.unsqueeze(0)
        lr_batch = F.interpolate(hr_batch, scale_factor=1 / self.scale_factor, mode="bicubic", align_corners=False)
        lr_tensor = lr_batch.squeeze(0)

        return {"lr": lr_tensor, "hr": hr_tensor}
