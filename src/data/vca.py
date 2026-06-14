import numpy as np
from scipy.sparse.linalg import svds


def extract_endmembers_vca(hsi_data: np.ndarray, num_endmembers: int) -> np.ndarray:
    if hsi_data.ndim == 3:
        h, w, bands = hsi_data.shape
        Y = hsi_data.reshape(h * w, bands).T
    else:
        Y = hsi_data
        bands, _ = Y.shape

    U, _, _ = svds(Y.astype(np.float64), k=num_endmembers)
    X = U.T @ Y

    indices = np.zeros(num_endmembers, dtype=int)
    A = np.zeros((num_endmembers, num_endmembers))
    A[-1, 0] = 1.0

    for i in range(num_endmembers):
        w = np.random.randn(num_endmembers, 1)

        f = w - A @ np.linalg.pinv(A) @ w
        f = f / np.linalg.norm(f)

        v = np.abs(f.T @ X)
        idx = np.argmax(v)
        indices[i] = idx

        A[:, i] = X[:, idx]

    endmembers = Y[:, indices].T
    return endmembers
