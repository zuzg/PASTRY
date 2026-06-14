import torch
from torch import Tensor, nn


class AbundanceConstraint(nn.Module):
    """
    Modes: 'both' (Softmax), 'anc_only' (ReLU), 'asc_only' (Hyperplane Projection), 'none' (Identity)
    """
    def __init__(self, mode: str = "both"):
        super(AbundanceConstraint, self).__init__()
        self.mode = mode.lower()

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "both":
            # Enforces ANC (x > 0) and ASC (sum(x) == 1)
            return torch.softmax(x, dim=1)
            
        elif self.mode == "anc_only":
            # Enforces ANC (x >= 0) but allows sum != 1
            return torch.relu(x)
            
        elif self.mode == "asc_only":
            # Enforces ASC (sum(x) == 1) but allows negative values
            # Distributes the difference from 1 equally across all K channels
            K = x.shape[1]
            channel_sums = x.sum(dim=1, keepdim=True)
            return x + (1.0 - channel_sums) / K
            
        elif self.mode == "none":
            # Unconstrained latent representation
            return x
            
        else:
            raise ValueError(f"Unknown constraint mode: {self.mode}")
