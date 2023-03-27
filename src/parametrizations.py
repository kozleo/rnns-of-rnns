import torch
from torch import nn, Tensor

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)



class SymmetricStable(nn.Module):
    def __init__(self, n: int, epsilon: float, device):
        """Parameterization for symmetric matrix

        with eigenvalues strictly less than unity.

        Args:
            n (int): Dimension of matrix.
            epsilon (float): Enforces strict inequality.
        """
        super().__init__()

        self.register_buffer("Id", torch.eye(n, device=device))

    def forward(self, W: Tensor) -> Tensor:
        # I - W.T @ W
        return self.Id - W.T @ W - 1e-5 * self.Id


class InterarealMaskedAndStable(nn.Module):
    def __init__(self, n: int, M_hat: Tensor, B_mask: Tensor, device):
        super().__init__()
        self.device = device

        self.register_buffer("Id", torch.eye(n, device=self.device))
        self.register_buffer("B_mask", B_mask)
        self.register_buffer("M_hat", M_hat)
        self.register_buffer("M_hat_inv", torch.linalg.inv(M_hat))

    def forward(self, B: Tensor) -> Tensor:
        # L = B - M @ B @ M^-1
        return (B * self.B_mask) - self.M_hat @ (B * self.B_mask).T @ self.M_hat_inv

