import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VanillaRNNDiagonalMetric(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = 0.03
        self.hidden_size = 10

        # leak rate of neurons
        self.gamma = 0.01

        # diagonal metric to be learned
        self.Phi = nn.Parameter(
            torch.normal(1, 1 / np.sqrt(self.hidden_size), (self.hidden_size,))
        )

        # to parameterize weight matrix
        self.B = nn.Parameter(
            torch.normal(
                1, 1 / np.sqrt(self.hidden_size), (self.hidden_size, self.hidden_size)
            )
        )

        # recurrence used
        self.recurrence = nn.RNN(10, 20, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # project matrix B so that W = Phi @ B @ Phi^-1 is contracting in diagonal metric Phi^-2
        with torch.no_grad():
            # clip singular values here
            # We need ||Phi W Phi^-1|| < gamma
            # u,s,v = torch.svd(B)
            # s_clip = torch.clamp(s,0,0.99)
            # W = Phi @ (u @ s_clip @ v) @ Phi^-1

            d = None

        x = (1 - self.gamma * self.alpha) * x + self.alpha * self.recurrence(x)

        return x
