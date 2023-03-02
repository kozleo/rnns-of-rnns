import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Optional, overload
from src import utils
import torch.nn.utils.parametrize as parametrize


class SymmetricStable(nn.Module):
    def __init__(self, n: int, epsilon: float):
        """Parameterization for symmetric matrix

        with eigenvalues strictly less than unity.

        Args:
            n (int): Dimension of matrix.
            epsilon (float): Enforces strict inequality.
        """
        super().__init__()

        self.register_buffer("Id", torch.eye(n))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # I - W.T @ W
        return self.Id - W.T @ W - 1e-5 * self.Id


class VSRNN(nn.Module):
    # Vanilla Symmetric RNN (VSRNN)
    def __init__(self, param_dict: dict):
        super().__init__()

        # timestep for Euler integration
        self.alpha = param_dict["alpha"]

        # number of input, hidden, and output
        self.input_size = param_dict["input_size"]
        self.hidden_size = param_dict["hidden_size"]
        self.output_size = param_dict["output_size"]

        # leak rate of neurons
        self.gamma = param_dict["gamma"]

        # set nonlinearity of the vanilla RNN
        self.nonlinearity = param_dict["nonlinearity"]

        # recurrence
        self.rnn = nn.RNNCell(
            self.input_size, self.hidden_size, nonlinearity=self.nonlinearity
        )

        # parameterize W = I - A.T @ A - epsilon*I
        parametrize.register_parametrization(
            self.rnn, "weight_hh", SymmetricStable(n=self.hidden_size, epsilon=1e-4)
        )

        # output weights
        self.W_hidden_to_out = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = []

        # Initialize hidden state
        hx = torch.zeros(input.shape[0], self.hidden_size)

        # loop over input
        for i in range(input.shape[1]):
            fx = -hx + self.rnn(input[:, i, :], hx)
            hx = hx + self.alpha * (fx)
            y = self.W_hidden_to_out(hx)

            outputs += [y]

        # organize outputs and return
        return torch.stack(outputs).permute(1, 0, 2)

    def _check_stability(self, stability_type="symmetric"):
        with torch.no_grad():
            # check if svd condition is met
            if stability_type == "svd":
                s = torch.linalg.svdvals(
                    torch.diag(self.Phi)
                    @ self.rnn.weight_hh
                    @ torch.diag(self.Phi**-1)
                )
                if torch.any(s > 1):
                    print("RNN is not provably stable.")
                else:
                    print("RNN is provably stable in given metric.")
                return s

            # check if symmetric eigenvalue condition is met
            if stability_type == "symmetric":
                e, _ = torch.linalg.eigh(self.rnn.weight_hh)
                if torch.any(e > 1):
                    print("RNN is not provably stable with symmetric condition. ")
                else:
                    print("RNN is provably stable with symmetric condition.")
                return e


class InterarealMaskedAndStable(nn.Module):
    def __init__(self, n: int, M_hat: torch.Tensor, B_mask: torch.Tensor):
        super().__init__()

        self.register_buffer("Id", torch.eye(n))
        self.register_buffer("B_mask", B_mask)
        self.register_buffer("M_hat", M_hat)
        self.register_buffer("M_hat_inv", torch.linalg.inv(M_hat))

    def forward(self, B: torch.Tensor) -> torch.Tensor:
        # L = B - M @ B @ M^-1
        return (B * self.B_mask) - self.M_hat @ (B * self.B_mask).T @ self.M_hat_inv


class GWRNN(nn.Module):
    def __init__(self, param_dict: dict):
        super().__init__()

        # timestep for Euler integration
        self.alpha = param_dict["alpha"]

        # number of input, hidden, and output
        self.ns = param_dict["ns"]

        self.input_size = param_dict["input_size"]
        self.hidden_size = sum(self.ns)
        self.output_size = param_dict["output_size"]

        # leak rate of neurons
        self.gamma = param_dict["gamma"]

        # set nonlinearity of the vanilla RNN
        self.nonlinearity = param_dict["nonlinearity"]

        # pretrained hidden-to-hidden, input-to-hidden, and hidden-to-output weights
        self.W_hh = param_dict["W_hh"]
        self.W_ih = param_dict["W_ih"]
        self.W_ho = param_dict["W_ho"]

        self.rnn = nn.RNNCell(
            self.input_size, self.hidden_size, nonlinearity=self.nonlinearity
        )

        self.readout = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size
        )

        # use pretrained subnetwork weights
        if self.W_hh is not None:
            self.rnn.weight_hh = nn.Parameter(self.W_hh)
        if self.W_ih is not None:
            self.rnn.weight_ih = nn.Parameter(self.W_ih)
        if self.W_ho is not None:
            self.readout.weight = nn.Parameter(self.W_ho)

        # Do not train hidden to hidden weights. Input weights and biases can be trained.
        self.rnn.weight_hh.requires_grad = False

        # parameterize interareal connectivity matrix
        self.L_hat = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=False
        )

        self.M_hat = param_dict["M_hat"]
        self.B_mask = param_dict["B_mask"]

        parametrize.register_parametrization(
            self.L_hat,
            "weight",
            InterarealMaskedAndStable(
                n=self.hidden_size, M_hat=self.M_hat, B_mask=self.B_mask
            ),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        outputs = []

        # Initialize hidden state
        hx = torch.zeros(input.shape[0], self.hidden_size)

        # loop over input
        for i in range(input.shape[1]):
            fx = -hx + self.rnn(input[:, i, :], hx) + self.L_hat(hx)
            hx = hx + self.alpha * (fx)
            y = self.readout(hx)
            outputs += [y]

        # organize outputs and return
        return torch.stack(outputs).permute(1, 0, 2)
