import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Optional, overload


class VanillaRNNDiagonalMetric(nn.Module):
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
        if self.nonlinearity == "tanh":
            self.phi = torch.tanh

        if self.nonlinearity == "relu":
            self.phi = F.relu

        if self.nonlinearity == "none":
            print("Nl = none")
            self.phi = torch.nn.Identity()

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

        # initialize the input-to-hidden weights
        self.weight_ih = nn.Parameter(
            torch.normal(
                0, 1 / np.sqrt(self.hidden_size), (self.hidden_size, self.input_size)
            )
        )

        # initialize the hidden-to-output weights
        self.weight_ho = nn.Parameter(
            torch.normal(
                0, 1 / np.sqrt(self.hidden_size), (self.output_size, self.hidden_size)
            )
        )

        # initialize the output bias weights
        self.bias_oh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(self.hidden_size), (1, self.output_size))
        )

        # initialize the hidden bias weights
        self.bias_hh = nn.Parameter(
            torch.normal(0, 1 / np.sqrt(self.hidden_size), (1, self.hidden_size))
        )

        # output weights
        self.W_hidden_to_out = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(input.shape[0], self.hidden_size)

        # for storing RNN outputs and hidden states
        outputs = []

        # loop over input
        for i in range(input.shape[1]):
            # compute output
            hy = state @ self.weight_ho.T + self.bias_oh

            # save output and hidden states
            outputs += [hy]

            # compute the RNN update
            fx = -state + self.phi(
                state @ (torch.diag(self.Phi**-1) @ self.B @ torch.diag(self.Phi)).T
                + input[:, i, :] @ self.weight_ih.T
                + self.bias_hh
            )

            # step hidden state foward using Euler discretization
            state = state + self.alpha * (fx)

        # organize states and outputs and return
        return torch.stack(outputs).permute(1, 0, 2)


import torch.nn.utils.parametrize as parametrize


class SymmetricStable(nn.Module):
    def __init__(self, n: int, epsilon: float):
        super().__init__()

        self.register_buffer("Id", torch.eye(n))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # I - W.T @ W
        return self.Id - W.T @ W - 1e-5 * self.Id


class VanillaRNNDiagonalMetricTorchCell(nn.Module):
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


# TODO: Write method to splice weights from pretrained RNNs into this RNN of RNNs. For the RNN of RNNs we only train the interareal conenctions so remember to turn off gradients.
class GlobalWorkSpaceRNNofRNNs(nn.Module):
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

        self.M_hat = None
        self.B_mask = None

        self.L_hat = nn.Linear(in_features=100, out_features=100, bias=False)

        # list of rnns
        # self.rnn

        # parameterize L_hat
        parametrize.register_parametrization(
            self.L_hat,
            "weight",
            InterarealMaskedAndStable(
                n=self.hidden_size, M_hat=self.M_hat, B_mask=self.B_mask
            ),
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
