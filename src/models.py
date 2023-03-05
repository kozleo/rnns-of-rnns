import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple, Optional, overload
from src import utils
import torch.nn.utils.parametrize as parametrize


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

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        # I - W.T @ W
        return self.Id - W.T @ W - 1e-5 * self.Id


import torch.jit as jit


class vRNNLayer(nn.Module):
    """Vanilla RNN layer in continuous time."""

    def __init__(self, param_dict: dict):
        super(vRNNLayer, self).__init__()

        self.device = param_dict["device"]

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

        # set nonlinearity of the vRNN
        if self.nonlinearity == "tanh":
            self.phi = torch.tanh
        if self.nonlinearity == "relu":
            self.phi = F.relu
        if self.nonlinearity == "none":
            print("Nl = none")
            self.phi = torch.nn.Identity()

        # initialize weights
        self.W_in = nn.Linear(
            in_features=self.input_size, out_features=self.hidden_size
        )
        self.W_hh = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size
        )
        self.W_out = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size
        )

    def forward(self, input: Tensor) -> Tensor:
        # initialize state at the origin. randn is there just in case we want to play with this later.
        hx = torch.zeros(
            size=(input.shape[1], self.hidden_size),
            device=self.device,
            requires_grad=False,
        )

        inputs = input.unbind(0)

        # for storing RNN outputs and hidden states
        outputs = []  # torch.jit.annotate(List[torch.Tensor], [])

        # loop over input
        for i in range(len(inputs)):
            # save output and hidden states
            outputs += [self.W_out(hx)]

            # compute the RNN update
            fx = -hx + self.phi(self.W_hh(hx) + self.W_in(inputs[i]))

            # step hidden state foward using Euler discretization
            hx = hx + self.alpha * (fx)

        # organize states and outputs and return
        return torch.stack(outputs)  # .permute(1, 0, 2)


'''
class VSRNN(jit.ScriptModule):
    # Vanilla Symmetric RNN (VSRNN)
    def __init__(self, param_dict: dict):
        super().__init__()

        self.device = param_dict["device"]

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
        # self.rnn.weight_hh = nn.Parameter(torch.eye(self.hidden_size))

        # self.rnn.to(self.device)

        """
        # parameterize W = I - A.T @ A - epsilon*I
        parametrize.register_parametrization(
            self.rnn,
            "weight_hh",
            SymmetricStable(n=self.hidden_size, epsilon=1e-4, device=self.device),
        )
        """

        # output weights
        self.W_hidden_to_out = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size
        )

    @jit.script_method
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        inputs = input.unbind(1)

        outputs = torch.jit.annotate(List[torch.Tensor], [])

        # Initialize hidden state
        hx = (1 / self.hidden_size) * torch.randn(
            input.shape[0], self.hidden_size, device=self.device
        )

        # loop over input
        for i in range(input.shape[1]):
            fx = -hx + self.rnn(inputs[i], hx)
            hx = hx + self.alpha * (fx)
            outputs += [self.W_hidden_to_out(hx)]

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
'''


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


# TODO: Check if it is faster to put RNNs and weights in a list and loop through, rather than vectorize
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
        for t in range(input.shape[1]):
            fx = -hx + self.rnn(input[:, t, :], hx) + self.L_hat(hx)
            hx = hx + self.alpha * (fx)
            y = self.readout(hx)
            outputs += [y]

        # organize outputs and return
        return torch.stack(outputs).permute(1, 0, 2)


# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math


class CTRNN(nn.Module):
    """Continuous-time RNN.

    Parameters:
        input_size: Number of input neurons
        hidden_size: Number of hidden neurons
        dt: discretization time step in ms.
            If None, dt equals time constant tau

    Inputs:
        input: tensor of shape (seq_len, batch, input_size)
        hidden: tensor of shape (batch, hidden_size), initial hidden activity
            if None, hidden is initialized through self.init_hidden()

    Outputs:
        output: tensor of shape (seq_len, batch, hidden_size)
        hidden: tensor of shape (batch, hidden_size), final hidden activity
    """

    def __init__(self, input_size, hidden_size, device, dt=None, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.device = device

        self.input2h = nn.Linear(input_size, hidden_size, device=self.device)
        self.h2h = nn.Linear(hidden_size, hidden_size, device=self.device)

        self.h2h.weight = nn.Parameter(torch.eye(self.hidden_size, device=self.device))

        parametrize.register_parametrization(
            self.h2h,
            "weight",
            SymmetricStable(n=self.hidden_size, epsilon=1e-4, device=self.device),
        )

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size)

    def recurrence(self, input, hidden):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden))
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
            hidden = self.init_hidden(input.shape).to(input.device)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden


class RNNNet(nn.Module):
    """Recurrent network model.

    Parameters:
        input_size: int, input size
        hidden_size: int, hidden size
        output_size: int, output size

    Inputs:
        x: tensor of shape (Seq Len, Batch, Input size)

    Outputs:
        out: tensor of shape (Seq Len, Batch, Output size)
        rnn_output: tensor of shape (Seq Len, Batch, Hidden size)
    """

    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__()

        # Continuous time RNN
        self.rnn = CTRNN(input_size, hidden_size, **kwargs)

        # nn.utils.spectral_norm(self.rnn.h2h, "weight")

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output
