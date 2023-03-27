import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import List, Tuple, Optional, overload
from src import utils, parametrizations
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import orthogonal, spectral_norm


"""
# Define networks
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import math
"""


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
            parametrizations.InterarealMaskedAndStable(
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


class CTRNN(nn.Module):
    """Continuous-time RNN. Forked from Robert Yang's implementation.

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

    def __init__(
        self, input_size, hidden_size, device, dt=None, constraint="None", **kwargs
    ):
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

        # Initialize as identity (Le et al, 2015, "A Simple to Initialize Recurrent Networks of Rectified Linear Units")
        self.h2h.weight = nn.Parameter(torch.eye(self.hidden_size, device=self.device))

        if constraint == "spectral":
            # constrain weights to be semi-contractive
            spectral_norm(self.h2h, name="weight")

        if constraint == "sym":
            # constrain weights to be symmetric and have eigenvalues less than unity
            parametrize.register_parametrization(
                self.h2h,
                "weight",
                parametrizations.SymmetricStable(
                    n=self.hidden_size, epsilon=1e-4, device=self.device
                ),
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

        # Add an output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output




class GW_CTRNN(nn.Module):
    """Continuous-time RNN. Forked from Robert Yang's implementation.

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

    def __init__(
        self, stacked_wb, input_size, ns, device, M_hat, B_mask, dt = None, **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.ns = ns
        self.hidden_size = sum(self.ns)
        self.tau = 100
        if dt is None:
            alpha = 1
        else:
            alpha = dt / self.tau
        self.alpha = alpha
        self.device = device

        # load in pretrained weights and biases
        # self.input2h_weight_bar = torch.block_diag(*stacked_wb['input2h_weight_bar'])
        # self.rnn_input2h_bias = stacked_wb['rnn_input2h_bias']
        self.rnn_h2h_weight = torch.block_diag(*stacked_wb["rnn_h2h_weight"])
        self.rnn_h2h_bias = torch.cat(stacked_wb["rnn_h2h_bias"], dim=0)

        # self.fc_weight_bar = torch.block_diag(*stacked_wb['fc_weight_bar'])
        # self.fc_bias_bar = stacked_wb['fc_bias_bar']

        self.input2h = nn.Linear(input_size, self.hidden_size, device=self.device)

        self.h2h = nn.Linear(self.hidden_size, self.hidden_size, device=self.device)
        self.h2h.weight = nn.Parameter(self.rnn_h2h_weight)
        self.h2h.bias = nn.Parameter(self.rnn_h2h_bias)
        self.h2h.weight.requires_grad = False
        self.h2h.bias.requires_grad = True
        

        # parameterize interareal connectivity matrix
        self.L_hat = nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias=True, device = self.device
        )

        self.M_hat = M_hat
        self.B_mask = B_mask

        #spectral_norm(self.L_hat, name = "weight")


            
        parametrize.register_parametrization(
            self.L_hat,
            "weight",
            parametrizations.InterarealMaskedAndStable(
                n = self.hidden_size,
                M_hat = self.M_hat,
                B_mask = self.B_mask,
                device = self.device,
            ),
        )
        

        #self.L_hat.weight.requires_grad = True

    def init_hidden(self, input_shape):
        batch_size = input_shape[1]
        return torch.zeros(batch_size, self.hidden_size,device = self.device)

    def recurrence(self, input, hidden):
        """Run network for one time step.

        Inputs:
            input: tensor of shape (batch, input_size)
            hidden: tensor of shape (batch, hidden_size)

        Outputs:
            h_new: tensor of shape (batch, hidden_size),
                network activity at the next time step
        """
        h_new = torch.relu(self.input2h(input) + self.h2h(hidden)) + self.L_hat(hidden)
        h_new = hidden * (1 - self.alpha) + h_new * self.alpha
        return h_new

    def forward(self, input, hidden=None):
        """Propogate input through the network."""

        # If hidden activity is not provided, initialize it
        if hidden is None:
           # hidden = self.init_hidden(input.shape).to(input.device)
           hidden = self.init_hidden(input.shape)

        # Loop through time
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)

        # Stack together output from all time steps
        output = torch.stack(output, dim=0)  # (seq_len, batch, hidden_size)
        return output, hidden


class GW_RNNNet(nn.Module):
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

    def __init__(
        self,
        stacked_wb,
        input_size,
        ns,
        output_size,
        M_hat,
        B_mask,
        device,
        dt,
        **kwargs
    ):
        super().__init__()

        self.input_size = input_size
        self.ns = ns
        self.hidden_size = sum(self.ns)
        self.output_size = output_size
        self.device = device
        self.M_hat = M_hat
        self.B_mask = B_mask
        self.dt = dt

        # Continuous time RNN
        self.rnn = GW_CTRNN(
            stacked_wb,
            input_size=self.input_size,
            ns=self.ns,
            output_size=self.output_size,
            device=self.device,
            M_hat=self.M_hat,
            B_mask=self.B_mask,
            dt=self.dt,
        )

        # Add an output layer
        self.fc = nn.Linear(self.hidden_size, self.output_size,device = self.device)

    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output
