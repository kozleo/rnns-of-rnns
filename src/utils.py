import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.jit as jit
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, Subset
from torch._utils import _accumulate
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split

# from tqdm.auto import tqdm

import numpy as np
import numpy.random as npr
import scipy.sparse
import scipy.integrate

import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from src import models


def pickle_store(filename, your_data, path="./"):
    # Store data (serialize)

    filepath = path + filename + ".pickle"

    with open(filepath, "wb") as handle:
        pickle.dump(your_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(filename, path="./"):
    # Load data (deserialize)
    filepath = path + filename + ".pickle"

    with open(filepath, "rb") as handle:
        unserialized_data = pickle.load(handle)

    return unserialized_data


# function that checks the Theorem 1 condition for an input square matrix
def check_cond(W):
    W_diag_only = np.diag(np.diag(W))
    W_diag_pos_only = W_diag_only.copy()
    W_diag_pos_only[W_diag_pos_only < 0] = 0.0
    W_abs_cond = np.abs(W - W_diag_only) + W_diag_pos_only
    max_eig_abs_cond = np.max(np.real(np.linalg.eigvals(W_abs_cond)))
    if max_eig_abs_cond < 1:
        return True
    else:
        return False


# sampling function to use for each non-zero element in a generated matrix
# (uniform but between -1 and 1 instead of default 0 to 1)
def uniform_with_neg(x):
    return np.random.uniform(low=-1.0, high=1.0, size=x)


# function that creates a square matrix of a given size with a given density and distribution (+ scalar for the distribution)
# also zeroes out the diagonal after generation
# the uniform_with_neg function is used as the sampling function by default
def generate_random_sparse_matrix(
    num_units, density, dist_multiplier, dist_func=uniform_with_neg
):
    test = scipy.sparse.random(
        num_units, num_units, density=density, format="csr", data_rvs=dist_func
    )
    np_test = test.toarray()
    np_test = dist_multiplier * np_test
    np.fill_diagonal(np_test, 0)
    return np_test


# create a full set of RNN modules using the above functions
# will only keep a given random RNN if it meets the condition for theorem 1
def create_modules(module_sizes, density, dist_multiplier, post_select_multiplier):
    modules = []
    for m in module_sizes:
        okay = False
        while not okay:
            cur_matrix = generate_random_sparse_matrix(m, density, dist_multiplier)
            okay = check_cond(cur_matrix)
        cur_matrix = (
            post_select_multiplier * cur_matrix
        )  # once reach here the current matrix is one of the ones selected
        modules.append(cur_matrix)
    return modules


# function to put together the generated component RNNs into one big block diagonal weight matrix
def combine_W(W_list):
    shapes = [w.shape[0] for w in W_list]
    total_size = np.sum(shapes)
    full_W = np.zeros((total_size, total_size))
    for i in range(len(W_list)):
        cur_W = W_list[i]
        cur_size = shapes[i]
        first_index = int(np.sum(shapes[0:i]))
        last_index = first_index + cur_size
        full_W[first_index:last_index, first_index:last_index] = cur_W
    return full_W


# use theorem 1 + info about linear stability to find a metric for a given weight matrix (expected to be generated from the above)
def find_M(W_inp):
    # what we actually want to find metric for is W - I -> just W won't be stable here
    # also first need to focus on abs(W), not W itself! that is what linear stable test can find, the same metric will then work for the other (per Thm 1)
    W = np.abs(W_inp)  # diagonal is set to 0 already so no need to worry about that
    W = W - np.identity(W.shape[0])
    # this just finds some M that will work, could be many others
    Q = np.identity(W.shape[0])
    # solve for M in -Q = M * W + np.transpose(W) * M
    # using integration formula for LTI system
    P = np.zeros(W.shape)
    for i in range(W.shape[0]):
        # integrating elementwise
        # keep off-diags as 0 to save time with larger martrix, as know there will be some diagonal metric, expect good odds that will find one with Q = I
        # will confirm the metric works before moving forward though (done in final function below), to be sure with stability guarantee
        def func_to_integrate(t):
            og_func = np.exp(np.transpose(W) * t) * Q * np.exp(W * t)
            return og_func[i, i]

        P[i, i] = scipy.integrate.quad(func_to_integrate, 0, np.inf)[0]
    if np.max(np.linalg.eigvals(P)) <= 0:
        # guaranteed M will be symmetric as it is definitely diagonal here, but also need it be PD for it to be a valid metric
        # should never reach this in theory, but add as a check to be safe
        print("returned metric not PD, problem!")
        return None
    return P


# replaced version of this function with more efficient one when I started using this notebook for the final version of the CIFAR10 experiment repetitions
# put everything together to get W and M to perform the training with
# W is of course part of the network
# M is used in finding negative feedback connections between components of W that maintain stability
# neither are themselves updated over the course of training
def generate_initial_W_M(
    module_sizes, density, dist_multiplier, post_select_multiplier
):
    individual_networks = create_modules(
        module_sizes, density, dist_multiplier, post_select_multiplier
    )
    full_W = combine_W(individual_networks)

    # in prior experiments 0.5 * I was metric generally found
    # so just use that if it will work - only integrate if necessary
    matching_M = 0.5 * np.identity(full_W.shape[0])
    check_formula = (
        matching_M * (np.abs(full_W) - np.identity(full_W.shape[0]))
        + np.transpose(np.abs(full_W) - np.identity(full_W.shape[0])) * matching_M
    )
    if np.max(np.linalg.eigvals(check_formula)) >= 0:
        matching_M = find_M(full_W)

        # confirm that M does work to satisfy the Theorem 1 condition with this W
        check_formula = (
            matching_M * (np.abs(full_W) - np.identity(full_W.shape[0]))
            + np.transpose(np.abs(full_W) - np.identity(full_W.shape[0])) * matching_M
        )
        if np.max(np.linalg.eigvals(check_formula)) >= 0:
            print("problem with found metric!")
            return None

    # return the final W and M
    # need these to be tensors for computation to work - but they aren't parameters!
    return (
        torch.from_numpy(full_W).float().cuda(),
        torch.from_numpy(matching_M).float().cuda(),
    )


# helper functions for training


def add_channels(X):
    # reshaping necessary when loading the training data
    if len(X.shape) == 2:
        return X.reshape(X.shape[0], 1, X.shape[1], 1)
    elif len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    else:
        return "dimenional error"


def exp_lr_scheduler(epoch, optimizer, strategy="normal", decay_eff=0.1, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""

    if strategy == "normal":
        if epoch in decayEpoch:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= decay_eff
            print("New learning rate is: ", param_group["lr"])
    else:
        print("wrong strategy")
        raise ValueError("A very specific bad thing happened.")

    return optimizer


# get adjacency matrix specifying which modules should be in negative feedback with each other
# for full negative feedback frac_zeros would be 0, but will play with sparsity settings
# (this is part of initialization)
def create_random_A(ns, frac_zeros):
    num_networks = len(ns)
    A = torch.cuda.FloatTensor(num_networks, num_networks).uniform_() > frac_zeros
    A_tril = torch.tril(A)
    A_tril.fill_diagonal_(0)
    # note only lower triangular needs to be trained, as bidirectional version of connection determined by negative feedback stability cond
    return A_tril


def create_mask_given_A(A, ns):
    """
    Creates 'hidden' mask for training, given an arbitrary adjacency matrix.

    ARGS:
        - A: adjacency matrix
        - ns: list of neural population sizes (e.g ns = [5,4,18,2]).
    OUTS:
        - A: mask
    """

    N_nets = len(ns)
    mask = []

    for i in range(N_nets):
        mask_row_i = torch.cat(
            [
                torch.ones((ns[i], ns[j]))
                if A[i, j] == 1 and i >= j
                else torch.zeros((ns[i], ns[j]))
                for j in range(N_nets)
            ],
            1,
        )
        mask.append(mask_row_i)

    final = torch.cat(mask, 0)
    # final.to(device)
    return final


def create_random_block_diagonal_metric(ns):
    # for testing
    Ms = []

    for i in range(len(ns)):
        B = F.dropout(torch.randn((ns[i], ns[i])), 0.9)
        Ms.append(B.T @ B)

    return torch.block_diag(*Ms)


def create_random_block_stable_symmetric_weights(ns):
    # for testing
    Ws = []

    for i in range(len(ns)):
        B = F.dropout((1 / ns[i]) * torch.randn((ns[i], ns[i])), 0.9)
        Ws.append((1 - 1e-3) * torch.eye(ns[i]) - B.T @ B)
        # Ws.append((1 - 1e-3) * torch.eye(ns[i]))

    return torch.block_diag(*Ws), Ws


def get_M_given_sym_W(W, eps=1e-3):
    gamma_sqr = 1 / (4 * (1 - 1e-5))
    gamma = np.sqrt(gamma_sqr)

    eig_W, Q = torch.linalg.eigh(W)

    I = torch.eye(W.shape[0])

    T = 1 / (4 * gamma_sqr) * I - torch.diag_embed(eig_W)

    S = Q @ torch.sqrt(T) @ Q.T
    R = (1 / gamma) * S + (1 / (2 * gamma_sqr)) * I
    M = gamma_sqr * R @ R

    # reconstruction error
    e = torch.norm(R - M - W, "fro")
    if e >= 1e-2:
        print(f"Error in getting M from a given symmetric W, error is {e}")

    return M


def get_M_hat_given_W_hat(Ws):
    # for testing
    Ms = []

    for i in range(len(Ws)):
        M = get_M_given_sym_W(Ws[i], eps=1e-5)
        Ms.append(M)

    return torch.block_diag(*Ms)


def log_abs(tensor, eps=1e-3):
    return torch.log(torch.abs(tensor) + 1e-3)


def get_directory_names(path_string):
    """
    This function takes a string containing a path of the form 'dir1/dir2/etc'
    and returns a list of the directory names in the form ['dir1', 'dir2', 'etc'].
    """
    directories = path_string.split("/")
    return [directory for directory in directories if directory != ""]


def get_constraint_type_and_task(PATH):
    dir_list = get_directory_names(PATH)
    return dir_list[-3], dir_list[-2]


def get_model_info(PATH):
    param_dict = torch.load(PATH)

    hidden_size = param_dict["rnn.h2h.bias"].shape[0]
    input_size = param_dict["rnn.input2h.weight"].shape[1]
    output_size = param_dict["fc.weight"].shape[0]

    c_type, task = get_constraint_type_and_task(PATH)

    return input_size, hidden_size, output_size, c_type, task


def load_model_from_path(PATH):
    input_size, hidden_size, output_size, c_type, task = get_model_info(PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = models.RNNNet(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        device=device,
        dt=30,
        constraint=c_type,
    ).to(device)

    net.load_state_dict(torch.load(PATH))

    return net


def get_path_from_task_and_constraint(task_and_constraint):
    # task_and_constraint: tuple of (task,constraint)

    base_path = "/om2/user/leokoz8/code/rnns-of-rnns/models"

    return (
        base_path
        + f"/{task_and_constraint[1]}"
        + f"/{task_and_constraint[0]}"
        + "/small_model.pickle"
    )


def get_stacked_weights_and_biases(tasks_and_constraints):
    input2h_weight_bar = []
    rnn_input2h_bias_bar = []
    rnn_h2h_weight_bar = []
    rnn_h2h_bias_bar = []
    fc_weight_bar = []
    fc_bias_bar = []

    for el in tasks_and_constraints:
        path = get_path_from_task_and_constraint(el)

        param_dict = torch.load(path)

        net = load_model_from_path(path)

        # print(param_dict)

        input2h_weight_bar.append(param_dict["rnn.input2h.weight"])
        rnn_input2h_bias_bar.append(param_dict["rnn.input2h.bias"])
        rnn_h2h_weight_bar.append(net.rnn.h2h.weight)
        rnn_h2h_bias_bar.append(param_dict["rnn.h2h.bias"])
        fc_weight_bar.append(param_dict["fc.weight"])
        fc_bias_bar.append(param_dict["fc.bias"])

    return (
        input2h_weight_bar,
        rnn_input2h_bias_bar,
        rnn_h2h_weight_bar,
        rnn_h2h_bias_bar,
        fc_weight_bar,
        fc_bias_bar,
    )


def build_GWNET_from_pretrained(
    tasks_and_constraints,
    env,
    device,
    pretrained_hidden_size=32,
    gw_hidden_size=16,
    interareal_constraint="conformal",
):
    p = len(tasks_and_constraints) + 1
    c_type = tasks_and_constraints[0][1]

    # import pretrained weights and biases

    (
        input2h_weight_bar,
        rnn_input2h_bias,
        rnn_h2h_weight,
        rnn_h2h_bias,
        fc_weight_bar,
        fc_bias_bar,
    ) = get_stacked_weights_and_biases(tasks_and_constraints)

    stacked_wb = {
        "input2h_weight_bar": input2h_weight_bar,
        "rnn_input2h_bias": rnn_input2h_bias,
        "rnn_h2h_weight": rnn_h2h_weight,
        "rnn_h2h_bias": rnn_h2h_bias,
        "fc_weight_bar": fc_weight_bar,
        "fc_bias_bar": fc_bias_bar,
    }

    # Build GW Net

    ns = [pretrained_hidden_size for _ in range(p - 1)]
    ns.append(gw_hidden_size)

    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    A_tril = torch.zeros((len(ns), len(ns)))
    A_tril[-1, :] = 1

    B_mask = create_mask_given_A(A_tril, ns)

    if interareal_constraint == "None":
        # No stability constraint on interareal weights
        B_mask = 0.5 * (B_mask + B_mask.T)
        M_hat = torch.eye(sum(ns), device=device)

    if interareal_constraint == "conformal":
        # Stability constraint on interareal weights, conformal to the stability constraint on the subnetworks

        if c_type == "spectral":
            M_hat = torch.eye(sum(ns), device=device)

        if c_type == "sym" or c_type == "None":
            with torch.no_grad():
                Ms = compute_metric_from_weights(
                    stacked_wb["rnn_h2h_weight"], ctype=c_type, device=device
                )
                M_hat = torch.block_diag(*Ms)

    B_mask = F.dropout(B_mask, 0.5)

    # Create global workspace weights and biases here
    W_hat, Ws = create_random_block_stable_symmetric_weights([gw_hidden_size])
    W_hat = W_hat.to(device)
    tmp = nn.Linear(gw_hidden_size, gw_hidden_size, bias=True, device=device)
    b_hat = tmp.bias

    stacked_wb["rnn_h2h_weight"].append(W_hat)
    stacked_wb["rnn_h2h_bias"].append(b_hat)

    B_mask = B_mask.to(device)

    net = models.GW_RNNNet(
        stacked_wb,
        input_size,
        ns,
        output_size,
        M_hat,
        B_mask,
        device,
        interareal_constraint,
        dt=30,
    )
    net.to(device)

    return net


def centorrino_theta(b, Lambda, device):
    # Method introduced in Centorrino, 2023

    scalar_theta = lambda z: 2 * b * (1 + torch.sqrt(1 - z / b))

    n = Lambda.shape[0]
    theta = torch.zeros(n)

    for i in range(n):
        theta[i] = scalar_theta(Lambda[i])

    return theta.to(device)


def compute_metric_from_sym_weight_matrix(W, device):
    # Uses the metric introduced in Centorrino, 2023

    L, U = torch.linalg.eigh(W)

    alpha_W = torch.max(L)

    with torch.no_grad():
        theta = centorrino_theta(alpha_W, L, device)

    return U @ torch.diag(theta**2) @ U.T


def compute_metric_from_weights(Ws, ctype, device):
    Ms = []

    for W in Ws:
        if ctype == "sym":
            M = compute_metric_from_sym_weight_matrix(W, device)
        if ctype == "None":
            M = compute_linear_metric_from_weight_matrix(W, device)

        Ms.append(M)

    return Ms


from scipy import linalg


def compute_linear_metric_from_weight_matrix(W, device):
    # Solves the Lyapunov equation to obtain metric

    W = W.detach().cpu().numpy()

    I = np.eye(W.shape[0])

    J = -I + W

    M = linalg.solve_continuous_lyapunov(J.T, -I)

    return torch.from_numpy(M).to(device).float()
