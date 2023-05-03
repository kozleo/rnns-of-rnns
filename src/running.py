import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import tqdm
from src import models, utils, parametrizations, tasks


def train(net, dataset, criterion, optimizer, scheduler, training_params, device):
    env = dataset.env

    running_loss = 0.0

    num_gradient_steps = training_params["num_gradient_steps"]
    eval_every = training_params["eval_every"]

    act_size = env.action_space.n

    count_eval = 0
    perf_over_training = torch.zeros(int(num_gradient_steps / eval_every) + 1)

    for i in range(num_gradient_steps):
        inputs, labels = dataset()
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)

        

        labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs, rnn_outputs = net(inputs)

        loss = criterion(outputs.reshape(-1, act_size), labels)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % eval_every == eval_every - 1 or i == 0:
            with torch.no_grad():
                perf = get_performance(net, env, device)
                perf_over_training[count_eval] = perf
                print(
                    f"Gradient step: {i} \t Performance: {perf} \t Training loss: {running_loss:.2f}"
                )
            count_eval += 1

            running_loss = 0.0

            # break if performance above 90
            if perf >= 0.90:
                break
            scheduler.step(perf)

    # pbar.set_description(f'Performance is {perf} at gradient step {i}')

    return net, perf_over_training


def get_performance(net, env, device,noise_level = 0):
    perf = 0
    num_trial = 200
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        ob = ob[:, np.newaxis, :]  # Add batch axis
        inputs = torch.from_numpy(ob).type(torch.float).to(device)
        inputs += noise_level*torch.randn(inputs.shape,device = device)

        action_pred, _ = net(inputs)
        action_pred = action_pred.cpu().detach().numpy()
        action_pred = np.argmax(action_pred, axis=-1)
        perf += gt[-1] == action_pred[-1, 0]

    perf /= num_trial
    # print('Average performance in {:d} trials'.format(num_trial))
    return perf
