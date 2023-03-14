import os, argparse
from src import models, utils
import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn

np.random.seed(42)
torch.manual_seed(42)


parser = argparse.ArgumentParser(description="Yang Task Set RNN Training")

"""
parser.add_argument(
    "--data_path",
    required=True,
    help="path to ImageNet folder that contains train and val folders",
)
"""
# parser.add_argument("-o", "--output_path", default=None, help="path for storing ")

parser.add_argument("--constraint", choices = ['spectral','sym', 'None'], default=None, help=" type of rnn stability constraint to use")

parser.add_argument(
    "--num_gradient_steps",
    default=500,
    type=int,
    help="number of total gradient steps to take",
)
parser.add_argument(
    "--hidden_size", default=256, type=int, help="number of hidden units in the RNN"
)
parser.add_argument(
    "--eval_every",
    default=100,
    type=int,
    help="how often to evaulate model performance",
)
parser.add_argument("--task_ID", default=0, type=int, help="which task to perform from the extended Yang task set")
parser.add_argument("--dt", default=30, type=int, help="rnn step size")
parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size")
parser.add_argument(
     "--learning_rate", default=1e-3, type=float, help="initial learning rate"
)
parser.add_argument(
    "--step_size",
    default=10,
    type=int,
    help="after how many epochs learning rate should be decreased 10x",
)

parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay ")
FLAGS, FIRE_FLAGS = parser.parse_known_args()


# Load in all of the neurogym tasks
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
from neurogym import Dataset
from Mod_Cog.mod_cog_tasks import *

envs = [
    go(),
    rtgo(),
    dlygo(),
    anti(),
    rtanti(),
    dlyanti(),
    dm1(),
    dm2(),
    ctxdm1(),
    ctxdm2(),
    multidm(),
    dlydm1(),
    dlydm2(),
    ctxdlydm1(),
    ctxdlydm2(),
    multidlydm(),
    dms(),
    dnms(),
    dmc(),
    dnmc(),
    dlygointr(),
    dlygointl(),
    dlyantiintr(),
    dlyantiintl(),
    dlydm1intr(),
    dlydm1intl(),
    dlydm2intr(),
    dlydm2intl(),
    ctxdlydm1intr(),
    ctxdlydm1intl(),
    ctxdlydm2intr(),
    ctxdlydm2intl(),
    multidlydmintr(),
    multidlydmintl(),
    dmsintr(),
    dmsintl(),
    dnmsintr(),
    dnmsintl(),
    dmcintr(),
    dmcintl(),
    dnmcintr(),
    dnmcintl(),
    goseqr(),
    rtgoseqr(),
    dlygoseqr(),
    antiseqr(),
    rtantiseqr(),
    dlyantiseqr(),
    dm1seqr(),
    dm2seqr(),
    ctxdm1seqr(),
    ctxdm2seqr(),
    multidmseqr(),
    dlydm1seqr(),
    dlydm2seqr(),
    ctxdlydm1seqr(),
    ctxdlydm2seqr(),
    multidlydmseqr(),
    dmsseqr(),
    dnmsseqr(),
    dmcseqr(),
    dnmcseqr(),
    goseql(),
    rtgoseql(),
    dlygoseql(),
    antiseql(),
    rtantiseql(),
    dlyantiseql(),
    dm1seql(),
    dm2seql(),
    ctxdm1seql(),
    ctxdm2seql(),
    multidmseql(),
    dlydm1seql(),
    dlydm2seql(),
    ctxdlydm1seql(),
    ctxdlydm2seql(),
    multidlydmseql(),
    dmsseql(),
    dnmsseql(),
    dmcseql(),
    dnmcseql(),
]

env_names = [
    "go",
    "rtgo",
    "dlygo",
    "anti",
    "rtanti",
    "dlyanti",
    "dm1",
    "dm2",
    "ctxdm1",
    "ctxdm2",
    "multidm",
    "dlydm1",
    "dlydm2",
    "ctxdlydm1",
    "ctxdlydm2",
    "multidlydm",
    "dms",
    "dnms",
    "dmc",
    "dnmc",
    "dlygointr",
    "dlygointl",
    "dlyantiintr",
    "dlyantiintl",
    "dlydm1intr",
    "dlydm1intl",
    "dlydm2intr",
    "dlydm2intl",
    "ctxdlydm1intr",
    "ctxdlydm1intl",
    "ctxdlydm2intr",
    "ctxdlydm2intl",
    "multidlydmintr",
    "multidlydmintl",
    "dmsintr",
    "dmsintl",
    "dnmsintr",
    "dnmsintl",
    "dmcintr",
    "dmcintl",
    "dnmcintr",
    "dnmcintl",
    "goseqr",
    "rtgoseqr",
    "dlygoseqr",
    "antiseqr",
    "rtantiseqr",
    "dlyantiseqr",
    "dm1seqr",
    "dm2seqr",
    "ctxdm1seqr",
    "ctxdm2seqr",
    "multidmseqr",
    "dlydm1seqr",
    "dlydm2seqr",
    "ctxdlydm1seqr",
    "ctxdlydm2seqr",
    "multidlydmseqr",
    "dmsseqr",
    "dnmsseqr",
    "dmcseqr",
    "dnmcseqr",
    "goseql",
    "rtgoseql",
    "dlygoseql",
    "antiseql",
    "rtantiseql",
    "dlyantiseql",
    "dm1seql",
    "dm2seql",
    "ctxdm1seql",
    "ctxdm2seql",
    "multidmseql",
    "dlydm1seql",
    "dlydm2seql",
    "ctxdlydm1seql",
    "ctxdlydm2seql",
    "multidlydmseql",
    "dmsseql",
    "dnmsseql",
    "dmcseql",
    "dnmcseql",
]





# main training loop
def train(net, criterion, optimizer, training_params):
    running_loss = 0.0

    num_gradient_steps = training_params["num_gradient_steps"]
    eval_every = training_params["eval_every"]

    pbar = tqdm.trange(num_gradient_steps)
    count_eval = 0
    perf_over_training = torch.zeros(int(num_gradient_steps / eval_every))

    for i in pbar:
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
        if i % eval_every == eval_every - 1:
            running_loss = 0.0

            with torch.no_grad():
                perf = utils.get_performance(net, env, device)
                perf_over_training[count_eval] = perf
                pbar.set_description(f"Performance is {perf} at gradient step {i}")
            count_eval += 1

            # break if performance above 90
            if perf >= 0.90:
                break

    pbar.set_description(f"Performance is {perf} at gradient step {i}")

    return net, perf_over_training


if __name__ == "__main__":
    # fire.Fire(command=FIRE_FLAGS)

    # define task here
    task_index = FLAGS.task_ID

    dataset = Dataset(envs[task_index], batch_size=32, seq_len=100)
    env = dataset.env
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.n    

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = models.RNNNet(
        input_size=ob_size,
        hidden_size=FLAGS.hidden_size,
        output_size=act_size,
        device=device,
        dt=FLAGS.dt,
        constraint = FLAGS.constraint
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)

    num_gradient_steps = FLAGS.num_gradient_steps
    eval_every = FLAGS.eval_every

    training_params = {
        "num_gradient_steps": num_gradient_steps,
        "eval_every": eval_every,
    }

    net_trained, perf_over_training = train(net, criterion, optimizer, training_params)

 
    # specify folder name
    folder_name = '/om2/user/leokoz8/code/rnns-of-rnns/models/' + FLAGS.constraint + '/' + env_names[task_index]

    # create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

    torch.save(net_trained.state_dict(), folder_name + '/model.pickle')







