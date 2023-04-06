import os, argparse
from src import models, utils,tasks, running
import numpy as np
import pandas
import tqdm
import fire

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

# Load in all of the neurogym tasks
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
from neurogym import Dataset
from Mod_Cog.mod_cog_tasks import *
import pickle



np.random.seed(42)
torch.manual_seed(42)


parser = argparse.ArgumentParser(description="Global Workspace From Pretrained")

parser.add_argument("--constraint", choices = ['spectral','sym', 'None'], default=None, help=" type of rnn stability constraint to use")
parser.add_argument("--interareal_constraint", choices = ['None','conformal'], default=None, help=" type of rnn stability constraint to use")

parser.add_argument(
    "--num_gradient_steps",
    default=5000,
    type=int,
    help="number of total gradient steps to take",
)
parser.add_argument(
    "--GW_hidden_size", default=16, type=int, help="number of hidden units in the RNN"
)
parser.add_argument(
    "--eval_every",
    default=25,
    type=int,
    help="how often to evaulate model performance",
)
parser.add_argument("--task_ID", default=0, type=int, help="which task to perform from the extended Yang task set")
parser.add_argument("--dt", default=30, type=int, help="rnn step size")
parser.add_argument("--batch_size", default=16, type=int, help="mini-batch size")
parser.add_argument(
     "--learning_rate", default=1e-2, type=float, help="initial learning rate"
)
parser.add_argument(
    "--step_size",
    default=10,
    type=int,
    help="after how many epochs learning rate should be decreased 10x",
)

parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay ")
FLAGS, FIRE_FLAGS = parser.parse_known_args()

envs, env_names = tasks.load_all_mod_cog_tasks()


if __name__ == "__main__":

    # define task here
    task_index = FLAGS.task_ID

    dataset = Dataset(envs[task_index], batch_size=64, seq_len=100)
    env = dataset.env
    device = "cuda" if torch.cuda.is_available() else "cpu"


    tasks_and_constraints  = [(env_names[i],'spectral') for i in range(0,19)]
    
    net = utils.build_GWNET_from_pretrained(tasks_and_constraints, env, device, gw_hidden_size = FLAGS.GW_hidden_size, interareal_constraint = FLAGS.interareal_constraint)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'max', factor = 0.1)


    num_gradient_steps = FLAGS.num_gradient_steps
    eval_every = FLAGS.eval_every

    training_params = {
        "num_gradient_steps": num_gradient_steps,
        "eval_every": eval_every,
    }

    net_trained, perf_over_training = running.train(net,dataset,criterion,optimizer, scheduler, training_params, device)

 
    # specify folder name
    folder_name = '/om2/user/leokoz8/code/rnns-of-rnns/models/' + FLAGS.constraint + '/' + env_names[task_index] + '/' + FLAGS.interareal_constraint

    # create folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully!")
    else:
        print(f"Folder '{folder_name}' already exists!")

    torch.save(net_trained.state_dict(), folder_name + '/gw_model.pickle')
    torch.save(perf_over_training, folder_name + '/perf_over_training.pickle')
   





