#!/bin/bash
#SBATCH --job-name=jup_nb
#SBATCH --mem=16GB
#SBATCH --output=/om2/user/leokoz8/code/rnns-of-rnns/results/slurm_out/%j.out

#SBATCH --qos=normal
#SBATCH --partition=normal

##SBATCH --qos=fiete
##SBATCH --partition=fiete

#SBATCH --gres=gpu:1
#SBATCH --time=0-5:00:00

#cd /om2/user/leokoz8/code/rnns-of-rnns
unset XDG_RUNTIME_DIR
#conda activate rnns-of-rnns-env
jupyter notebook --ip=0.0.0.0 --port=9000 --no-browser --NotebookApp.token='' --NotebookApp.password=''