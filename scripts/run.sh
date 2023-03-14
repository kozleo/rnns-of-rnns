#!/bin/bash
#SBATCH --job-name=jup_nb
#SBATCH --mem=32GB
#SBATCH --output=/om2/user/leokoz8/code/rnns-of-rnns/results/slurm_out/%A_%a.out

#SBATCH --qos=normal
#SBATCH --partition=normal

##SBATCH --qos=fietelab
##SBATCH --partition=fietelab

#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --array=1-82%10

cd /om2/user/leokoz8/code/rnns-of-rnns
unset XDG_RUNTIME_DIR
python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint None --hidden_size 128
python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint sym  --hidden_size 128
python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint spectral  --hidden_size 128