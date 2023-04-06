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
#SBATCH --array=20-82

cd /om2/user/leokoz8/code/rnns-of-rnns
unset XDG_RUNTIME_DIR
python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run_pretrained.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint None --interareal_constraint None


python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run_pretrained.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint sym --interareal_constraint conformal
python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run_pretrained.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint sym --interareal_constraint None

python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run_pretrained.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint spectral --interareal_constraint conformal
python /om2/user/leokoz8/code/rnns-of-rnns/scripts/run_pretrained.py --task_ID $SLURM_ARRAY_TASK_ID --num_gradient_steps 10000 --constraint spectral --interareal_constraint None