#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=md1823
#SBATCH --output=out_sa_hyper_%j.out


source /vol/bitbucket/md1823/taskmaster/learning/venv/bin/activate

export PYTHONUNBUFFERED=TRUE

python3 -m speed_test