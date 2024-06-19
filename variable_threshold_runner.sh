#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=md1823
#SBATCH --output=out_variable_threshold_%j.out


source /vol/bitbucket/md1823/taskmaster/learning/venv/bin/activate

export PYTHONUNBUFFERED=TRUE

python3 -m al.variable_threshold_solver
