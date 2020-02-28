#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1
#SBATCH -t 01:00:00

# Load a anaconda module
module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh

#activate environment
conda activate tf-gpu

export PYTHONUNBUFFERED=TRUE
# Run script
python main_process cnnmodel_example