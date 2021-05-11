#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# <https://doc.eresearch.unige.ch/hpc/slurm#gpgpu_jobs>

#SBATCH --partition=shared-gpu
#SBATCH --time=00:30:00
#SBATCH --gpus=1
#SBATCH --output=slurm-%x-%j.out

echo "DEBUG - full hostname: $(hostname -f)"
echo "DEBUG - work directory: ${PWD}"

echo "INFO - started loading Python, CUDA and cuDNN"
module load GCCcore/9.3.0 Python/3.8.2
module load cuDNN/8.0.4.30-CUDA-11.1.1
# if you need to know the allocated CUDA device, you can obtain it here:
echo "DEBUG - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
srun nvidia-smi

echo "INFO - started activating virtual environment"
#python -m venv $HOME/workspace/pytorch-env
source ${HOME}/workspace/pytorch-env/bin/activate
#python -m pip install --upgrade pip
#python -m pip install -r requirements.txt

echo "INFO - started running Python script"
srun --unbuffered python \
    ${PWD}/torch.cuda.is_available.py

deactivate
