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

echo "INFO - started loading cuDNN and CUDA"
module load cuDNN/8.0.4.30-CUDA-11.1.1
# if you need to know the allocated CUDA device, you can obtain it here:
echo "DEBUG - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
srun nvidia-smi

echo "INFO - started loading Singularity and build PyTorch SIF"
module load GCC/9.3.0 Singularity/3.7.3-GCC-9.3.0-Go-1.14
# https://hub.docker.com/r/pytorch/pytorch/tags?page=1&ordering=last_updated
PYTORCH_VERSION="1.8.1-cuda11.1-cudnn8-runtime"
if [ ! -r ${PWD}/pytorch_${PYTORCH_VERSION}.sif ]; then
    srun --unbuffered singularity \
        build \
        ${PWD}/pytorch_${PYTORCH_VERSION}.sif \
        docker://pytorch/pytorch:${PYTORCH_VERSION}
    if [ $? -ne 0 ]; then
        echo "ERROR - failed to build ${PWD}/pytorch_${PYTORCH_VERSION}.sif"
        exit 1
    fi
    echo "DEBUG - finished building ${PWD}/pytorch_${PYTORCH_VERSION}.sif"
fi

echo "INFO - started running Python script"
srun --unbuffered singularity \
    exec \
    --nv \
    ${PWD}/pytorch_${PYTORCH_VERSION}.sif \
        python \
        ${PWD}/torch.cuda.is_available.py
