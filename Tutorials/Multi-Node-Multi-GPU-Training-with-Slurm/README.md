# Multi-Node Multi-GPU Training with Slurm

* [Demo Structure](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/Multi-Node-Multi-GPU-Training-with-Slurm#demo-structure)
* [Requirements](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/Multi-Node-Multi-GPU-Training-with-Slurm#requirements)
* [Usage](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/Multi-Node-Multi-GPU-Training-with-Slurm#usage)



## Demo Structure

```bash
.
├── README.md
├── mnist_ddp_slurm.py
├── mnist_ddp_slurm_pytorch_venv.sbatch
├── mnist_ddp_slurm_pytorch_sif.sbatch
└── requirements.txt
```



## Requirements

```
torch==1.8.1
torchvision==0.9.1
```



> `pip install -r requirements.txt` can handle all package dependencies.



## Usage

#### Interactive Mode

Start an interactive session on a node using `salloc`, then submit a job directly using `srun`:

```bash
srun python mnist_ddp_slurm.py --batch_size=1024
```



#### Batch Mode

Submit a job with a Bash script using `sbatch`:

```bash
sbatch mnist_ddp_slurm_pytorch_venv.sbatch
```

or

```bash
sbatch mnist_ddp_slurm_pytorch_sif.sbatch
```
