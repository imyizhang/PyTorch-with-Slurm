# All-in-One Template Using PyTorch Bolt

* [Demo Structure](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/All-in-One-Template-Using-PyTorch-Bolt#demo-structure)
* [Requirements](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/All-in-One-Template-Using-PyTorch-Bolt#requirements)
* [Usage](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/All-in-One-Template-Using-PyTorch-Bolt#usage)
  * [Basic Usage](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/All-in-One-Template-Using-PyTorch-Bolt#basic-usage)
  * [Usage on Slurm Cluster](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/All-in-One-Template-Using-PyTorch-Bolt#usage-on-slurm-cluster)
* [Reference](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/All-in-One-Template-Using-PyTorch-Bolt#reference)



## Demo Structure

```bash
.
├── README.md
├── data
│   ├── __init__.py
│   └── mnist_datamodule.py
├── model
│   ├── __init__.py
│   └── mnist_classifier.py
├── mnist.py
├── mnist_pytorch_sif.sbatch
├── mnist_pytorch_venv.sbatch
└── requirements.txt
```



## Requirements

```
torch==1.8.1
torchvision==0.9.1
tensorboard==2.4.1
torch_bolt==0.0.1
```



> `pip install -r requirements.txt` can handle all package dependencies.



## Usage

### Basic Usage

Look up optional arguments:

```bash
python mnist.py --help
```

Single-node single-GPU training:

```bash
python mnist.py --batch_size=256 --num_workers=4 --pin_memory
```

Single-node multi-GPU training using `torch.distributed.launch`:

```bash
python -m torch.distributed.launch --use_env --nproc_per_node=4 mnist.py --batch_size=1024 --num_workers=4 --pin_memory --distributed
```



> Specify CUDA devices by setting environment variable `CUDA_VISIBLE_DEVICES`.



### Usage on Slurm Cluster

#### Interactive Mode

Start an interactive session on a node using `salloc`, then submit a job directly using `srun`:

```bash
srun python mnist.py --batch_size=1024 --num_workers=4 --pin_memory --distributed --use_slurm
```



#### Batch Mode

Submit a job with a Bash script using `sbatch`:

```bash
sbatch mnist_pytorch_venv.sbatch
```

or

```bash
sbatch mnist_pytorch_sif.sbatch
```



## Reference

* [PyTorch Bolt](https://github.com/yzhang-dev/PyTorch-Bolt)
