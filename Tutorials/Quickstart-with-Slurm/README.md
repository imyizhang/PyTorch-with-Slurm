# Quickstart with Slurm

* [Demo Structure](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/Quickstart-with-Slurm#demo-structure)
* [Dependencies](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/Quickstart-with-Slurm#dependencies)
* [Usage](https://github.com/yzhang-dev/PyTorch-with-Slurm/tree/main/Tutorials/Quickstart-with-Slurm#usage)



## Demo Structure

```bash
.
├── README.md
├── torch.cuda.is_available.py
├── torch.cuda.is_available_pytorch_venv.sbatch
└── torch.cuda.is_available_pytorch_sif.sbatch
```



## Dependencies

```
torch==1.8.1
```



> `pip install -r requirements.txt` can handle all package dependencies.



## Usage

Submit a job with a Bash script using `sbatch`:

```bash
sbatch torch.cuda.is_available_pytorch_venv.sbatch
```

or

```bash
sbatch torch.cuda.is_available_pytorch_sif.sbatch
```
