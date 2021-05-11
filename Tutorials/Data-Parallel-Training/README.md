# Data Parallel Training

* [Dependencies]()
* [Training Examples on MNIST]()
* [Reference]()



## Dependencies

```
torch==1.8.1
torchvision==0.9.1
```



> `pip install -r requirements.txt` can handle all package dependencies.



## Training Examples on MNIST

- [x] Use single-node single-GPU training [[mnist.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/blob/main/Tutorials/Data-Parallel-Training/mnist.py)]

  ```bash
  python mnist.py --batch_size=256
  ```

- [x] Use single-node multi-GPU [`DataParallel`](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html) (DP) training [[mnist_dp.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/blob/main/Tutorials/Data-Parallel-Training/mnist_dp.py)]

  ```bash
  python mnist_dp.py --batch_size=1024
  ```

- [x] **Use multi-node (or single-node) multi-GPU [`DistributedDataParallel`](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html) (DDP) training**
  
  - [x] with [`torch.distributed.launch`](https://pytorch.org/docs/stable/distributed.html#launch-utility) module [[mnist_ddp_launch.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/blob/main/Tutorials/Data-Parallel-Training/mnist_ddp_launch.py)]
  
    ```bash
    python -m torch.distributed.launch --use_env --nproc_per_node=4 mnist_ddp_launch.py --batch_size=1024
    ```
  
  - [x] with [`torch.multiprocessing.spawn`](https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses) module [[mnist_ddp_mp.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/blob/main/Tutorials/Data-Parallel-Training/mnist_ddp_mp.py)]
  
    ```bash
    python mnist_ddp_mp.py --batch_size=1024
    ```
  
  - [x] with [Slurm](https://slurm.schedmd.com/quickstart.html) cluster workload manager [[mnist_ddp_slurm.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/blob/main/Tutorials/Data-Parallel-Training/mnist_ddp_slurm.py)]
  
    ```bash
    srun python mnist_ddp_slurm.py --batch_size=1024
    ```



## Reference

* [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
* [`torch.distributed.launch` Script](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)

