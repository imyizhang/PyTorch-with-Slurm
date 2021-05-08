# Data Parallel Training



## Training Examples on MNIST

- [x] **Use single-node single-GPU training**
  * **[[mnist.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/Tutorials/Data-Parallel-Training/mnist.py)]**
- [x] **Use single-node multi-GPU [DataParallel](https://pytorch.org/docs/master/generated/torch.nn.DataParallel.html) (`dp`) training**
  * **[[mnist_dp.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/Tutorials/Data-Parallel-Training/mnist_dp.py)]**
- [x] **Use multi-node (or single-node) multi-GPU [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html) (`ddp`) training**
  * **with [`torch.distributed.launch`](https://pytorch.org/docs/stable/distributed.html#launch-utility) module [[mnist_ddp_launch.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/Tutorials/Data-Parallel-Training/mnist_ddp_launch.py)]**
  * **with [`torch.multiprocessing.spawn`](https://pytorch.org/docs/stable/multiprocessing.html#spawning-subprocesses) module [[mnist_ddp_mp.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/Tutorials/Data-Parallel-Training/mnist_ddp_mp.py)]**
  * **with [Slurm](https://slurm.schedmd.com/quickstart.html) cluster workload manager [[mnist_ddp_slurm.py](https://github.com/yzhang-dev/PyTorch-with-Slurm/Tutorials/Data-Parallel-Training/mnist_ddp_slurm.py)]**



## Reference

* **[PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)**
* **[`torch.distributed.launch` Script](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)**

