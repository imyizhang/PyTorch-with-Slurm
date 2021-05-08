#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Multi-Node Multi-GPU Training
# (with torch.nn.parallel.DistributedDataParallel and torch.distributed.launch)
#
# <https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py>

"""How to use `torch.distributed.launch`
0. To look up optional arguments:
::
    >>> python -m torch.distributed.launch --help
1. Single-Node Multi-GPU Training:
::
    >>> python -m torch.distributed.launch \
        --use_env \
        --nproc_per_node=GPUS_PER_NODE \
        YOUR_TRAINING_SCRIPT.py --arg1 --arg2 ... (arguments of your script)
2. Multi-Node Multi-GPU Training (e.g., two nodes):
Node 1:
::
    >>> python -m torch.distributed.launch \
        --use_env \
        --nproc_per_node=GPUS_PER_NODE \
        --nnodes=2 \
        --node_rank=0 \
        --master_addr='127.0.0.1' \
        --master_port=29500 \
        YOUR_TRAINING_SCRIPT.py --arg1 --arg2 ... (arguments of your script)
Node 2:
::
    >>> python -m torch.distributed.launch \
        --use_env \
        --nproc_per_node=GPUS_PER_NODE \
        --nnodes=2 \
        --node_rank=1 \
        --master_addr='127.0.0.1' \
        --master_port=29500 \
        YOUR_TRAINING_SCRIPT.py --arg1 --arg2 ... (arguments of your script)
"""

import argparse
import os
import time

import torch
import torchvision


def main(args):

    # 0. set up distributed device
    gpus_per_node = torch.cuda.device_count()
    print(
        "\n Let's play with MNIST using torch {} on {} GPU(s)!\n".format(
            torch.__version__, gpus_per_node
        )
    )
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    print(
        '             Local Rank: {} | Global Rank: {}\n'.format(
            local_rank, rank
        )
    )
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    # NCCL backend is the recommended backend to use for GPU training
    torch.distributed.init_process_group(backend='nccl')

    # 1. define dataloader
    trainset = torchvision.datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        shuffle=True,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=trainsampler,
    )

    # 2. define pretrained neural network
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    num_classes = 10
    model.fc = torch.nn.Linear(num_features, num_classes)
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    # 3. define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    since = time.time()

    print('            =======  Training Started ======= \n')

    # 4. start to finetune
    model.train()
    for epoch in range(1, args.max_epochs + 1):
        train_loss, train_acc = 0.0, 0.0
        size = 0
        trainloader.sampler.set_epoch(epoch)
        for batch_idx, batch in enumerate(trainloader):
            # training_step
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            correct = torch.eq(outputs.argmax(dim=1), targets).type(torch.float).sum()
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # training_step_end
            train_loss += loss.item()
            train_acc += correct.item()
            size += targets.size(0)
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(trainloader):
                print(
                    ' Step: [{:3}/{}] [{}/{}] | Loss: {:.3f} | Accuracy: {:.3f}%'.format(
                        batch_idx + 1,
                        len(trainloader),
                        epoch,
                        args.max_epochs,
                        train_loss / size,
                        100.0 * train_acc / size,
                    )
                )

    print('\n            =======  Training Finished  ======= \n')

    time_elapsed = time.time() - since
    print(
        '                Finished in {:.0f} min {:.0f} sec :D\n'.format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    torch.distributed.destroy_process_group()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        default=256,
        type=int,
        help='batch size'
    )
    parser.add_argument(
        '--max_epochs',
        default=5,
        type=int,
        help='max epoches for training'
    )
    args = parser.parse_args()

    main(args)
