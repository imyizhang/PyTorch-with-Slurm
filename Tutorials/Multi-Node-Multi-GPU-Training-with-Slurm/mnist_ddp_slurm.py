#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Single/Multi-Node Multi-GPU Training
# (with torch.nn.parallel.DistributedDataParallel and Slurm)

import argparse
import os
import time

import torch
import torchvision


def setup_environ():
    hostname = os.environ['HOSTNAME']
    nodelist = os.environ['SLURM_NODELIST']
    world_size = int(os.environ['SLURM_NTASKS'])
    rank = int(os.environ['SLURM_PROCID'])
    nproc_per_node = torch.cuda.device_count()
    print(
        '\n       Master Node: {} | Node List: {}\n'.format(
            hostname, nodelist
        )
    )
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank % nproc_per_node)
    os.environ['MASTER_ADDR'] = hostname #'127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

def main(args):

    # 0. set up distributed device
    setup_environ()
    gpus_per_node = torch.cuda.device_count()
    print(
        " Let's play with MNIST using torch {} on {} GPU(s)!\n".format(
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
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
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
