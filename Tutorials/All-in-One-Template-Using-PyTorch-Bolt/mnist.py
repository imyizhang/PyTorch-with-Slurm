#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""How to run `mnist.py`
0. To look up optional arguments:
::
    >>> python mnist.py --help
1. Single-Node Single-GPU Training:
::
    >>> python mnist.py --batch_size=256 --num_workers=4 --pin_memory
2. Single-Node Multi-GPU Training:
::
    >>> python -m torch.distributed.launch --use_env --nproc_per_node=4 \
    main.py --batch_size=1024 --num_workers=4 --pin_memory --distributed
3. Multi-Node Multi-GPU Training on Slurm Cluster:
::
    >>> srun python main.py --batch_size=1024 --num_workers=4 \
    --pin_memory --distributed --use_slurm
"""

import argparse
import os

import pytorch_bolt
from pytorch_bolt.Trainer import get_rank

from data import MNISTDataModule
from model import MNISTClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Classification')
    parser = MNISTDataModule.add_argparse_args(parser)
    parser = MNISTClassifier.add_argparse_args(parser)
    parser = pytorch_bolt.Loggers.add_argparse_args(parser)
    parser = pytorch_bolt.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args


def main(args):
    # 0.
    # add required arguments for loggers
    args.tracker_keys = ['loss', 'score']
    # setup loggers
    loggers = pytorch_bolt.Loggers.from_argparse_args(args)
    # get root logger
    logger = loggers.configure_root_logger('mnist')
    logging_info(logger, 'logging started')

    # 1. setup datamodule, defined based on `pytorch_bolt.DataModule`
    mnist = MNISTDataModule.from_argparse_args(args)
    logging_info(logger, 'datamodule setup')

    # 2. setup classifier, defined based on `pytorch_bolt.Module`
    classifier = MNISTClassifier.from_argparse_args(args)
    logging_info(logger, 'classifier setup')

    # 3.
    # add required arguments for trainer
    args.datamodule = mnist
    args.model = classifier
    args.loggers = loggers
    # setup trainer
    trainer = pytorch_bolt.Trainer.from_argparse_args(args)
    logging_info(logger, 'trainer setup')

    # 4. fit classifier
    trainer.fit()
    logging_info(logger, 'fitting finished')

    # 5. test classifier
    trainer.test()
    logging_info(logger, 'test finished')

    # 6. destroy trainer
    trainer.destory()
    logging_info(logger, 'logging ended')


def logging_info(logger, message):
    if get_rank() == 0:
        logger.info(message)


if __name__ == '__main__':
    args = parse_args()
    main(args)
