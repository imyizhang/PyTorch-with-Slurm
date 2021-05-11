#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#<https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision>
#<https://pytorch.org/vision/stable/datasets.html>

import argparse

import torch
import torchvision
import pytorch_bolt


class MNISTDataModule(pytorch_bolt.DataModule):

    def __init__(self, args):
        super().__init__(args)

    def _setup_dataset(self):
        # for fit stage
        _trainset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self._transform()
        )
        _trainsize = len(_trainset)
        valsize = int(_trainsize / self.num_splits)
        trainsize = _trainsize - valsize
        trainset, valset = torch.utils.data.random_split(
            _trainset, lengths=[trainsize, valsize]
        )
        # for test stage
        testset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self._transform()
        )
        return trainset, valset, testset

    def _transform(self):
        return torchvision.transforms.ToTensor()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser = pytorch_bolt.DataModule.add_argparse_args(parser)
        return parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(args)
