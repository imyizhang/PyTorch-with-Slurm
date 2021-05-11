#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#<https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html>
#<https://pytorch.org/vision/stable/models.html>

import argparse

import torch
import torchvision
import pytorch_bolt


class Accuracy(torch.nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        if reduction not in ('sum', 'mean'):
            raise ValueError
        self.reduction = reduction

    def forward(self, outputs, targets):
        with torch.no_grad():
            correct = torch.eq(outputs.argmax(dim=1), targets).type(torch.float)
            if self.reduction == 'sum':
                return correct.sum()
            elif self.reduction == 'mean':
                return correct.mean()


class MNISTClassifier(pytorch_bolt.Module):

    def __init__(self, args):
        super().__init__()
        # setup model
        self.in_channels = args.in_channels
        self.num_classes = args.num_classes
        self.net = args.net
        self.pretrained = args.pretrained
        self.finetuning = args.finetuning
        self.model = self._setup_model()
        # configure criterion and metric
        self.reduction = args.reduction
        # configure optimizer and lr_scheduler
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.lr_step = args.lr_step
        self.lr_gamma = args.lr_gamma

    def _setup_model(self):
        model = None
        if self.net == 'resnet':
            model = torchvision.models.resnet18(pretrained=self.pretrained)
            self._set_parameters_requires_grad(model)
            # parameters of newly constructed modules have `requires_grad=True` by default
            if self.in_channels != 3:
                model.conv1 = torch.nn.Conv2d(
                    self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            num_features = model.fc.in_features
            model.fc = torch.nn.Linear(num_features, self.num_classes)
        else:
            raise ValueError
        return model

    def _set_parameters_requires_grad(self, model):
        # consider pretrained model as feature extracter
        if self.pretrained and (not self.finetuning):
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, inputs):
        return self.model(inputs)

    def parameters_to_update(self):
        params_to_update = self.model.parameters()
        # consider pretrained model as feature extracter
        if self.pretrained and (not self.finetuning):
            params_to_update = []
            for param in self.model.parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
        return params_to_update

    def configure_criterion(self):
        return torch.nn.CrossEntropyLoss(reduction=self.reduction)

    def configure_metric(self):
        return Accuracy(reduction=self.reduction)

    def configure_optimizer(self):
        optimizer = torch.optim.Adam(
            self.parameters_to_update(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step,
            gamma=self.lr_gamma
        )
        return optimizer, lr_scheduler

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--in_channels',
            default=1,
            type=int,
            help='[MNISTClassifier] channels of inputs',
        )
        parser.add_argument(
            '--num_classes',
            default=10,
            type=int,
            help='[MNISTClassifier] number of classes',
        )
        parser.add_argument(
            '--net',
            default='resnet',
            type=str,
            help='[MNISTClassifier] neural network name',
        )
        parser.add_argument(
            '--pretrained',
            default=False,
            action='store_true',
            help='[MNISTClassifier] set to True to use a pretrained model',
        )
        parser.add_argument(
            '--finetuning',
            default=False,
            action='store_true',
            help='[MNISTClassifier] set to True to finetune a pretrained model, otherwise consider it as feature extracter',
        )
        parser.add_argument(
            '--reduction',
            default='mean',
            type=str,
            help='[MNISTClassifier] reduction to apply to the outputs, "mean" or "sum"',
        )
        parser.add_argument(
            '--lr',
            default=1e-3,
            type=float,
            help='[MNISTClassifier] learning rate',
        )
        parser.add_argument(
            '--weight_decay',
            default=0.0,
            type=float,
            help='[MNISTClassifier] weight decay (L2 penalty)',
        )
        parser.add_argument(
            '--lr_step',
            default=100,
            type=int,
            help='[MNISTClassifier] epochs of learning rate decay',
        )
        parser.add_argument(
            '--lr_gamma',
            default=0.1,
            type=float,
            help='[MNISTClassifier] multiplicative factor of learning rate decay',
        )
        return parser

    @classmethod
    def from_argparse_args(cls, args):
        return cls(args)
