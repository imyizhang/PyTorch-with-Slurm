#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Check if CUDA device is available


import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('INFO - running on {} device with torch {}'.format(DEVICE, torch.__version__))
