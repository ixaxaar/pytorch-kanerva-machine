#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Memory(nn.Module):
  def __init__(self, input_size, memory_size, keys_size):
    super(Memory, self).__init__()
    pass
