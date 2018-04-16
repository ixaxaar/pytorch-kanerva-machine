#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class Kanerva(nn.Module):
  def __init__(self, input_size=786, memory_size=100, hidden_size=400, representation_size=20):
    super(Kanerva, self).__init__()
    pass

