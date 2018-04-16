#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class ZNet(nn.Module):
  def __init__(self, input_size=786, hidden_size=400, representation_size=20):
    super(Y_Encoder, self).__init__()

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc21 = nn.Linear(hidden_size, representation_size)
    self.fc22 = nn.Linear(hidden_size, representation_size)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def encode(self, x, read):
    input = T.cat([x, read], dim=-1)

    h1 = self.relu(self.fc1(input))
    return self.fc21(h1), self.fc22(h1)

