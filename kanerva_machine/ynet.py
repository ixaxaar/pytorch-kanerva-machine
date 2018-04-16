#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class YNet(nn.Module):
  def __init__(self, keys_size, input_size=786, hidden_size=400, representation_size=20):
    super(Y_Encoder, self).__init__()

    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc21 = nn.Linear(hidden_size, representation_size)
    self.fc22 = nn.Linear(hidden_size, representation_size)

    self.fc3 = nn.Linear(representation_size, keys_size)
    self.fc4 = nn.Linear(keys_size, keys_size)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def encode(self, x):
    h1 = self.relu(self.fc1(x))
    return self.fc21(h1), self.fc22(h1)

  def decode(self, y):
    h3 = self.relu(self.fc3(z))
    return self.sigmoid(self.fc4(h3))

  def read(self, y, addresses, memory):
    weights = T.bmm(y.transpose(1,2), addresses)
    read_data = T.bmm(weights.transpose(1,2), memory).mean(dim=-1)

    return weights, read_data

  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = Variable(std.data.new(std.size()).normal_())
      return eps.mul(std).add_(mu)
    else:
      return mu

  def forward(self, x, addresses, memory):
    mu, logvar = self.encode(x.view(-1, self.input_size))
    y = self.reparameterize(mu, logvar)
    decoded = self.decode(y)
    weights, read_data = self.read(decoded, addresses, memory)

    return weights, read_data, mu, logvar

  def loss(self, mean, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

