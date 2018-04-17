#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

# Modified from https://goo.gl/XAdi8y

class MNIST_VAE(nn.Module):
  def __init__(self, input_size=786, output_size=786, hidden_size=400, representation_size=20):
    super(VAE, self).__init__()

    self.input_size = input_size

    self.fc1 = nn.Linear(input_size, input_size)
    self.fc21 = nn.Linear(input_size, representation_size)
    self.fc22 = nn.Linear(input_size, representation_size)

    self.fc3 = nn.Linear(representation_size, output_size)
    self.fc4 = nn.Linear(output_size, output_size)

    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def encode(self, x):
    h1 = self.relu(self.fc1(x))
    return self.fc21(h1), self.fc22(h1)

  def decode(self, z):
    h3 = self.relu(self.fc3(z))
    return self.sigmoid(self.fc4(h3))

  def reparameterize(self, mu, logvar):
    if self.training:
      std = logvar.mul(0.5).exp_()
      eps = Variable(std.data.new(std.size()).normal_())
      return eps.mul(std).add_(mu)
    else:
      return mu

  def forward(self, x):
    mu, logvar = self.encode(x.view(-1, self.input_size))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), z, mu, logvar

  # def loss(self, reconstructed, original, mean, logsigma):
  #   BCE = F.binary_cross_entropy(reconstructed, original.view(-1, self.input_size), size_average=False)

  #   # see Appendix B from VAE paper:
  #   # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
  #   # https://arxiv.org/abs/1312.6114
  #   # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
  #   KLD = -0.5 * torch.sum(1 + logsigma - mean.pow(2) - logsigma.exp())

  #   return BCE + KLD
