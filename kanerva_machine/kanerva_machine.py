#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class KanervaMachine(nn.Module):

  def __init__(
    self,
    vae,
    keys_size=100,
    input_size=786,
    memory_size=100,
    hidden_size=400,
    representation_size=20,
    batch_size=50
  ):
    super(KanervaMachine, self).__init__()

    self.batch_size = batch_size
    self.keys_size = keys_size
    self.memory_size = memory_size
    self.representation_size = representation_size

    B = batch_size
    K = keys_size
    C = memory_size
    S = representation_size

    # params for sampling memory
    self.register_parameter('R', nn.Parameter(T.randn(C, C)))
    self.register_parameter('U', nn.Parameter(T.randn(K, K)))
    self.register_parameter('V', nn.Parameter(T.eye(C, C)))

    # self.memory = self.gen_memory(self.R, self.U, self.V) # BxKxC
    self.addresses = nn.Parameter(T.randn(B, K, S))
    self.register_parameter('A', self.addresses)

    self.ynet = VAE(keys_size, input_size, hidden_size, representation_size)
    self.znet = VAE(keys_size, input_size, hidden_size, representation_size)

  def gen_memory(self, R, U, V):
    # generate memory with matrix normal sampling
    memory = []
    for b in range(self.batch_size):
      m = matrix_gaussian(R, U, V)
      memory.append(m)

    return T.stack(memory)

  def read(self, y, memory):
    weights = T.bmm(y.transpose(1,2), self.addresses)
    read_data = T.bmm(weights.transpose(1,2), memory).mean(dim=-1)

    return weights, read_data

  def write(self, memory, z, w, R, U):
    δ = z - (w * R)
    Σ_c = w * U
    Σ_z = w * U * W.transpose(1,2)

    R = R + Σ_c.transpose(1,2) * Σ_z.transpose(1,2) * δ
    U = U - Σ_c.transpose(1,2) * T.inverse(Σ_z) * Σ_c

    self.R = R
    self.U = U

  def forward(self, input):
    # generate memory with params
    memory = self.gen_memory(self.R, self.U, self.V)

    # pass through the reading inference model
    decoded, y, y_mu, y_logvar = self.ynet(input)
    weights, read_data = self.read(y, memory)

    # pass thgough the writing inference model
    z_input = T.cat([input, read_data], dim=-1)
    reconstructed, z, z_mu, z_logvar = self.znet(z_input)
    self.write(memory, z, weights, self.R, self.U)

    return read_weights, read_data, reconstructed,
        (y, z),
        (y_mu, y_logvar, z_mu, z_logvar)

  def loss(self, ):
    pass
