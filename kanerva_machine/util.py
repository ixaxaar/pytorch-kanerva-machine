#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np


def to_numpy(x):
  if type(x) is var:
    x = x.cpu() if x.data.is_cuda else x
    return x.data.numpy()
  else:
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def to_var_like(x, y):
  x = T.from_numpy(x)
  return x.cuda() if y.is_cuda else x


# Refer https://goo.gl/KEJirH
def matrix_gaussian(M, U, V):
  M, U, V = to_numpy(M), to_numpy(U), to_numpy(V)

  return to_var_like(np.random.multivariate_normal(M.ravel(), np.kron(V, U)).reshape(M.shape), M)

