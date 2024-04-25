
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .scVI_distribution import *

import os
import copy
import random
import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset

# Define Objective Functions/ Loss Functions
# scaled data: scaled mRNA RC/logCPM or scaled Ephys
def scaled_loss(x, mu):
    # Missing data are nan's: [normally observed in Ephys datasets]
    mask = torch.isnan(x)
    out = (mu[~mask]-x[~mask])**2
    loss = out.mean()
    return loss



# binarized data: binarized ATAC peaks
# def binarized_loss(x, p):
#     loss_func = nn.BCELoss()
#     loss = loss_func(p, x)
#     return loss
def binarized_loss(x, p):
    # Missing data are nan's
    mask = torch.isnan(x)
    loss_func = nn.BCELoss()
    loss = loss_func(p[~mask], x[~mask])
    return loss


# raw data: raw counts of mRNA or AC
# px_mu_scale, px_theta, px_mu_rate, px_pi are the output
def raw_loss(x, px_mu_rate, px_theta, px_mu_scale, px_pi, likelihood="zinb"):
    if likelihood == "nb":
       loss = - NegativeBinomial(mu=px_mu_rate, theta=px_theta, scale=px_mu_scale).log_prob(x).mean()
    elif likelihood == "zinb":
       loss = - ZeroInflatedNegativeBinomial(mu=px_mu_rate, theta=px_theta, scale=px_mu_scale, zi_logits=px_pi).log_prob(x).mean()
    return loss
