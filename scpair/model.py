# [MESPRIT]: Multi-modal End-to-end Supervised Predictive model for Representation learning and Integration Tasks
# Hongru Hu, hrhu@ucdavis.edu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .scVI_distribution import *
from .utils import *
from .loss import *

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


# modules of the multiview end-to-end predictive model
# Input (mRNA or AC) --> meta-feature/representation layer [ENCODER]
class Input_Module(nn.Module):
    def __init__(self,
                 input_dim:int,
                 input_batch_num:int,
                 hidden_layer:int,
                 layernorm=True,
                 activation=nn.ReLU(),
                 batchnorm=False,
                 dropout_rate=0,
                 add_linear_layer=False,
                 infer_library_size=True,
                 clip_threshold=None):
        super(Input_Module, self).__init__()
        self.infer_library_size = infer_library_size
        self.add_linear_layer = add_linear_layer
        self.clip_threshold = clip_threshold
        # encoder latent representation for each cell
        self.hidden_rep = FCNN([input_dim+input_batch_num] + hidden_layer,
                                layernorm=layernorm,
                                activation=activation,
                                batchnorm=batchnorm,
                                dropout_rate=dropout_rate)
        # add an extra linear layer after hidden_z network to match the structure of encoder of AE/VAE
        if self.add_linear_layer is True:
           self.extra_linear_rep = nn.Linear(hidden_layer[-1], hidden_layer[-1], bias=True) # nn.Sequential(nn.Linear(hidden_layer[-1], hidden_layer[-1], bias=True), nn.LayerNorm(hidden_layer[-1]))
        # encode library size factor for each cell
        if self.infer_library_size == True:
           self.hidden_lib = nn.Sequential(FCNN([input_dim+input_batch_num] + hidden_layer,
                                                 activation=activation,
                                                 layernorm=layernorm,
                                                 batchnorm=batchnorm,
                                                 dropout_rate=dropout_rate),
                                            nn.Linear(hidden_layer[-1], 1, bias=True))
           # self.hidden_lib = nn.Linear(input_dim+input_batch_num, 1, bias=True)
    def forward(self, input, input_batch):
        # infer the latent representation/ meta-feature
        if input_batch is not None:
           latent_rep = self.hidden_rep(torch.cat([input, input_batch], dim=1))
        else:
           latent_rep = self.hidden_rep(input)
        if self.add_linear_layer is True:
           latent_rep = self.extra_linear_rep(latent_rep)
        else:
           latent_rep = latent_rep
        # infer the library size factor
        if self.infer_library_size == True:
           if input_batch is not None:
              latent_lib = self.hidden_lib(torch.cat([input, input_batch], dim=1))
           else:
              latent_lib = self.hidden_lib(input)
        else:
           latent_lib = None
        return latent_rep, latent_lib


# Bernoulli output: BCE loss [binarized_count_loss]
# sample factor might be useful since during cross-modal prediction (mRNA -> AC)
# there is no information about the library size of each sample's AC profile
class Output_Module_Ber(nn.Module):
      def __init__(self,
                   output_dim:int,
                   output_batch_num:int,
                   hidden_layer:int,
                   infer_library_size=True,
                   sample_factor=True,
                   feature_factor=True):
          super(Output_Module_Ber, self).__init__()
          self.infer_library_size = infer_library_size
          self.sample_factor = sample_factor           # sample-specific scalar
          self.feature_factor = feature_factor
          if self.feature_factor == True:
             self.feature_scalar = torch.nn.Parameter(torch.zeros(output_dim))
          self.output_layer = nn.Linear(hidden_layer[-1]+output_batch_num, output_dim, bias=True)
      def forward(self, latent_rep, output_batch, lib, latent_lib):
          if output_batch is not None:
             latent_rep = torch.cat([latent_rep, output_batch], dim=1)
          else:
             latent_rep = latent_rep
          # if latent_lib is not None:
          #    latent_rep = torch.cat([latent_rep, torch.sigmoid(latent_lib)], dim=1)
          # else:
          #    latent_rep = latent_rep
          self.output_temp = self.output_layer(latent_rep)
          output_px = torch.sigmoid(self.output_temp)
          if self.sample_factor == True:
             output_sample_factor = torch.sigmoid(lib)
             output_result = output_px * output_sample_factor
          elif (self.sample_factor == False) and (latent_lib is not None):
             output_sample_factor = torch.sigmoid(latent_lib)
             output_result = output_px * output_sample_factor
          else:
             output_sample_factor = None
             output_result = output_px
          if  self.feature_factor == True:
              output_feature_factor = torch.sigmoid(self.feature_scalar)
              output_result = output_result * output_feature_factor
          else:
              output_feature_factor = None
              output_result = output_result
          return output_result, output_px, output_sample_factor, output_feature_factor



# meta-feature/representation layer --> Output (mRNA, AC, Ephys, etc) [LINEAR DECODER]
# (Zero-Inflated) Negative Binomial output: raw_count_loss
class Output_Module_Raw(nn.Module):
      def __init__(self,
                   output_dim:int,
                   output_batch_num:int,
                   hidden_layer:int,
                   infer_library_size=True,
                   sample_factor=True,
                   feature_factor=False,
                   zero_inflated=True,
                   dispersion="feature-cell",
                   eps=1e-8):
          super(Output_Module_Raw, self).__init__()
          self.infer_library_size = infer_library_size
          self.sample_factor = sample_factor           # sample-specific scalar
          self.feature_factor = feature_factor     # feature-specific scalar
          self.zero_inflated = zero_inflated                  # zero-inflated: bool
          self.dispersion = dispersion             # dispersion tyep
          self.epsilon = eps                       # constant: epsilon
          if self.feature_factor == True:
             self.feature_scalar = torch.nn.Parameter(torch.zeros(output_dim))  # for all the features
          # mean: mu >= 0
          self.output_mean = nn.Sequential(nn.Linear(hidden_layer[-1]+output_batch_num, output_dim, bias=True),
                                           nn.ReLU())
          self.transformation_function = nn.LeakyReLU()
          self.scale_function = nn.Softmax(dim=-1)
          # dispersion: phi = 1/theta > 0
          if self.dispersion == "feature":            # all cells share the same set of dispersion parameters
             self.output_inverse_dispersion = torch.nn.Parameter(torch.randn(output_dim))
          elif self.dispersion == "feature-cell":     # each cell has its own set of dispersion parameters [recommended]
             self.output_inverse_dispersion = nn.Linear(output_dim, output_dim, bias=True)
          # dropout: pi
          if self.zero_inflated == True:
             self.output_dropout = nn.Linear(output_dim, output_dim, bias=True)
      def forward(self, latent_rep, output_batch, lib, latent_lib):
          if output_batch is not None:
             latent_rep = torch.cat([latent_rep, output_batch], dim=1)
          else:
             latent_rep = latent_rep
          # if latent_lib is not None:
          #    latent_rep = torch.cat([latent_rep, latent_lib], dim=1)
          # else:
          #    latent_rep = latent_rep
          px_mu = self.output_mean(latent_rep)
          if self.feature_factor == True:
             px_mu_scale = self.scale_function(px_mu * torch.sigmoid(self.feature_scalar)) # F.softplus(self.feature_scalar)
          else:
             px_mu_scale = self.scale_function(px_mu)
          if self.sample_factor == True:
             px_mu_rate = lib * px_mu_scale
          elif (self.sample_factor == False) and (latent_lib is not None):
             px_mu_rate = torch.exp(self.transformation_function(latent_lib)) * px_mu_scale
          else:
             raise Exception("library size is missing")
          if self.dispersion == "feature":
             px_theta = torch.exp(self.output_inverse_dispersion)#F.softplus(self.output_inverse_dispersion)              # dispersion > 0
          elif self.dispersion == "feature-cell":
             px_theta = torch.exp(px_mu)# + 1e-8
             #px_theta = torch.exp(torch.clip(self.output_inverse_dispersion(px_mu), min = -120, max=12)) + 1e-8
             #px_theta = F.softplus(self.output_inverse_dispersion(px_mu))  # dispersion > 0
          if self.zero_inflated == True:
             px_pi = self.output_dropout(px_mu)                            # dropout [0,1] log(sigmoid(x)) = -softplus(-x) see loss function
          else:
             px_pi = None
          return px_mu_scale, px_theta, px_mu_rate, px_pi


# Gaussian output: used for the prediction of:
#                  scaled mRNA RC/logCPM,
#                  scaled AC TF-IDF,
#                  scaled Ephys features
# apply MSE loss [scaled_loss]
# latent_factor is supposed to be applied in scaled AC TF-IDF prediction only
class Output_Module_Gau(nn.Module):
      def __init__(self,
                   output_dim:int,
                   output_batch_num:int,
                   hidden_layer:int,
                   infer_library_size=True,
                   feature_factor=False):
          super(Output_Module_Gau, self).__init__()
          self.infer_library_size = infer_library_size
          self.feature_factor = feature_factor
          if self.feature_factor == True:
             self.feature_scalar = torch.nn.Parameter(torch.zeros(output_dim))
          self.output_layer = nn.Linear(hidden_layer[-1]+output_batch_num+self.infer_library_size, output_dim, bias=True)
          lib = None
      def forward(self, latent_rep, output_batch, lib, latent_lib):
          if output_batch is not None:
             latent_rep = torch.cat([latent_rep, output_batch], dim=1)
          else:
             latent_rep = latent_rep
          if latent_lib is not None:
             latent_rep = torch.cat([latent_rep, latent_lib], dim=1)
          else:
             latent_rep = latent_rep
          self.output_temp = self.output_layer(latent_rep)
#           if  latent_lib is not None:
#               self.output_temp_ = latent_lib * self.output_temp
#           else:
#               self.output_temp_ = self.output_temp
          if  self.feature_factor == True:
              output_result = self.output_temp * self.feature_scalar # F.softplus(self.feature_scalar)
          else:
              output_result = self.output_temp
          return output_result



# TRANSLATION NETWORK
class Module_Module(nn.Module):
    def __init__(self,
                 origin_module_dim:int,
                 target_module_dim:int,
                 translational_hidden_nodes:int,
                 non_neg=True,
                 layernorm=False,
                 activation=nn.ReLU(),
                 batchnorm=False,
                 dropout_rate=0):
        super(Module_Module, self).__init__()
        self.nonneg = non_neg
        self.translational_hidden_nodes = translational_hidden_nodes
        if self.translational_hidden_nodes is not None:
           self.translate = nn.Sequential(FCNN([origin_module_dim] + translational_hidden_nodes,
                                                activation=activation,
                                                batchnorm=batchnorm,
                                                layernorm=layernorm,
                                                dropout_rate=dropout_rate),
                                          nn.Linear(translational_hidden_nodes[-1], target_module_dim, bias=True))
           # self.translate = FCNN([origin_module_dim] + translational_hidden_nodes + [target_module_dim],
           #                                      activation=activation,
           #                                      batchnorm=batchnorm,
           #                                      layernorm=layernorm,
           #                                      dropout_rate=dropout_rate)
        else:
           self.translate = nn.Linear(origin_module_dim, target_module_dim, bias=True)
    def forward(self, x):
        translation = self.translate(x)
        if self.nonneg == True:
           translation = torch.clamp(translation, min=0)
        return translation


###############################################################################################################################################
