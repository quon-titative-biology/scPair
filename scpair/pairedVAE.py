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



# VAE Encoder
class VAE_ENCODER(nn.Module):
    def __init__(self,
                input_dim:int,
                input_batch_num:int,
                hidden_dim:int,
                latent_dim:int,
                layernorm=True,
                activation=nn.LeakyReLU(),
                batchnorm=False,
                dropout_rate=0,
                infer_library_size=False):
        super(VAE_ENCODER, self).__init__()
        self.infer_library_size=infer_library_size
        self.enc_hidden = FCNN([input_dim+input_batch_num] + hidden_dim,
                                layernorm=layernorm,
                                activation=activation,
                                batchnorm=batchnorm,
                                dropout_rate=dropout_rate)
        if self.infer_library_size==True:
            self.enc_lib = nn.Sequential(FCNN([input_dim+input_batch_num] + hidden_dim,
                                                layernorm=layernorm,
                                                activation=activation,
                                                batchnorm=batchnorm,
                                                dropout_rate=dropout_rate),
                                        nn.Linear(hidden_dim[-1], 1, bias=True))
        self.latent_mu = nn.Linear(hidden_dim[-1], latent_dim, bias=True)
        self.latent_sigma = nn.Linear(hidden_dim[-1], latent_dim, bias=True)
    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        return z
    def bottleneck(self, enc_h):
        mu = self.latent_mu(enc_h)
        sigma = F.softplus(self.latent_sigma(enc_h)) + 1e-7
        return mu, sigma
    def forward(self, x, x_batch):
        if x_batch is not None:
            enc_h = self.enc_hidden(torch.cat([x, x_batch], dim=1))
        else:
            enc_h = self.enc_hidden(x)
        mu, sigma = self.bottleneck(enc_h)
        z = self.reparameterize(mu, sigma)
        if self.infer_library_size==True:
            if x_batch is not None:
                latent_lib = self.enc_lib(torch.cat([x, x_batch], dim=1))
            else:
                latent_lib = self.enc_lib(x)
            latent_lib = F.softplus(latent_lib)
        else:
            latent_lib = None
        return mu, sigma, z, latent_lib


# VAE Decoder
class VAE_DECODER(nn.Module):
    def __init__(self,
                input_dim:int,
                input_batch_num:int,
                hidden_dim:int,
                latent_dim:int,
                layernorm=True,
                activation=nn.LeakyReLU(),
                batchnorm=False,
                dropout_rate=0,
                infer_library_size=False,
                sample_factor=True,
                feature_factor=False,
                distribution="zinb",
                dispersion="feature-cell"):
        super(VAE_DECODER, self).__init__()
        hidden_dim_decoder = list(np.flip(hidden_dim))
        self.feature_factor = feature_factor
        self.infer_library_size = infer_library_size
        self.distribution = distribution
        self.dispersion = dispersion
        self.sample_factor = sample_factor  # sample-specific scalar, lib size
        if self.feature_factor == True:
            self.feature_scalar = torch.nn.Parameter(torch.zeros(input_dim))
        if self.infer_library_size == True:
            lib_dim = 1
        else:
            lib_dim = 0
        self.dec_hidden = FCNN([latent_dim + input_batch_num + lib_dim] + hidden_dim_decoder,
                                layernorm=layernorm,
                                activation=activation,
                                batchnorm=batchnorm,
                                dropout_rate=dropout_rate)
        self.output = nn.Linear(hidden_dim_decoder[-1], input_dim, bias=True)
        if (self.distribution  == "zinb") or (self.distribution  == "nb"):
            self.scale_function = nn.Softmax(dim=-1)
            if self.dispersion == "feature":            # all cells share the same set of dispersion parameters
                self.output_inverse_dispersion = torch.nn.Parameter(torch.randn(input_dim))
            elif self.dispersion == "feature-cell":     # each cell has its own set of dispersion parameters [recommended]
                self.output_inverse_dispersion = nn.Linear(hidden_dim_decoder[-1], input_dim, bias=True)
            if self.distribution  == "zinb":
                self.output_dropout = nn.Linear(hidden_dim_decoder[-1], input_dim, bias=True)
        self.transformation_function = nn.LeakyReLU()
    def reconstruction(self, dec_h):
        output_temp = self.output(dec_h)
        return output_temp
    def forward(self, z, x_batch, lib, latent_lib):
        if x_batch is not None:
            z = torch.cat([z, x_batch], dim=1)
        if latent_lib is not None:
            z = torch.cat([z, latent_lib], dim=1)
        else:
            z = z
        dec_h = self.dec_hidden(z)
        output_temp = self.reconstruction(dec_h)
        if   self.distribution == 'gau':
                if  self.feature_factor == True:
                    px = output_temp * self.feature_scalar
                else:
                    px = output_temp
                return px
        elif self.distribution == 'ber':
                if  self.feature_factor == True:
                    px = torch.sigmoid(output_temp * self.feature_scalar) # F.softplus(self.feature_scalar))
                else:
                    px = torch.sigmoid(output_temp)
                return px
        elif self.distribution == 'nb':
                px_pi = None
                if  self.feature_factor == True:
                    # px_mu_scale = self.scale_function(output_temp * self.feature_scalar)
                    px_mu_scale = self.scale_function(output_temp * torch.sigmoid(self.feature_scalar)) # F.softplus(self.feature_scalar)
                else:
                    px_mu_scale = self.scale_function(output_temp)
                if self.sample_factor == True:
                    assert lib is not None
                    px_mu_rate = lib * px_mu_scale
                elif (self.sample_factor == False) and (latent_lib is not None):
                    px_mu_rate = torch.exp(self.transformation_function(latent_lib)) * px_mu_scale
                else:
                    px_mu_rate = px_mu_scale
                if self.dispersion == "feature":
                    # px_theta = F.softplus(self.output_inverse_dispersion)
                    px_theta = torch.exp(self.output_inverse_dispersion) #F.softplus(self.output_inverse_dispersion)  # dispersion > 0
                elif self.dispersion == "feature-cell":
                    # px_theta = torch.exp(output_temp) # + 1e-8, or torch.exp(dec_h) 
                    px_theta = F.softplus(self.output_inverse_dispersion(dec_h)) + 1e-8
                return px_mu_scale, px_theta, px_mu_rate, px_pi
        elif self.distribution == 'zinb':
                px_pi = self.output_dropout(dec_h)
                if  self.feature_factor == True:
                    # px_mu_scale = self.scale_function(output_temp * self.feature_scalar)
                    px_mu_scale = self.scale_function(output_temp * torch.sigmoid(self.feature_scalar)) # F.softplus(self.feature_scalar)
                else:
                    px_mu_scale = self.scale_function(output_temp)
                if self.sample_factor == True:
                    assert lib is not None
                    px_mu_rate = lib * px_mu_scale
                elif (self.sample_factor == False) and (latent_lib is not None):
                    px_mu_rate = torch.exp(self.transformation_function(latent_lib)) * px_mu_scale
                else:
                    px_mu_rate = px_mu_scale
                if self.dispersion == "feature":
                    # px_theta = F.softplus(self.output_inverse_dispersion)
                    px_theta = torch.exp(self.output_inverse_dispersion) #F.softplus(self.output_inverse_dispersion)  # dispersion > 0
                elif self.dispersion == "feature-cell":
                    # px_theta = torch.exp(output_temp) # + 1e-8, or torch.exp(dec_h) 
                    # px_theta = torch.exp(torch.clip(self.output_inverse_dispersion(dec_h), min = -120, max=12)) + 1e-8 
                    px_theta = F.softplus(self.output_inverse_dispersion(dec_h)) + 1e-8
                return px_mu_scale, px_theta, px_mu_rate, px_pi


