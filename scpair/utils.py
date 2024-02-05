
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .scVI_distribution import *
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
#from torch.utils.tensorboard import SummaryWriter
#from prettytable import PrettyTable
#def model_params_overview(model):
#    table = PrettyTable(["Modules", "Parameters"])
#    total_params = 0
#    for name, parameter in model.named_parameters():
#        if not parameter.requires_grad: continue
#        param = parameter.numel()
#        table.add_row([name, param])
#        total_params+=param
#    print(table)
#    print(f"Total Trainable Parameters: {total_params}")



# Random Seed Setting: allow for reproducibility test
def set_seed(seed=None):
    if seed is None:
       seed = np.random.choice(int(1e2))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Fully Connected Neural Networks: Linear -> LayerNorm -> Activation (-> BatchNorm) -> Dropout
def FCNN(layers, layernorm=True, activation=nn.ReLU(), batchnorm=False, dropout_rate=0):
    fc_nn = []
    for i in range(1, len(layers)):
        fc_nn.append(nn.Linear(layers[i-1], layers[i]))
        if layernorm:
            fc_nn.append(nn.LayerNorm(layers[i]))
        fc_nn.append(activation)
        if batchnorm:
            fc_nn.append(nn.BatchNorm1d(layers[i]))
        fc_nn.append(nn.Dropout(dropout_rate))
    return nn.Sequential(*fc_nn)


# Parameter Initialization
def init_weights(nn_modules, method="xavier_normal"):
    if type(nn_modules) == nn.Linear:
       if method == "xavier_normal":
          torch.nn.init.xavier_normal_(nn_modules.weight)
       elif method == "kaiming_normal":
          torch.nn.init.kaiming_normal_(nn_modules.weight, mode='fan_in', nonlinearity='relu')
       elif method == "xavier_uniform":
          torch.nn.nn.init.xavier_uniform_(nn_modules.weight)
       elif method == "kaiming_uniform":
          torch.nn.init.kaiming_uniform_(nn_modules.weight, mode='fan_in', nonlinearity='relu')
       nn_modules.bias.data.fill_(0.0)


# Weight Decay
def add_weight_decay(model, output_layer=()):
    decay_param = []
    nodecay_param = []
    decay_name = []
    nodecay_name = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.split("_")[0] in output_layer:
            nodecay_name.append(name)
            nodecay_param.append(param)
        else:
            decay_name.append(name)
            decay_param.append(param)
    return decay_param, nodecay_param, decay_name, nodecay_name


#%%
def training_mode(model1, model2=None):
    model1.train()
    if model2 is not None:
       model2.train()
    else:
       pass
    print("...")


#%%
def evaluating_mode(model1, model2=None):
    model1.eval()
    if model2 is not None:
       model2.eval()
    else:
       pass
    print("...")


#%%
def train_net(input, output, library_size, encoder, decoder, optimizer, likelihood_type="zinb", add_cov=False, input_batch=None, output_batch=None):
    if   add_cov == True:
         latent_rep, latent_lib = encoder(input, input_batch)
         if library_size is not None:
            outputs = decoder(latent_rep, output_batch, library_size, None)
         else:
            outputs = decoder(latent_rep, output_batch, None, latent_lib)
    else:
         latent_rep, latent_lib = encoder(input, None)
         if library_size is not None:
            outputs = decoder(latent_rep, None, library_size, None)
         else:
            outputs = decoder(latent_rep, None, None, latent_lib)
    if  (likelihood_type == "zinb") or (likelihood_type == "nb"):
         px_mu_scale, px_theta, px_mu_rate, px_pi = outputs
         training_loss = raw_loss(x=output, px_mu_rate=px_mu_rate, px_theta=px_theta, px_mu_scale=px_mu_scale, px_pi=px_pi, likelihood=likelihood_type)
    elif likelihood_type == "ber":
         training_loss = binarized_loss(output, outputs[0])
    elif likelihood_type == "gau":
         training_loss = scaled_loss(output, outputs)
    else:
         print("ERROR")
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()


#%%
def eval_net(input, output, library_size, encoder, decoder, likelihood_type="zinb", add_cov=False, input_batch=None, output_batch=None):
    if   add_cov == True:
         latent_rep, latent_lib = encoder(input, input_batch)
         if library_size is not None:
            outputs = decoder(latent_rep, output_batch, library_size, None)
         else:
            outputs = decoder(latent_rep, output_batch, None, latent_lib)
    else:
         latent_rep, latent_lib = encoder(input, None)
         if library_size is not None:
            outputs = decoder(latent_rep, None, library_size, None)
         else:
            outputs = decoder(latent_rep, None, None, latent_lib)
    if  (likelihood_type == "zinb") or (likelihood_type == "nb"):
         px_mu_scale, px_theta, px_mu_rate, px_pi = outputs
         eval_total_loss = raw_loss(x=output, px_mu_rate=px_mu_rate, px_theta=px_theta, px_mu_scale=px_mu_scale, px_pi=px_pi, likelihood=likelihood_type)
    elif likelihood_type == "ber":
         eval_total_loss = binarized_loss(output, outputs[0])
    elif likelihood_type == "gau":
         eval_total_loss = scaled_loss(output, outputs)
    else:
         print("ERROR")
    return  eval_total_loss


#
def train_VAE(input, encoder, decoder, optimizer, kl_weight=1, likelihood_type="zinb", add_cov=False, input_batch=None):
    if add_cov == True:
       mu, sigma, z, latent_lib = encoder(input, input_batch)
       outputs = decoder(z, input_batch, latent_lib, input)
    else:
       mu, sigma, z, latent_lib = encoder(input, None)
       outputs = decoder(z, None, latent_lib, input)
    kl_divergence = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0,1)).mean()
    if  (likelihood_type == "zinb") or (likelihood_type == "nb"):
         px_mu_scale, px_theta, px_mu_rate, px_pi, _ = outputs
         reconstruction_loss = raw_loss(x=input, px_mu_rate=px_mu_rate, px_theta=px_theta, px_mu_scale=px_mu_scale, px_pi=px_pi, likelihood=likelihood_type)
    elif likelihood_type == "ber":
         px = outputs[0]
         reconstruction_loss = binarized_loss(input, px)
    elif likelihood_type == "gau":
         px = outputs[0]
         reconstruction_loss = scaled_loss(input, px)
    else:
         print("ERROR")
    total_loss = kl_weight * kl_divergence + reconstruction_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#
def eval_VAE(input, encoder, decoder, kl_weight=1, likelihood_type="zinb", add_cov=False, input_batch=None):
    if add_cov == True:
       mu, sigma, z, latent_lib = encoder(input, input_batch)
       outputs = decoder(z, input_batch, latent_lib, input)
    else:
       mu, sigma, z, latent_lib = encoder(input, None)
       outputs = decoder(z, None, latent_lib, input)
    eval_kl_divergence = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0,1)).mean()
    if  (likelihood_type == "zinb") or (likelihood_type == "nb"):
         px_mu_scale, px_theta, px_mu_rate, px_pi, _ = outputs
         eval_recon_loss = raw_loss(x=input, px_mu_rate=px_mu_rate, px_theta=px_theta, px_mu_scale=px_mu_scale, px_pi=px_pi, likelihood=likelihood_type)
    elif likelihood_type == "ber":
         px = outputs[0]
         eval_recon_loss = binarized_loss(input, px)
    elif likelihood_type == "gau":
         px = outputs[0]
         eval_recon_loss = scaled_loss(input, px)
    else:
         print("ERROR")
    eval_total_loss = kl_weight * eval_kl_divergence + eval_recon_loss
    return  eval_total_loss, eval_recon_loss, eval_kl_divergence



from scipy import sparse
import torch
def collate_wrapper(batch, omic_combn):
    dataset = [x[1] for x in batch]
    batch = [x[0] for x in batch]
    dataset = [torch.tensor(list(x)) if include else None
               for x, include in zip(zip(*dataset), omic_combn)]
    batch = [torch.from_numpy(sparse.vstack(x).toarray()).float() if include else None
             for x, include in zip(zip(*batch), omic_combn)]
    return batch, dataset

