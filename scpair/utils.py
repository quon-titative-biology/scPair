
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
from sklearn.model_selection import train_test_split
import anndata
import scanpy as sc
import scvi
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
    model1.train();
    if model2 is not None:
       model2.train();
    else:
       pass


#%%
def evaluating_mode(model1, model2=None):
    model1.eval();
    if model2 is not None:
       model2.eval();
    else:
       pass


#%%
def train_net(input, output, library_size, encoder, decoder, optimizer, likelihood_type="zinb", add_cov=False, input_batch=None, output_batch=None, scaler=None):
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
    
    if scaler is not None:
        scaler.scale(training_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
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
def train_VAE(input, lib, encoder, decoder, optimizer, kl_weight=1, likelihood_type="zinb", add_cov=False, input_batch=None):
    if add_cov == True:
        mu, sigma, z, latent_lib = encoder(input, input_batch)
        outputs = decoder(z, input_batch, lib, latent_lib)
    else:
        mu, sigma, z, latent_lib = encoder(input, None)
        outputs = decoder(z, None, lib, latent_lib)
    kl_divergence = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0,1)).mean()
    if  (likelihood_type == "zinb") or (likelihood_type == "nb"):
        px_mu_scale, px_theta, px_mu_rate, px_pi = outputs
        reconstruction_loss = raw_loss(x=input, px_mu_rate=px_mu_rate, px_theta=px_theta, px_mu_scale=px_mu_scale, px_pi=px_pi, likelihood=likelihood_type)
    elif likelihood_type == "ber":
        px = outputs[0]
        reconstruction_loss = binarized_loss(input, px)
    elif likelihood_type == "gau":
        px = outputs
        reconstruction_loss = scaled_loss(input, px)
    else:
        print("ERROR")
        raise ValueError
    total_loss = kl_weight * kl_divergence + reconstruction_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


#
def eval_VAE(input, lib, encoder, decoder, kl_weight=1, likelihood_type="zinb", add_cov=False, input_batch=None):
    with torch.no_grad():
        if add_cov == True:
            mu, sigma, z, latent_lib = encoder(input, input_batch)
            outputs = decoder(z, input_batch, lib, latent_lib)
        else:
            mu, sigma, z, latent_lib = encoder(input, None)
            outputs = decoder(z, None, lib, latent_lib)
        eval_kl_divergence = D.kl_divergence(D.Normal(mu, sigma), D.Normal(0,1)).mean()
        if  (likelihood_type == "zinb") or (likelihood_type == "nb"):
            px_mu_scale, px_theta, px_mu_rate, px_pi = outputs
            eval_recon_loss = raw_loss(x=input, px_mu_rate=px_mu_rate, px_theta=px_theta, px_mu_scale=px_mu_scale, px_pi=px_pi, likelihood=likelihood_type)
        elif likelihood_type == "ber":
            px = outputs[0]
            eval_recon_loss = binarized_loss(input, px)
        elif likelihood_type == "gau":
            px = outputs
            eval_recon_loss = scaled_loss(input, px)
        else:
            print("ERROR")
            raise ValueError
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


# # merge the paired data
# def merge_paired_data(paired_objects = [None, None], modality_names = ['Gene Expression','Peaks'], modality_distributions = ['zinb', 'ber']):
#     assert len(paired_objects) == 2, 'paired_objects should be a list of 2 scanpy AnnData objects'
#     assert len(modality_names) == 2, 'modality_name should be a list of 2 strings (e.g. ["Gene Expression", "Peaks"])'
#     assert len(modality_distributions) == 2, 'modality_distributions should be a list of 2 strings (e.g. ["zinb", "ber"])'
#     assert paired_objects[0].obs.index.equals(paired_objects[1].obs.index), 'paired_objects should have the same sample index'
#     paired_obj_dict = dict(zip(modality_names, paired_objects))
#     paired_dist_dict = dict(zip(modality_names, modality_distributions))
#     for name in modality_names:
#         paired_obj_dict[name].var['modality'] = name
#         print('processing modality: {}'.format(name))
#         paired_obj_dict[name].var['modality_distribution'] = paired_dist_dict[name]
#         if paired_dist_dict[name] == 'ber':
#             # check if the data is binary
#             if paired_obj_dict[name].X.min() < 0 or paired_obj_dict[name].X.max() > 1:
#                 print('Bernoulli distribution was set for', name, 'data,', 'but', name, 'data is not binary. Converting to binary...')
#                 paired_obj_dict[name].X = (paired_obj_dict[name].X > 0).astype(int)
#         elif paired_dist_dict[name] == 'zinb' or paired_dist_dict[name] == 'nb':
#             if paired_obj_dict[name].X.min() < 0:
#                 raise ValueError('(Zero-Inflated) Negative Binomial distribution was set for {} data, but {} data has negative values.'.format(name, name))
#             elif paired_obj_dict[name].X.dtype != 'int': # not integer
#                 # warning
#                 warnings.warn('(Zero-Inflated) Negative Binomial distribution was set for {} data, but {} data is not integer.'.format(name, name))
#         elif paired_dist_dict[name] == 'gau':
#             if paired_obj_dict[name].X.min() > 0:
#                 raise ValueError('Gaussian distribution was set for {} data, but {} data has non-negative values.'.format(name, name))
#     adata_paired = anndata.concat([paired_obj_dict[modality_names[0]].T, paired_obj_dict[modality_names[1]].T], merge="same")
#     # adata_paired = paired_obj_dict[modality_names[0]].T.concatenate(paired_obj_dict[modality_names[1]].T, batch_key=None, join='inner')
#     adata_paired = adata_paired.T
#     return adata_paired


# # split the data
# def training_split(scobj, fracs=[0.8, 0.1, 0.1], seed=0, batch_key=None, pre_split=None):
#     if pre_split is not None:
#         train_set, val_set, test_set = pre_split
#         metadata_ = scobj.obs.copy()
#         metadata_['scPair_split'] = 'TBD'
#         metadata_.loc[train_set, 'scPair_split'] = 'train'
#         metadata_.loc[val_set, 'scPair_split'] = 'val'
#         metadata_.loc[test_set, 'scPair_split'] = 'test'
#         if scobj.obs.index.equals(metadata_.index):
#             scobj.obs = metadata_.copy()
#         else:
#             print('The provided metadata does not match the input AnnData object. Returning the metadata only...')
#             return metadata_
#     else:
#         print('No pre-split data provided. Splitting data based on provided fractions...')
#         assert np.isclose(np.sum(fracs), 1, atol=1e-9), 'train_frac + val_frac + test_frac should be approximately 1'
#         assert len(fracs) == 3, 'fracs should be a list of 3 fractions [train_frac, val_frac, test_frac]'
#         train_frac, val_frac, test_frac = fracs
#         assert train_frac >= 0 and val_frac >= 0 and test_frac >= 0, 'train_frac, val_frac, and test_frac should be >= 0'
#         np.random.seed(seed)
#         if batch_key is None:
#             train_set, non_train_set = train_test_split(scobj.obs.index, test_size=val_frac + test_frac, random_state=seed)
#             if test_frac == 0:
#                 val_set = non_train_set
#                 test_set = []
#             else:
#                 val_set, test_set = train_test_split(non_train_set, test_size=test_frac/(val_frac + test_frac), random_state=seed)
#             metadata_ = scobj.obs.copy()
#             metadata_['scPair_split'] = 'TBD'
#             metadata_.loc[train_set, 'scPair_split'] = 'train'
#             metadata_.loc[val_set, 'scPair_split'] = 'val'
#             metadata_.loc[test_set, 'scPair_split'] = 'test'
#             if scobj.obs.index.equals(metadata_.index):
#                 scobj.obs = metadata_.copy() 
#             else:
#                 print('The provided metadata does not match the input AnnData object. Returning the metadata only...')
#                 return metadata_
#         else:
#             print('split the data based on', batch_key, 'column from the metadata')
#             if isinstance(batch_key, list):
#                 for bk in batch_key:
#                     assert bk in scobj.obs.columns, 'batch_key {} should be in the metadata'.format(bk)
#             elif isinstance(batch_key, str):
#                 assert batch_key in scobj.obs.columns, 'batch_key should be in the metadata'
#             else:
#                 raise ValueError('batch_key should be a string or a list of strings')
#             metadata_ = scobj.obs.copy()
#             metadata_['cell_index'] = metadata_.index.tolist()
#             train_set = metadata_.groupby(batch_key).apply(lambda x: x.sample(frac=train_frac, random_state=seed))['cell_index'].tolist()
#             metadata_rest = metadata_.loc[~metadata_.index.isin(train_set)]
#             val_set = metadata_rest.groupby(batch_key).apply(lambda x: x.sample(frac=val_frac/(val_frac + test_frac), random_state=seed))['cell_index'].tolist()
#             test_set = metadata_rest.loc[~metadata_rest.index.isin(val_set)]['cell_index'].tolist()
#             metadata_['scPair_split'] = 'TBD'
#             metadata_.loc[train_set, 'scPair_split'] = 'train'
#             metadata_.loc[val_set, 'scPair_split'] = 'val'
#             metadata_.loc[test_set, 'scPair_split'] = 'test'
#             if scobj.obs.index.equals(metadata_.index):
#                 scobj.obs = metadata_.copy()
#             else:
#                 print('The provided metadata does not match the input AnnData object. Returning the metadata only...')
#                 return metadata_
#     return scobj.copy()


# optimized merge_paired_data and training_split functions
def check_distribution(name, obj, dist):
    if dist == 'ber':
        if obj.X.min() < 0 or obj.X.max() > 1:
            print(f'Bernoulli distribution was set for {name} data, but {name} data is not binary. Converting to binary...')
            obj.X = (obj.X > 0).astype(int)
    elif dist in ['zinb', 'nb']:
        if obj.X.min() < 0:
            raise ValueError(f'(Zero-Inflated) Negative Binomial distribution was set for {name} data, but {name} data has negative values.')
        elif obj.X.dtype != 'int':
            warnings.warn(f'(Zero-Inflated) Negative Binomial distribution was set for {name} data, but {name} data is not integer.')
    elif dist == 'gau':
        if obj.X.min() > 0:
            raise ValueError(f'Gaussian distribution was set for {name} data, but {name} data has non-negative values.')


def merge_paired_data(paired_objects = [None, None], modality_names = ['Gene Expression','Peaks'], modality_distributions = ['zinb', 'ber']):
    paired_obj_dict = dict(zip(modality_names, paired_objects))
    paired_dist_dict = dict(zip(modality_names, modality_distributions))
    for name, obj in paired_obj_dict.items():
        obj.var['modality'] = name
        print(f'processing modality: {name}')
        obj.var['modality_distribution'] = paired_dist_dict[name]
        check_distribution(name, obj, paired_dist_dict[name])
    adata_paired = anndata.concat([paired_obj_dict[modality_names[0]].T, paired_obj_dict[modality_names[1]].T], merge="same")
    adata_paired = adata_paired.T
    return adata_paired


def split_data(scobj, train_frac, val_frac, test_frac, seed, batch_key=None):
    if batch_key is None:
        train_set, non_train_set = train_test_split(scobj.obs.index, test_size=val_frac + test_frac, random_state=seed)
        if test_frac == 0:
            val_set = non_train_set
            test_set = []
        else:
            val_set, test_set = train_test_split(non_train_set, test_size=test_frac/(val_frac + test_frac), random_state=seed)
    else:
        metadata_ = scobj.obs.copy()
        metadata_['cell_index'] = metadata_.index.tolist()
        train_set = metadata_.groupby(batch_key).apply(lambda x: x.sample(frac=train_frac, random_state=seed))['cell_index'].tolist()
        metadata_rest = metadata_.loc[~metadata_.index.isin(train_set)]
        val_set = metadata_rest.groupby(batch_key).apply(lambda x: x.sample(frac=val_frac/(val_frac + test_frac), random_state=seed))['cell_index'].tolist()
        test_set = metadata_rest.loc[~metadata_rest.index.isin(val_set)]['cell_index'].tolist()
    return train_set, val_set, test_set


def training_split(scobj, fracs=[0.8, 0.1, 0.1], seed=0, batch_key=None, pre_split=None):
    assert np.isclose(np.sum(fracs), 1, atol=1e-9), 'train_frac + val_frac + test_frac should be approximately 1'
    assert len(fracs) == 3, 'fracs should be a list of 3 fractions [train_frac, val_frac, test_frac]'
    train_frac, val_frac, test_frac = fracs
    assert train_frac >= 0 and val_frac >= 0 and test_frac >= 0, 'train_frac, val_frac, and test_frac should be >= 0'
    np.random.seed(seed)
    if pre_split is not None:
        train_set, val_set, test_set = pre_split
    else:
        print('No pre-split data provided. Splitting data based on provided fractions...')
        train_set, val_set, test_set = split_data(scobj, train_frac, val_frac, test_frac, seed, batch_key)
    metadata_ = scobj.obs.copy()
    metadata_['scPair_split'] = 'TBD'
    metadata_.loc[train_set, 'scPair_split'] = 'train'
    metadata_.loc[val_set, 'scPair_split'] = 'val'
    metadata_.loc[test_set, 'scPair_split'] = 'test'
    if scobj.obs.index.equals(metadata_.index):
        scobj.obs = metadata_.copy()
    else:
        print('The provided metadata does not match the input AnnData object. Returning the metadata only...')
        return metadata_
    return scobj.copy()