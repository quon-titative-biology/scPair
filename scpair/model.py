from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .scVI_distribution import *
from .utils import *
from .loss import *

import os
import copy
import random
import gc
import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint


# Memory-efficient dataset class for lazy GPU loading
class LazyGPUDataset(Dataset):
    """Memory-efficient dataset that loads data to GPU only when needed."""
    def __init__(self, *tensors, device='cuda'):
        self.tensors = tensors
        self.device = device
        self.length = len(tensors[0]) if tensors else 0
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        # Only move the specific batch to GPU when accessed
        return tuple(tensor[idx] for tensor in self.tensors)


# Memory management utilities
def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def move_to_device_safe(tensor, device):
    """Safely move tensor to device with memory management."""
    if tensor is None:
        return None
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=True)


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
        # Use gradient checkpointing for memory efficiency only during training
        if self.training and input.requires_grad:
            return self._forward_checkpointed(input, input_batch)
        else:
            return self._forward_normal(input, input_batch)
    
    def _forward_checkpointed(self, input, input_batch):
        """Forward pass with gradient checkpointing for memory efficiency."""
        def checkpointed_forward(input, input_batch):
            if input_batch is not None:
                latent_rep = self.hidden_rep(torch.cat([input, input_batch], dim=1))
            else:
                latent_rep = self.hidden_rep(input)
            if self.add_linear_layer is True:
                latent_rep = self.extra_linear_rep(latent_rep)
            return latent_rep
        
        latent_rep = checkpoint(checkpointed_forward, input, input_batch)
        
        # infer the library size factor
        if self.infer_library_size == True:
           if input_batch is not None:
              latent_lib = self.hidden_lib(torch.cat([input, input_batch], dim=1))
           else:
              latent_lib = self.hidden_lib(input)
        else:
           latent_lib = None
        return latent_rep, latent_lib
    
    def _forward_normal(self, input, input_batch):
        """Normal forward pass without checkpointing."""
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
             raise Exception("The library size is missing. Please ensure that either the 'sample_factor' or 'infer_library_size' parameter for the raw counts decoder is set to True.")
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
import tqdm
class scPair_object():
    def __init__(self,
                 scobj,
                 modalities = {'Gene Expression': 'zinb','Peaks': 'ber'}, 
                 cov = ['labels'],
                 SEED = 0,
                 hidden_layer = [900, 40],
                 dropout_rate = 0.1,
                 batchnorm = False,
                 layernorm = True,
                 learning_rate_prediction = 1e-3,
                 batch_size = 130,
                 L2_lambda = 1e-9,
                 max_epochs = 1000,
                 activation = nn.LeakyReLU(),
                 early_stopping_activation = True,
                 early_stopping_patience = 25,
                 weight_decay = None, # 'ExponentialLR', 'StepLR', 'ReduceLROnPlateau'
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 save_model = True,
                 save_path = None,
                 input_module = Input_Module,
                 output_module = {'nb':Output_Module_Raw, 'zinb':Output_Module_Raw, 'ber':Output_Module_Ber, 'gau':Output_Module_Gau},
                 add_linear_layer = True, 
                 sample_factor_rna = True,
                 feature_factor_rna = False,
                 sample_factor_atac = False,
                 feature_factor_atac = False,
                 zero_inflated = True,
                 dispersion = "feature-cell",
                 infer_library_size_rna = False,
                 infer_library_size_atac = True,
                 mapping_module = Module_Module,
                 mapping_hidden_nodes = None,
                 mapping_learning_rate = 1e-3,
                 mapping_non_neg = False,
                 mapping_layernorm = False,
                 mapping_batchnorm = False,
                 mapping_activation = nn.LeakyReLU(),
                 mapping_dropout_rate = 0,
                 use_mixed_precision = False,
                 gradient_checkpointing = True,
                 pin_memory = True,
                 memory_efficient_loading = True):
        """
        scPair framework and optimization
        Input: each modality's cell x feature matrix, one-hoe encoded covariates
        Output: the other modality's cell x feature matrix
        """
        super(scPair_object, self).__init__()
        self.scobj = scobj
        self.modalities = modalities
        self.modality_names = list(modalities.keys())
        self.modality_distributions = list(modalities.values())
        self.cov = cov
        self.SEED = SEED
        self.hidden_layer = hidden_layer
        self.dropout_rate = dropout_rate
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.learning_rate_prediction = learning_rate_prediction
        self.mapping_learning_rate = mapping_learning_rate
        self.batch_size = batch_size
        self.L2_lambda = L2_lambda
        self.max_epochs = max_epochs
        self.activation = activation
        self.early_stopping_activation = early_stopping_activation
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
        self.device = device
        self.save_model = save_model
        self.save_path = save_path
        self.input_module = input_module
        self.output_module = output_module
        self.add_linear_layer = add_linear_layer
        self.sample_factor_rna = sample_factor_rna
        self.feature_factor_rna = feature_factor_rna
        self.sample_factor_atac = sample_factor_atac
        self.feature_factor_atac = feature_factor_atac
        self.zero_inflated = zero_inflated
        self.dispersion = dispersion
        self.infer_library_size_rna = infer_library_size_rna
        self.infer_library_size_atac = infer_library_size_atac
        self.mapping_module = mapping_module
        self.mapping_hidden_nodes = mapping_hidden_nodes
        self.mapping_non_neg = mapping_non_neg
        self.mapping_layernorm = mapping_layernorm
        self.mapping_batchnorm = mapping_batchnorm
        self.mapping_activation = mapping_activation
        self.mapping_dropout_rate = mapping_dropout_rate
        self.use_mixed_precision = use_mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.pin_memory = pin_memory
        self.memory_efficient_loading = memory_efficient_loading
        
        # Initialize mixed precision scaler if needed
        self.scaler = GradScaler() if self.use_mixed_precision and torch.cuda.is_available() else None
        if self.add_linear_layer is True and self.mapping_non_neg is True:
            raise ValueError('add_linear_layer and mapping_non_neg cannot be both True')
        if self.mapping_non_neg is True:
            Warning('mapping_non_neg is True, the output of the mapping module will be clipped to be non-negative.' '\n',
                    'Please ensure the activate function of input modules has non-negative output.')
        if self.mapping_non_neg is False and self.add_linear_layer is False:
            Warning('mapping_non_neg is False and add_linear_layer is False, the output of the mapping module will be unbounded.' '\n',
                    'Please ensure the activate function of input modules has unbounded output.')
        if self.sample_factor_rna is False and self.infer_library_size_rna is False:
            raise ValueError('at leaset one of sample_factor_rna and infer_library_size_rna should be True')
        if self.sample_factor_rna is True and self.infer_library_size_rna is True:
            raise ValueError('sample_factor_rna and infer_library_size_rna cannot be both True')
    def data_loader_builder(self):
        data = self.scobj.copy()
        cov = self.cov
        # Check the type of each covariate and process accordingly
        discrete_cov = []
        continuous_cov = []
        if cov is not None:
            for c in cov:
                if data.obs[c].dtype.name == 'category' or data.obs[c].dtype.name == 'object':
                    # Convert categorical covariates to dummy variables
                    cov_discrete = pd.get_dummies(data.obs[c]).astype(int)
                    discrete_cov.append(cov_discrete)
                else:
                    # Normalize continuous covariates
                    cov_continuous = (data.obs[c] -  data.obs[c].mean())/ data.obs[c].std()
                    continuous_cov.append(cov_continuous)
        self.cov_discrete = pd.concat(discrete_cov, axis=1) if len(discrete_cov) > 0 else None
        self.cov_continuous = pd.concat(continuous_cov, axis=1) if len(continuous_cov) > 0 else None
        if self.cov_discrete is None and self.cov_continuous is None:
            self.cov_dummy = None
        elif self.cov_discrete is not None and self.cov_continuous is not None:
            self.cov_dummy = pd.concat((self.cov_discrete, self.cov_continuous), axis=1)
        elif self.cov_discrete is not None and self.cov_continuous is None:
            self.cov_dummy = self.cov_discrete
        elif self.cov_discrete is None and self.cov_continuous is not None:
            self.cov_dummy = self.cov_continuous
        if self.cov_dummy is not None:
            data.obs = data.obs = pd.concat([data.obs['scPair_split'], self.cov_dummy], axis=1) # pd.concat([data.obs, self.cov_dummy], axis=1)
    # def data_loader_builder(self):
    #         data = self.scobj.copy()
    #         cov = self.cov
    #         # Convert categorical covariates to dummy variables
    #         cov_dummy = pd.get_dummies(data.obs[cov]).astype(int) if cov is not None else None
    #         if cov_dummy is not None:
    #             data.obs = pd.concat([data.obs, cov_dummy], axis=1)
    #         self.cov_dummy = cov_dummy
        # Split data into train, validation, and test sets
        split_data = {split: data[data.obs['scPair_split'] == split] for split in ['train', 'val', 'test']}
        self.train_metadata = split_data['train'].obs.copy()
        self.val_metadata = split_data['val'].obs.copy()
        self.test_metadata = split_data['test'].obs.copy() if 'test' in split_data else None
        # Convert sparse matrix to dense matrix
        if scipy.sparse.issparse(data.X):
            print('Converting sparse matrix to dense matrix...')
            data.X = data.X.toarray()
        # Initialize data loaders with memory optimization
        print('Initializing data loaders...')
        data_loader_dict = {}
        for modality in self.modality_names:
            print('processing modality:', modality)
            for split in ['train', 'val', 'test']:
                if split in split_data:
                    # Keep data on CPU initially for memory efficiency
                    modality_data = split_data[split][:, data.var['modality'] == modality].X
                    if scipy.sparse.issparse(modality_data):
                        modality_data = modality_data.toarray()
                    
                    data_loader_dict[f'{modality}_{split}_mtx'] = torch.FloatTensor(modality_data)
                    data_loader_dict[f'{modality}_{split}_cov'] = torch.FloatTensor(self.cov_dummy.loc[split_data[split].obs.index].values) if cov is not None else None
                    if modality in ['Gene Expression', 'Peaks']:
                        data_loader_dict[f'{modality}_{split}_lib'] = data_loader_dict[f'{modality}_{split}_mtx'].sum(1).reshape(-1,1)
        # Store data loader keys and split data loaders by train, validation, and test sets
        self.data_loader_keys = list(data_loader_dict.keys())
        for split in ['train', 'val', 'test']:
            if f'{modality}_{split}_mtx' in data_loader_dict:
                setattr(self, f'data_loader_dict_{split}', {key: data_loader_dict[key] for key in self.data_loader_keys if split in key})
        return self.data_loader_dict_train, self.data_loader_dict_val, self.data_loader_dict_test
    def run(self):
        # Build data loaders and train predicting networks
        self.encoder_dict, self.decoder_dict = self.train_predicting_networks(*self.data_loader_builder())
        # Generate reference embeddings
        low_dim_embeddings, _ = self.reference_embeddings()
        # Train bidirectional mapping and store the mapping dictionary
        self.mapping_dict = self.train_bidirectional_mapping()
        # Generate mapped embeddings
        low_dim_embeddings_mapped, _ = self.mapped_embeddings()
        return self.encoder_dict, self.decoder_dict, self.mapping_dict, low_dim_embeddings, low_dim_embeddings_mapped
    def reference_embeddings(self):
        embeddings = {}
        df_embeddings = {}
        for modality in self.modality_names:
            # Get the encoder for the current modality
            encoder_cal = self.encoder_dict[modality + ' to ' + [mn for mn in self.modality_names if mn != modality][0]]
            encoder_cal.eval()
            encoder_cal.to(self.device)
            # Generate embeddings for train, validation, and test sets
            for split in ['train', 'val', 'test']:
                if hasattr(self, f'data_loader_dict_{split}'):
                    data_loader_dict = getattr(self, f'data_loader_dict_{split}')
                    embeddings[f'{modality}_{split}'] = encoder_cal(data_loader_dict[f'{modality}_{split}_mtx'].to(self.device), data_loader_dict[f'{modality}_{split}_cov'].to(self.device))[0].cpu().detach() if self.cov is not None else encoder_cal(data_loader_dict[f'{modality}_{split}_mtx'].to(self.device), None)[0].cpu().detach()
                    df_embeddings[f'{modality}_{split}'] = pd.DataFrame(embeddings[f'{modality}_{split}'].numpy(), index = getattr(self, f'{split}_metadata').index)
        self.embeddings = embeddings
        return embeddings, df_embeddings
    def mapped_embeddings(self):
        mapped_embeddings = {}
        df_mapped_embeddings = {}
        for input_modality in self.modality_names:
            output_modality = [mn for mn in self.modality_names if mn != input_modality][0]
            mapping_network = self.mapping_dict[input_modality + '_to_' + output_modality]
            mapping_network.eval()
            mapping_network.to(self.device)
            # Generate mapped embeddings for train, validation, and test sets
            for split in ['train', 'val', 'test']:
                if hasattr(self, f'data_loader_dict_{split}'):
                    input_embedding = self.embeddings[input_modality + f'_{split}']
                    mapped_embeddings[f'{input_modality} to {output_modality}_{split}'] = mapping_network(input_embedding.to(self.device)).cpu().detach()
                    df_mapped_embeddings[f'{input_modality} to {output_modality}_{split}'] = pd.DataFrame(mapped_embeddings[f'{input_modality} to {output_modality}_{split}'].numpy(), index = getattr(self, f'{split}_metadata').index)
        return mapped_embeddings, df_mapped_embeddings
    def predict(self):
        predictions = {}
        # Iterate over each modality
        for input_modality in self.modality_names:
            output_modality = [mn for mn in self.modality_names if mn != input_modality][0]
            print('predicting from', input_modality, 'to', output_modality)
            print('input_distribution:', self.modalities[input_modality], 'output_distribution:', self.modalities[output_modality])
            # Get the decoder for the current modality
            decoder_cal = self.decoder_dict[input_modality + ' to ' + output_modality]
            decoder_cal.eval()
            decoder_cal.to(self.device)
            # Check the output distribution and infer library size if necessary
            if self.modalities[output_modality] in ['ber', 'nb', 'zinb'] and ((output_modality == 'Peaks' and self.infer_library_size_atac) or (output_modality == 'Gene Expression' and self.infer_library_size_rna)):
                encoder_cal = self.encoder_dict[input_modality + ' to ' + output_modality]
                encoder_cal.eval()
                encoder_cal.to(self.device)
                latent = {split: encoder_cal(getattr(self, f'data_loader_dict_{split}')[input_modality + f'_{split}_mtx'].to(self.device),
                                            getattr(self, f'data_loader_dict_{split}')[input_modality + f'_{split}_cov'].to(self.device) if self.cov is not None else None)[1]
                        for split in ['train', 'val', 'test'] if hasattr(self, f'data_loader_dict_{split}')}
            else:
                latent = {split: None for split in ['train', 'val', 'test']}
            # Generate predictions for train, validation, and test sets
            for split in ['train', 'val', 'test']:
                if hasattr(self, f'data_loader_dict_{split}'):
                    prediction = decoder_cal(latent_rep=self.embeddings[input_modality + f'_{split}'].to(self.device),
                                            output_batch=getattr(self, f'data_loader_dict_{split}')[output_modality + f'_{split}_cov'].to(self.device) if self.cov is not None else None,
                                            lib=torch.FloatTensor([1]).to(self.device) if self.modalities[output_modality] in ['ber', 'nb', 'zinb'] else None,
                                            latent_lib=latent[split])
                    # If the output distribution is 'gau', don't index the prediction
                    if self.modalities[output_modality] == 'gau':
                        predictions[output_modality + f"_{split}"] = prediction.cpu().detach().numpy()
                    else:
                        predictions[output_modality + f"_{split}"] = prediction[0].cpu().detach().numpy()
        return predictions
    def get_scheduler(self, optimizer):
        """Create a learning rate scheduler."""
        if self.weight_decay == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1, threshold=0.00001)
        elif self.weight_decay == 'StepLR':
            return torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.1, last_epoch=-1)
        elif self.weight_decay == 'ReduceLROnPlateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
        else:
            print('Warning: weight_decay should be one of the following: ExponentialLR, StepLR, ReduceLROnPlateau. No weight decay applied.')
            return None
    def train_predicting_networks(self, train_data_dict, val_data_dict, test_data_dict):
        # Initialize dictionaries to store encoders and decoders
        encoder_dict = {}
        decoder_dict = {}
        # Loop over each modality
        for input_modality in self.modalities.keys():
            output_modality = [modality for modality in self.modalities.keys() if modality != input_modality][0]
            prediction_order = {'input':input_modality, 'output':output_modality}
            print('Set up predicting networks:', prediction_order)
            # Get training and validation data
            trainData, trainLabel = train_data_dict[input_modality + '_train_mtx'], train_data_dict[output_modality + '_train_mtx']
            valData, valLabel = val_data_dict[input_modality + '_val_mtx'].to(self.device), val_data_dict[output_modality + '_val_mtx'].to(self.device)
            trainBatch, valBatch = (train_data_dict[input_modality + '_train_cov'], val_data_dict[input_modality + '_val_cov'].to(self.device)) if self.cov is not None else (None, None)
            # Initialize encoder and decoder
            # Check if the input modality is one of the specified distributions
            if self.modalities[input_modality] in ['zinb', 'nb', 'ber', 'gau']:
                # Initialize encoder
                set_seed(self.SEED)
                encoder = self.input_module(input_dim = train_data_dict[input_modality + '_train_mtx'].shape[1],
                                            input_batch_num = train_data_dict[input_modality + '_train_cov'].shape[1] if self.cov is not None else 0,
                                            hidden_layer = self.hidden_layer,
                                            activation = self.activation,
                                            layernorm = self.layernorm,
                                            batchnorm = self.batchnorm,
                                            dropout_rate = self.dropout_rate,
                                            add_linear_layer = self.add_linear_layer,
                                            infer_library_size = self.infer_library_size_rna if output_modality == 'Gene Expression' else self.infer_library_size_atac)
                print('Set up encoder: from', input_modality, '"', self.modalities[input_modality],'distribution "', 'to', output_modality, '"', self.modalities[output_modality], 'distribution"')
                if self.modalities[output_modality] == 'gau':
                    output_distribution = self.modalities[output_modality]
                    # Initialize encoder and decoder
                    set_seed(self.SEED)
                    encoder = self.input_module(input_dim = train_data_dict[input_modality + '_train_mtx'].shape[1],
                                                input_batch_num = train_data_dict[input_modality + '_train_cov'].shape[1] if self.cov is not None else 0,
                                                hidden_layer = self.hidden_layer,
                                                activation = self.activation,
                                                layernorm = self.layernorm,
                                                batchnorm = self.batchnorm,
                                                dropout_rate = self.dropout_rate,
                                                add_linear_layer = self.add_linear_layer,
                                                infer_library_size = False) # set to False for now
                    set_seed(self.SEED)
                    decoder = self.output_module[output_distribution](output_dim = train_data_dict[output_modality + '_train_mtx'].shape[1],
                                                                        output_batch_num = train_data_dict[output_modality + '_train_cov'].shape[1] if self.cov is not None else 0,
                                                                        hidden_layer = self.hidden_layer,
                                                                        infer_library_size = False,
                                                                        feature_factor = False)
                    print('Set up decoder: from', input_modality, '"', self.modalities[input_modality],'distribution "', 'to', output_modality, '"', self.modalities[output_modality], 'distribution"')
                    trainLib = valLib = testLib = None
                else:
                    output_distribution = self.modalities[output_modality]
                    if output_distribution in ['zinb', 'nb']:
                        set_seed(self.SEED)
                        decoder = self.output_module[output_distribution](output_dim = train_data_dict[output_modality + '_train_mtx'].shape[1],
                                                                        output_batch_num = train_data_dict[output_modality + '_train_cov'].shape[1] if self.cov is not None else 0,
                                                                        hidden_layer = self.hidden_layer,
                                                                        infer_library_size = self.infer_library_size_rna,
                                                                        sample_factor = self.sample_factor_rna,
                                                                        feature_factor = self.feature_factor_rna,
                                                                        zero_inflated = self.zero_inflated,
                                                                        dispersion = self.dispersion)
                        print('Set up decoder: from', input_modality, '"', self.modalities[input_modality],'distribution "', 'to', output_modality, '"', self.modalities[output_modality], 'distribution"')
                        # Set up library size if not inferred
                        if self.sample_factor_rna is True:
                            trainLib = train_data_dict[output_modality + '_train_lib']
                            valLib = val_data_dict[output_modality + '_val_lib'].to(self.device)
                            testLib = test_data_dict[output_modality + '_test_lib'].to(self.device) if test_data_dict is not None else None
                        else:
                            trainLib = valLib = testLib = None
                    elif output_distribution == 'ber':
                        set_seed(self.SEED)
                        decoder = self.output_module[output_distribution](output_dim = train_data_dict[output_modality + '_train_mtx'].shape[1],
                                                                        output_batch_num = train_data_dict[output_modality + '_train_cov'].shape[1] if self.cov is not None else 0,
                                                                        hidden_layer = self.hidden_layer,
                                                                        infer_library_size = self.infer_library_size_atac,
                                                                        sample_factor = self.sample_factor_atac,
                                                                        feature_factor = self.feature_factor_atac)
                        print('Set up decoder: from', input_modality, '"', self.modalities[input_modality],'distribution "', 'to', output_modality, '"', self.modalities[output_modality], 'distribution"')
                        # Set up library size if not inferred
                        if self.sample_factor_atac is True:
                            trainLib = train_data_dict[output_modality + '_train_lib']
                            valLib = val_data_dict[output_modality + '_val_lib'].to(self.device)
                            testLib = test_data_dict[output_modality + '_test_lib'].to(self.device) if test_data_dict is not None else None
                        else:
                            trainLib = valLib = testLib = None
                    else:
                        raise ValueError('scPair distribution should be one of the following: zinb, nb, ber, gau')
            else:
                raise ValueError('scPair distribution should be one of the following: zinb, nb, ber, gau')
            # Apply weight initialization and move to device
            encoder.apply(init_weights).to(self.device)
            decoder.apply(init_weights).to(self.device)
            # Set up optimizer and scheduler
            decay_param_encoder, nodecay_param_encoder, _, _ = add_weight_decay(encoder, output_layer=['output'])
            decay_param_decoder, nodecay_param_decoder, _, _ = add_weight_decay(decoder, output_layer=['output'])
            optimizer = torch.optim.AdamW([{'params': decay_param_encoder, 'weight_decay': self.L2_lambda, 'lr': self.learning_rate_prediction},
                                        {'params': decay_param_decoder, 'weight_decay': self.L2_lambda, 'lr': self.learning_rate_prediction},
                                        {'params': nodecay_param_encoder, 'weight_decay': 0, 'lr': self.learning_rate_prediction}, 
                                        {'params': nodecay_param_decoder, 'weight_decay': 0, 'lr': self.learning_rate_prediction}])
            scheduler = self.get_scheduler(optimizer)
            # Prepare training data with memory optimization
            if self.memory_efficient_loading:
                # Use lazy loading to avoid moving all data to GPU at once
                train_data = LazyGPUDataset(trainData, *(trainBatch,) if self.cov is not None else (), *(trainLib,) if trainLib is not None else (), trainLabel, device=self.device)
            else:
                # Traditional approach - move all data to GPU
                train_data = TensorDataset(trainData.to(self.device), *(trainBatch.to(self.device),) if self.cov is not None else (), *(trainLib.to(self.device),) if trainLib is not None else (), trainLabel.to(self.device))
            
            from torch.utils.data import DataLoader, Dataset
            set_seed(self.SEED)
            DataLoader_train = DataLoader(train_data, 
                                        batch_size=self.batch_size, 
                                        shuffle=True,
                                        pin_memory=self.pin_memory and not self.memory_efficient_loading,
                                        num_workers=0)  # Set to 0 to avoid multiprocessing issues
            # Set up early stopping if enabled
            early_stopping_patience, epochs_no_improve, early_stop, min_val_loss = (self.early_stopping_patience, 0, False, np.inf) if self.early_stopping_activation else (None, None, None, None)
            print('Enabling early stopping with patience of', early_stopping_patience) if self.early_stopping_activation else None
            # Print training start message
            text = '\033[95m' + 'Start training feature predictor: {} '.format(prediction_order) + '\033[0m'            
            print('#' * (len(text)-5))
            print('# ' + text + ' #')
            print('#' * (len(text)-5))
            pbar = tqdm.tqdm(range(self.max_epochs))
            for epoch in pbar:
                training_mode(encoder, decoder)
                for idx, data in enumerate(DataLoader_train):
                    # Move data to device safely
                    if self.memory_efficient_loading:
                        data = tuple(move_to_device_safe(tensor, self.device) for tensor in data)
                    
                    if trainLib is not None:
                        if self.cov is not None:
                            x, b, lib, y = data
                            if self.use_mixed_precision and self.scaler is not None:
                                with autocast():
                                    train_net(x, y, lib, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=True, input_batch=b, output_batch=b, scaler=self.scaler)
                            else:
                                train_net(x, y, lib, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=True, input_batch=b, output_batch=b)
                        else:
                            x, lib, y = data
                            if self.use_mixed_precision and self.scaler is not None:
                                with autocast():
                                    train_net(x, y, lib, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=False, scaler=self.scaler)
                            else:
                                train_net(x, y, lib, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=False)
                    else:
                        if self.cov is not None:
                            x, b, y = data
                            if self.use_mixed_precision and self.scaler is not None:
                                with autocast():
                                    train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=True, input_batch=b, output_batch=b, scaler=self.scaler)
                            else:
                                train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=True, input_batch=b, output_batch=b)
                        else:
                            x, y = data
                            if self.use_mixed_precision and self.scaler is not None:
                                with autocast():
                                    train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=False, scaler=self.scaler)
                            else:
                                train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=self.modalities[output_modality], add_cov=False)
                    
                    # Clear memory periodically
                    if idx % 10 == 0:
                        clear_gpu_memory()
                # Evaluate the model
                evaluating_mode(encoder, decoder)
                with torch.no_grad():
                    # Move validation data to device safely
                    valData_safe = move_to_device_safe(valData, self.device)
                    valLabel_safe = move_to_device_safe(valLabel, self.device)
                    valLib_safe = move_to_device_safe(valLib, self.device) if valLib is not None else None
                    valBatch_safe = move_to_device_safe(valBatch, self.device) if valBatch is not None else None
                    
                    val_total_loss = eval_net(valData_safe, valLabel_safe, valLib_safe, encoder, decoder, likelihood_type=self.modalities[output_modality], add_cov=self.cov is not None, input_batch=valBatch_safe, output_batch=valBatch_safe)
                    pbar.set_postfix({"Epoch": epoch+1, "validation Loss": val_total_loss.item()})
                    
                    # Clear validation tensors from memory
                    del valData_safe, valLabel_safe, valLib_safe, valBatch_safe
                # Update the scheduler
                if optimizer.param_groups[0]['lr'] > 1e-7 and self.weight_decay in ['ExponentialLR', 'StepLR', 'ReduceLROnPlateau']:
                    scheduler.step()
                # Check for early stopping
                if val_total_loss.item() < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = val_total_loss.item()
                    encoder_ckpt = copy.deepcopy(encoder)
                    decoder_ckpt = copy.deepcopy(decoder)
                else:
                    epochs_no_improve += 1
                    pbar.set_postfix({'Message': 'Early stopping triggered!', 'Patience Left': early_stopping_patience - epochs_no_improve})
                if epoch > 1 and epochs_no_improve == early_stopping_patience:
                    print("Early Stopped!")
                    early_stop = True
                    break
                else:
                    continue
            # Save the trained models
            encoder_dict[input_modality + ' to ' + output_modality] = copy.deepcopy(encoder_ckpt.eval())
            decoder_dict[input_modality + ' to ' + output_modality] = copy.deepcopy(decoder_ckpt.eval())
            if self.save_model is True:
                if self.save_path is None:
                    self.save_path = os.getcwd()
                torch.save(encoder_dict[input_modality + ' to ' + output_modality], self.save_path + '/encoder_' + input_modality + '_to_' + output_modality + '.pt')
                torch.save(decoder_dict[input_modality + ' to ' + output_modality], self.save_path + '/decoder_' + input_modality + '_to_' + output_modality + '.pt')
            
            # Clear memory after training each modality
            clear_gpu_memory()
            print(f"Completed training for {input_modality} to {output_modality}")
        return encoder_dict, decoder_dict
    def train_bidirectional_mapping(self):
        embedding_dim = self.hidden_layer[-1]
        mapping_dict = {}
        for input_modality in self.modality_names:
            output_modality = [mn for mn in self.modality_names if mn != input_modality][0]
            print('mapping direction:', input_modality, 'to', output_modality)
            # Prepare training data with memory optimization
            input_embedding_train = self.embeddings[input_modality + '_train']
            output_embedding_train = self.embeddings[output_modality + '_train']
            input_embedding_val = self.embeddings[input_modality + '_val']
            output_embedding_val = self.embeddings[output_modality + '_val']
            
            if self.memory_efficient_loading:
                trainData = LazyGPUDataset(input_embedding_train, output_embedding_train, device=self.device)
            else:
                trainData = TensorDataset(input_embedding_train.to(self.device), output_embedding_train.to(self.device))
            
            from torch.utils.data import DataLoader, Dataset
            set_seed(self.SEED)
            DataLoader_train = DataLoader(trainData, 
                                        batch_size=self.batch_size, 
                                        shuffle=True,
                                        pin_memory=self.pin_memory and not self.memory_efficient_loading,
                                        num_workers=0)
            # Initialize mapping network
            mapping_network = self.mapping_module(origin_module_dim = embedding_dim,
                                                target_module_dim = embedding_dim,
                                                translational_hidden_nodes = self.mapping_hidden_nodes,
                                                non_neg = self.mapping_non_neg,
                                                activation = self.mapping_activation,
                                                layernorm = self.mapping_layernorm,
                                                batchnorm = self.mapping_batchnorm,
                                                dropout_rate = self.mapping_dropout_rate)
            mapping_network.apply(init_weights).to(self.device)
            # Set up optimizer and scheduler
            decay_param, nodecay_param, _, _ = add_weight_decay(mapping_network, output_layer="bias")
            optimizer = torch.optim.AdamW([{'params': decay_param, 'weight_decay': self.L2_lambda, 'lr': self.mapping_learning_rate},
                                        {'params': nodecay_param, 'weight_decay':0, 'lr': self.mapping_learning_rate}])
            scheduler = self.get_scheduler(optimizer)
            # Set up early stopping if enabled
            early_stopping_patience, epochs_no_improve, early_stop, min_val_loss = (self.early_stopping_patience, 0, False, np.inf) if self.early_stopping_activation else (None, None, None, None)
            print('Enabling early stopping with patience of', early_stopping_patience) if self.early_stopping_activation else None
            # Print training start message
            text = '\033[95m' + 'Start mapping {} embeddings'.format(input_modality + ' to ' + output_modality) + '\033[0m'            
            print('#' * (len(text)-5))
            print('# ' + text + ' #')
            print('#' * (len(text)-5))
            # Start training
            pbar = tqdm.tqdm(range(self.max_epochs))
            for epoch in pbar:
                mapping_network.train();
                for idx, (x, y) in enumerate(DataLoader_train):
                    # Move data to device safely
                    if self.memory_efficient_loading:
                        x = move_to_device_safe(x, self.device)
                        y = move_to_device_safe(y, self.device)
                    
                    if self.use_mixed_precision and self.scaler is not None:
                        with autocast():
                            pred_emb = mapping_network(x)
                            training_loss = scaled_loss(y, pred_emb)
                        optimizer.zero_grad()
                        self.scaler.scale(training_loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        pred_emb = mapping_network(x)
                        training_loss = scaled_loss(y, pred_emb)
                        optimizer.zero_grad()
                        training_loss.backward()
                        optimizer.step()
                    
                    # Clear memory periodically
                    if idx % 10 == 0:
                        clear_gpu_memory()
                        
                mapping_network.eval();
                with torch.no_grad():
                    val_x = move_to_device_safe(input_embedding_val, self.device)
                    val_y = move_to_device_safe(output_embedding_val, self.device)
                    val_pred_emb = mapping_network(val_x)
                    val_total_loss = scaled_loss(val_y, val_pred_emb)
                    pbar.set_postfix({"Epoch": epoch+1, "validation Loss": val_total_loss.item()})
                    
                    # Clear validation tensors
                    del val_x, val_y
                if optimizer.param_groups[0]['lr'] > 1e-7 and self.weight_decay in ['ExponentialLR', 'StepLR', 'ReduceLROnPlateau']:
                    scheduler.step()
                if val_total_loss.item() < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = val_total_loss.item()
                    mapping_network_ckpt = copy.deepcopy(mapping_network)
                else:
                    epochs_no_improve += 1
                    pbar.set_postfix({'Message': 'Early stopping triggered!', 'Patience Left': early_stopping_patience - epochs_no_improve})
                if epoch > 1 and epochs_no_improve == early_stopping_patience:
                    print("Early Stopped!")
                    early_stop = True
                    break
                else:
                    continue
            mapping_dict[input_modality + '_to_' + output_modality] = copy.deepcopy(mapping_network_ckpt)
            # Save the trained model
            if self.save_model is True:
                if self.save_path is None:
                    self.save_path = os.getcwd()
                torch.save(mapping_dict[input_modality + '_to_' + output_modality], self.save_path + '/mapping_' + input_modality + '_to_' + output_modality + '.pt')
            
            # Clear memory after training each mapping
            clear_gpu_memory()
            print(f"Completed mapping training for {input_modality} to {output_modality}")
        return mapping_dict
    def augment(self, unimodal_scobj, unimodal_modalities, unimodal_cov = None):
        # Set up unimodal data
        self.unimodal_scobj, self.unimodal_modalities, self.unimodal_cov = unimodal_scobj, unimodal_modalities, unimodal_cov
        self.unimodal_modality_names = list(unimodal_modalities.keys())
        self.unimodal_modality_distributions = list(unimodal_modalities.values())
        # Check if unimodal data matches multimodal data's counterpart
        if not set(self.unimodal_modality_names).issubset(self.modalities.keys()):
            print('Paired multimodal data:', self.modalities)
            print('Unimodal data:', self.unimodal_modalities)
            raise ValueError('Unimodal data should have the same modality as one of the multimodal data')
        # Handle covariate matrix
        if self.unimodal_cov != self.cov:
            if self.unimodal_cov is None and self.cov is not None:
                self.unimodal_cov = self.cov
                self.unimodal_cov_dummy = pd.DataFrame(data=np.zeros((self.unimodal_scobj.shape[0], self.cov_dummy.shape[1])), columns=self.cov_dummy.columns, index=self.unimodal_scobj.obs.index)
                print('Using multimodal cov for unimodal data, and set unimodal cov to all zeros during scPair_augment training')
            elif self.unimodal_cov is not None and self.cov is None:
                raise ValueError('Unimodal data have covariate matrix while multimodal data do not')
                # print('Using unimodal cov = None for unimodal data during inference, and set multimodal cov to all zeros to match the dimension during scPair_augment training')
                # self.unimodal_cov_dummy = pd.get_dummies(self.unimodal_scobj.obs[self.unimodal_cov])
                # self.cov_dummy_ = pd.DataFrame(data=np.zeros((self.scobj.shape[0], self.unimodal_cov_dummy.shape[1])), columns=self.unimodal_cov_dummy.columns, index=self.scobj.obs.index)
            else:
                self.unimodal_cov, self.unimodal_cov_dummy = None, None
        # Prepare data input of the augment encoder
        unimodal_train_input = torch.FloatTensor(self.unimodal_scobj.X.toarray()) if scipy.sparse.issparse(self.unimodal_scobj.X) else torch.FloatTensor(self.unimodal_scobj.X)
        unimodal_train_batch = torch.FloatTensor(self.unimodal_cov_dummy.to_numpy()) if self.unimodal_cov is not None else None
        bimodal_train_input = self.data_loader_dict_train[self.unimodal_modality_names[0] + '_train_mtx']
        bimodal_train_batch = self.data_loader_dict_train[self.unimodal_modality_names[0] + '_train_cov'] if self.cov is not None else None
        bimodal_val_input = self.data_loader_dict_val[self.unimodal_modality_names[0] + '_val_mtx']
        bimodal_val_batch = self.data_loader_dict_val[self.unimodal_modality_names[0] + '_val_cov'] if self.cov is not None else None
        bimodal_test_input = self.data_loader_dict_test[self.unimodal_modality_names[0] + '_test_mtx'] if self.data_loader_dict_test is not None else None
        bimodal_test_batch = self.data_loader_dict_test[self.unimodal_modality_names[0] + '_test_cov'] if self.cov is not None and self.data_loader_dict_test is not None else None
        # Prepare original pre-trained encoder
        encoder_to_be_updated = copy.deepcopy(self.encoder_dict[self.unimodal_modality_names[0] + ' to ' + [mn for mn in self.modalities.keys() if mn != self.unimodal_modality_names[0]][0]])
        encoder_to_be_updated.to(self.device)
        encoder_to_be_updated.eval()
        # Function to prepare labels/outputs of the augment encoder
        def prepare_labels(input_data, batch_data):
            input_data = input_data.to(self.device)
            batch_data = batch_data.to(self.device) if batch_data is not None else None
            return encoder_to_be_updated(input_data, batch_data)[0].detach()
        # Prepare data output of the augment encoder
        unimodal_train_label = prepare_labels(unimodal_train_input, unimodal_train_batch)
        bimodal_train_label = prepare_labels(bimodal_train_input, bimodal_train_batch)
        bimodal_val_label = prepare_labels(bimodal_val_input, bimodal_val_batch)
        bimodal_test_label = prepare_labels(bimodal_test_input, bimodal_test_batch) if bimodal_test_input is not None else None
        # Prepare training data
        trainData = TensorDataset(unimodal_train_input.to(self.device), *(unimodal_train_batch.to(self.device),) if unimodal_train_batch is not None else (), unimodal_train_label.to(self.device))
        # bimodal_val_input/batch/label here will be used for early stopping: concat bimodal train and val data
        unimodal_val_input = torch.cat((bimodal_train_input, bimodal_val_input), 0).to(self.device)
        unimodal_val_batch = torch.cat((bimodal_train_batch, bimodal_val_batch), 0).to(self.device) if self.unimodal_cov is not None else None
        unimodal_val_label = torch.cat((bimodal_train_label, bimodal_val_label), 0).to(self.device)
        # Data loader for training
        from torch.utils.data import DataLoader, Dataset
        set_seed(self.SEED)
        DataLoader_train = DataLoader(trainData, batch_size=self.batch_size, shuffle=True)
        # Initialize optimizer
        optimizer_unimodal = torch.optim.AdamW(encoder_to_be_updated.parameters(), lr=self.learning_rate_prediction, weight_decay=self.L2_lambda)
        # Set up early stopping if enabled
        early_stopping_patience, epochs_no_improve, early_stop, min_val_loss = (self.early_stopping_patience, 0, False, np.inf) if self.early_stopping_activation else (None, None, None, None)
        print('Enabling early stopping with patience of', early_stopping_patience) if self.early_stopping_activation else None
        augment_encoder_dict = {}
        pbar = tqdm.tqdm(range(self.max_epochs))
        for epoch in pbar:
            encoder_to_be_updated.train()
            for idx, data in enumerate(DataLoader_train):
                x, *b, y = data
                b = b[0] if b else None
                pred = encoder_to_be_updated(x, b)[0]
                loss = scaled_loss(y, pred)
                optimizer_unimodal.zero_grad()
                loss.backward()
                optimizer_unimodal.step()
            encoder_to_be_updated.eval()
            with torch.no_grad():
                val_total_loss = scaled_loss(unimodal_val_label, encoder_to_be_updated(unimodal_val_input, unimodal_val_batch if self.unimodal_cov is not None else None)[0])
                pbar.set_postfix({"Epoch": epoch+1, "validation Loss": val_total_loss.item()})
            # Check for early stopping
            if val_total_loss.item() < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_total_loss.item()
                self.augment_encoder_ckpt = copy.deepcopy(encoder_to_be_updated)
            else:
                epochs_no_improve += 1
            if epoch > 1 and epochs_no_improve == early_stopping_patience:
                print("Early Stopped!")
                early_stop = True
                break
            else:
                continue
        self.augment_encoder_ckpt.eval()
        # Save the trained model: augmented encoder
        if self.save_model is True:
            if self.save_path is None:
                self.save_path = os.getcwd()
            torch.save(self.augment_encoder_ckpt, self.save_path + '/augment_encoder_' + self.unimodal_modality_names[0] + '_to_' + [mn for mn in self.modalities.keys() if mn != self.unimodal_modality_names[0]][0] + '.pt')
        augment_encoder_dict['scPair_augment: ' + self.unimodal_modality_names[0] + ' to ' + [mn for mn in self.modalities.keys() if mn != self.unimodal_modality_names[0]][0]] = copy.deepcopy(self.augment_encoder_ckpt)
        self.augment_encoder_dict = augment_encoder_dict
        # Initialize dictionary to store updated embeddings
        augment_emb = {}
        # Define a helper function to generate predictions
        def generate_predictions(input_data, batch_data):
            return self.augment_encoder_ckpt(input_data.to(self.device), batch_data.to(self.device) if self.cov is not None else None)[0].cpu().detach().numpy()
        # Generate updated embeddings
        for split in ['unimodal_train', 'bimodal_train', 'bimodal_val', 'bimodal_test']:
            if split == 'bimodal_test' and bimodal_test_input is None:
                continue
            input_data = locals()[f'{split}_input']
            batch_data = locals()[f'{split}_batch']
            predictions = generate_predictions(input_data, batch_data)
            augment_emb[self.unimodal_modality_names[0] + f'_{split}'] = predictions
        return augment_encoder_dict, augment_emb