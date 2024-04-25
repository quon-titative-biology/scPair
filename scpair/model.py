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
                 weight_decay = None, # 'ExponentialLR',
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
                 mapping_dropout_rate = 0):
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
        data = self.scobj.copy()  # data: scanpy AnnData object
        cov = self.cov
        if cov is not None:
            cov_dummy = pd.get_dummies(data.obs[cov]).astype(int)
            data.obs = pd.concat([data.obs, cov_dummy], axis=1)
        else:
            cov_dummy = None
        self.cov_dummy = cov_dummy
        data_train = data[data.obs['scPair_split'] == 'train']
        data_val = data[data.obs['scPair_split'] == 'val']
        data_test = data[data.obs['scPair_split'] == 'test'] if np.sum(data.obs['scPair_split'] == 'test') > 0 else None
        self.train_metadata = data_train.obs.copy()
        self.val_metadata = data_val.obs.copy()
        self.test_metadata = data_test.obs.copy() if data_test is not None else None
        if scipy.sparse.issparse(data.X):
            print('Converting sparse matrix to dense matrix...')
            data.X = data.X.toarray()
        print('Initializing data loaders...')
        data_loader_dict = {}
        modality_names = self.modality_names
        for modality in modality_names:
            print('processing modality:', modality)
            data_loader_dict[modality + '_train_mtx'] = torch.FloatTensor(data_train[:, data.var['modality'] == modality].X)
            data_loader_dict[modality + '_train_cov'] = torch.FloatTensor(cov_dummy.loc[data_train.obs.index].values) if cov is not None else None
            data_loader_dict[modality + '_val_mtx'] = torch.FloatTensor(data_val[:, data.var['modality'] == modality].X)
            data_loader_dict[modality + '_val_cov'] = torch.FloatTensor(cov_dummy.loc[data_val.obs.index].values) if cov is not None else None
            if data_test is not None:
                data_loader_dict[modality + '_test_mtx'] = torch.FloatTensor(data_test[:, data.var['modality'] == modality].X)
                data_loader_dict[modality + '_test_cov'] = torch.FloatTensor(cov_dummy.loc[data_test.obs.index].values) if cov is not None else None             
            if modality == 'Gene Expression':
                data_loader_dict[modality + '_train_lib'] = data_loader_dict['Gene Expression_train_mtx'].sum(1).reshape(-1,1)
                data_loader_dict[modality + '_val_lib'] = data_loader_dict['Gene Expression_val_mtx'].sum(1).reshape(-1,1)
                data_loader_dict[modality + '_test_lib'] = data_loader_dict['Gene Expression_test_mtx'].sum(1).reshape(-1,1) if data_test is not None else None
            elif modality == 'Peaks':
                data_loader_dict[modality + '_train_lib'] = data_loader_dict['Peaks_train_mtx'].sum(1).reshape(-1,1)
                data_loader_dict[modality + '_val_lib'] = data_loader_dict['Peaks_val_mtx'].sum(1).reshape(-1,1)
                data_loader_dict[modality + '_test_lib'] = data_loader_dict['Peaks_test_mtx'].sum(1).reshape(-1,1) if data_test is not None else None
        data_loader_keys = list(data_loader_dict.keys())
        self.data_loader_keys = data_loader_keys
        train_keys = [key for key in data_loader_keys if 'train' in key]
        val_keys = [key for key in data_loader_keys if 'val' in key]
        # train_lib_keys = [key for key in train_keys if 'lib' in key]
        # val_lib_keys = [key for key in val_keys if 'lib' in key]
        data_loader_dict_train = {key: data_loader_dict[key] for key in train_keys}
        data_loader_dict_val = {key: data_loader_dict[key] for key in val_keys}
        self.data_loader_dict_train = data_loader_dict_train
        self.data_loader_dict_val = data_loader_dict_val
        # train_dataset = torch.utils.data.ConcatDataset([data_loader_dict_train[key].to(device) for key in data_loader_dict_train.keys()])
        # val_dataset = torch.utils.data.ConcatDataset([data_loader_dict_val[key].to(device) for key in data_loader_dict_val.keys()])
        test_keys = [key for key in data_loader_keys if 'test' in key] if data_test is not None else None
        data_loader_dict_test = {key: data_loader_dict[key] for key in test_keys} if data_test is not None else None
        self.data_loader_dict_test = data_loader_dict_test
        # test_lib_keys = [key for key in test_keys if 'lib' in key] if data_test is not None else None
        # test_dataset = torch.utils.data.ConcatDataset([data_loader_dict_test[key].to(device) for key in data_loader_dict_test.keys()]) if data_test is not None else None
        return data_loader_dict_train, data_loader_dict_val, data_loader_dict_test
    def run(self):
        data_loader_dict_train, data_loader_dict_val, data_loader_dict_test = self.data_loader_builder() # self, data = self.scobj, modality_names = self.modality_names, cov = self.cov)
        encoder_dict, decoder_dict = self.train_predicting_networks(data_loader_dict_train, data_loader_dict_val, data_loader_dict_test)
        self.encoder_dict = encoder_dict.copy()
        self.decoder_dict = decoder_dict.copy()
        low_dim_embeddings, _ = self.reference_embeddings()
        mapping_dict = self.train_bidirectional_mapping()
        self.mapping_dict = mapping_dict.copy()
        low_dim_embeddings_mapped, _ = self.mapped_embeddings()
        return encoder_dict, decoder_dict, mapping_dict, low_dim_embeddings, low_dim_embeddings_mapped
    def train_predicting_networks(self, train_data_dict, val_data_dict, test_data_dict):
        save_model = self.save_model
        save_path = self.save_path
        modalities = self.modalities
        device = self.device
        SEED = self.SEED
        cov = self.cov
        hidden_layer = self.hidden_layer
        dropout_rate = self.dropout_rate
        batchnorm = self.batchnorm
        layernorm = self.layernorm
        learning_rate_prediction = self.learning_rate_prediction
        batch_size = self.batch_size
        L2_lambda = self.L2_lambda
        max_epochs = self.max_epochs
        activation = self.activation
        early_stopping_activation = self.early_stopping_activation
        weight_decay = self.weight_decay
        add_linear_layer = self.add_linear_layer
        sample_factor_rna = self.sample_factor_rna
        feature_factor_rna = self.feature_factor_rna
        sample_factor_atac = self.sample_factor_atac
        feature_factor_atac = self.feature_factor_atac
        zero_inflated = self.zero_inflated
        dispersion = self.dispersion
        infer_library_size_rna = self.infer_library_size_rna
        infer_library_size_atac = self.infer_library_size_atac
        encoder_dict = {}
        decoder_dict = {}
        for input_modality in modalities.keys():
            output_modality = [modality for modality in modalities.keys() if modality != input_modality][0]
            prediction_order = {'input':input_modality, 'output':output_modality}
            print('Set up predicting networks:', prediction_order)
            # dataset split
            trainData = train_data_dict[input_modality + '_train_mtx']
            trainLabel = train_data_dict[output_modality + '_train_mtx']
            valData = val_data_dict[input_modality + '_val_mtx'].to(device)
            valLabel = val_data_dict[output_modality + '_val_mtx'].to(device)
            if cov is not None:
                trainBatch = train_data_dict[input_modality + '_train_cov']
                valBatch = val_data_dict[input_modality + '_val_cov'].to(device)
            else:
                trainBatch = None
                valBatch = None
            if modalities[output_modality] == 'gau':
                output_distribution = modalities[output_modality]
                set_seed(SEED)
                encoder = self.input_module(input_dim = train_data_dict[input_modality + '_train_mtx'].shape[1],
                                            input_batch_num = train_data_dict[input_modality + '_train_cov'].shape[1] if cov is not None else 0,
                                            hidden_layer = hidden_layer,
                                            activation = activation,
                                            layernorm = layernorm,
                                            batchnorm = batchnorm,
                                            dropout_rate = dropout_rate,
                                            add_linear_layer = add_linear_layer,
                                            infer_library_size = False) # set to False for now
                set_seed(SEED)
                decoder = self.output_module[output_distribution](output_dim = train_data_dict[output_modality + '_train_mtx'].shape[1],
                                                                    output_batch_num = train_data_dict[output_modality + '_train_cov'].shape[1] if cov is not None else 0,
                                                                    hidden_layer = hidden_layer,
                                                                    infer_library_size = False,
                                                                    feature_factor = False)
                encoder.apply(init_weights)
                encoder.to(device)
                decoder.apply(init_weights)
                decoder.to(device)
                output_layers = ['output']
                decay_param_encoder, nodecay_param_encoder, decay_name_encoder, nodecay_name_encoder = add_weight_decay(encoder, output_layer=output_layers)
                decay_param_decoder, nodecay_param_decoder, decay_name_decoder, nodecay_name_decoder = add_weight_decay(decoder, output_layer=output_layers)
                optimizer = torch.optim.AdamW([{'params': decay_param_encoder, 'weight_decay':L2_lambda, 'lr': learning_rate_prediction},
                                            {'params': decay_param_decoder, 'weight_decay':L2_lambda, 'lr': learning_rate_prediction},
                                            {'params': nodecay_param_encoder, 'weight_decay':0, 'lr': learning_rate_prediction}, 
                                            {'params': nodecay_param_decoder, 'weight_decay':0, 'lr': learning_rate_prediction}])
                if weight_decay == 'ExponentialLR':
                    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1)
                elif weight_decay == 'StepLR':
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.1, last_epoch=-1)
                elif weight_decay == 'ReduceLROnPlateau':
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
                else:
                    scheduler = None
                    Warning('weight_decay should be one of the following: ExponentialLR, StepLR, ReduceLROnPlateau. No weight decay applied.')
                train_data = TensorDataset(trainData.to(device), trainBatch.to(device), trainLabel.to(device)) if cov is not None else TensorDataset(trainData.to(device), trainLabel.to(device))
                from torch.utils.data import DataLoader, Dataset
                set_seed(SEED)
                DataLoader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                if early_stopping_activation is True:
                    early_stopping_patience = self.early_stopping_patience
                    epochs_no_improve = 0
                    early_stop = False
                    min_val_loss = np.inf
                    print('Enabling early stopping with patience of', early_stopping_patience)
                text = '\033[95m' + 'Start training feature predictor: {} '.format(prediction_order) + '\033[0m'            
                print('#' * (len(text)-5))
                print('# ' + text + ' #')
                print('#' * (len(text)-5))
                pbar = tqdm.tqdm(range(max_epochs))
                for epoch in pbar:
                    training_mode(encoder, decoder)
                    if cov is not None:
                        for idx, (x, b, y) in enumerate(DataLoader_train):
                            train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=modalities[output_modality], add_cov=True, input_batch=b, output_batch=b)
                    else:
                        for idx, (x, y) in enumerate(DataLoader_train):
                            train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=modalities[output_modality])
                    evaluating_mode(encoder, decoder)
                    with torch.no_grad():
                        if cov is not None:
                            val_total_loss = eval_net(valData, valLabel, None, encoder, decoder, likelihood_type=modalities[output_modality], add_cov=True, input_batch=valBatch, output_batch=valBatch)
                        else:
                            val_total_loss = eval_net(valData, valLabel, None, encoder, decoder, likelihood_type=modalities[output_modality])
                        pbar.set_postfix({"Epoch": epoch+1, "validation Loss": val_total_loss.item()})
                    if optimizer.param_groups[0]['lr'] > 1e-7:
                        # print(optimizer.param_groups[0]['lr'])
                        if weight_decay == 'ExponentialLR':
                            scheduler.step()
                        elif weight_decay == 'StepLR':
                            scheduler.step()
                        elif weight_decay == 'ReduceLROnPlateau':
                            scheduler.step()
                        else:
                            pass
                    if val_total_loss.item() < min_val_loss:
                        epochs_no_improve = 0
                        min_val_loss = val_total_loss.item()
                        encoder_ckpt = copy.deepcopy(encoder)
                        decoder_ckpt = copy.deepcopy(decoder)
                    else:
                        epochs_no_improve += 1
                        pbar.set_postfix({'Message': 'Early stopping triggered!', 'Patience Left': early_stopping_patience - epochs_no_improve})
                        # tqdm.tqdm.write("Early stopping triggered!", early_stopping_patience - epochs_no_improve)
                    if epoch > 1 and epochs_no_improve == early_stopping_patience:
                        print("Early Stopped!")
                        early_stop = True
                        break
                    else:
                        continue
                encoder_dict[input_modality + '_to_' + output_modality] = copy.deepcopy(encoder_ckpt)
                decoder_dict[input_modality + '_to_' + output_modality] = copy.deepcopy(decoder_ckpt)
                if save_model is True:
                    if save_path is None:
                        save_path = os.getcwd()
                    torch.save(encoder_dict[input_modality + '_to_' + output_modality].state_dict(), save_path + '/encoder_' + input_modality + '_to_' + output_modality + '.pt')
                    torch.save(decoder_dict[input_modality + '_to_' + output_modality].state_dict(), save_path + '/decoder_' + input_modality + '_to_' + output_modality + '.pt')
            elif modalities[input_modality] == 'zinb' or modalities[input_modality] == 'nb' or modalities[input_modality] == 'ber' or modalities[input_modality] == 'gau':
                if modalities[input_modality] != 'gau':
                    set_seed(SEED)
                    encoder = self.input_module(input_dim = train_data_dict[input_modality + '_train_mtx'].shape[1],
                                                input_batch_num = train_data_dict[input_modality + '_train_cov'].shape[1] if cov is not None else 0,
                                                hidden_layer = hidden_layer,
                                                activation = activation,
                                                layernorm = layernorm,
                                                batchnorm = batchnorm,
                                                dropout_rate = dropout_rate,
                                                add_linear_layer = add_linear_layer,
                                                infer_library_size = infer_library_size_rna if output_modality == 'Gene Expression' else infer_library_size_atac)
                    print('Set up encoder from', input_modality, '"', modalities[input_modality],'distribution "', 'to', output_modality)
                if modalities[input_modality] == 'gau':
                    set_seed(SEED)
                    encoder = self.input_module(input_dim = train_data_dict[input_modality + '_train_mtx'].shape[1],
                                                input_batch_num = train_data_dict[input_modality + '_train_cov'].shape[1] if cov is not None else 0,
                                                hidden_layer = hidden_layer,
                                                activation = activation,
                                                layernorm = layernorm,
                                                batchnorm = batchnorm,
                                                dropout_rate = dropout_rate,
                                                add_linear_layer = add_linear_layer,
                                                infer_library_size = infer_library_size_rna if output_modality == 'Gene Expression' else infer_library_size_atac)
                    print('Set up encoder from', input_modality, '[Gaussian distribution]', 'to', output_modality)
                if modalities[output_modality] == 'ber':
                    output_distribution = modalities[output_modality]
                    set_seed(SEED)
                    decoder = self.output_module[output_distribution](output_dim = train_data_dict[output_modality + '_train_mtx'].shape[1],
                                                                    output_batch_num = train_data_dict[output_modality + '_train_cov'].shape[1] if cov is not None else 0,
                                                                    hidden_layer = hidden_layer,
                                                                    infer_library_size = infer_library_size_atac,
                                                                    sample_factor = sample_factor_atac,
                                                                    feature_factor = feature_factor_atac)
                    if infer_library_size_atac is True:
                        trainLib = None
                        valLib = None
                        testLib = None
                    else:
                        trainLib = train_data_dict[output_modality + '_train_lib']
                        valLib = val_data_dict[output_modality + '_val_lib'].to(device)
                        if test_data_dict is not None:
                            testLib = test_data_dict[output_modality + '_test_lib'].to(device)
                    print('Set up decoder from', input_modality, 'to', output_modality)
                elif modalities[output_modality] == 'zinb' or modalities[output_modality] == 'nb':
                    output_distribution = modalities[output_modality]
                    if zero_inflated is True and output_distribution != 'zinb':
                        raise ValueError('zero_inflated should be True when output_distribution is zinb')
                    set_seed(SEED)
                    decoder = self.output_module[output_distribution](output_dim = train_data_dict[output_modality + '_train_mtx'].shape[1],
                                                                        output_batch_num = train_data_dict[output_modality + '_train_cov'].shape[1] if cov is not None else 0,
                                                                        hidden_layer = hidden_layer,
                                                                        infer_library_size = infer_library_size_rna,
                                                                        sample_factor = sample_factor_rna,
                                                                        feature_factor = feature_factor_rna,
                                                                        zero_inflated = zero_inflated,
                                                                        dispersion = dispersion)
                    if infer_library_size_rna is True:
                        trainLib = None
                        valLib = None
                        testLib = None
                    else:
                        trainLib = train_data_dict[output_modality + '_train_lib']
                        valLib = val_data_dict[output_modality + '_val_lib'].to(device)
                        if test_data_dict is not None:
                            testLib = test_data_dict[output_modality + '_test_lib'].to(device)
                    print('Set up decoder from', input_modality, 'to', output_modality)
            else:
                raise ValueError('scPair distribution should be one of the following: zinb, nb, ber, gau')
            encoder.apply(init_weights)
            encoder.to(device)
            decoder.apply(init_weights)
            decoder.to(device)
            output_layers = ['output']
            decay_param_encoder, nodecay_param_encoder, decay_name_encoder, nodecay_name_encoder = add_weight_decay(encoder, output_layer=output_layers)
            decay_param_decoder, nodecay_param_decoder, decay_name_decoder, nodecay_name_decoder = add_weight_decay(decoder, output_layer=output_layers)
            optimizer = torch.optim.AdamW([{'params': decay_param_encoder, 'weight_decay':L2_lambda, 'lr': learning_rate_prediction},
                                        {'params': decay_param_decoder, 'weight_decay':L2_lambda, 'lr': learning_rate_prediction},
                                        {'params': nodecay_param_encoder, 'weight_decay':0, 'lr': learning_rate_prediction}, 
                                        {'params': nodecay_param_decoder, 'weight_decay':0, 'lr': learning_rate_prediction}])
            if weight_decay == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1)
            elif weight_decay == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.1, last_epoch=-1)
            elif weight_decay == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
            else:
                scheduler = None
                Warning('weight_decay should be one of the following: ExponentialLR, StepLR, ReduceLROnPlateau. No weight decay applied.')
            if trainLib is not None:
                train_data = TensorDataset(trainData.to(device), trainBatch.to(device), trainLib.to(device), trainLabel.to(device)) if cov is not None else TensorDataset(trainData.to(device), trainLib.to(device), trainLabel.to(device))
            else:
                train_data = TensorDataset(trainData.to(device), trainBatch.to(device), trainLabel.to(device)) if cov is not None else TensorDataset(trainData.to(device), trainLabel.to(device))
            from torch.utils.data import DataLoader, Dataset
            set_seed(SEED)
            DataLoader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            if early_stopping_activation is True:
                early_stopping_patience = self.early_stopping_patience
                epochs_no_improve = 0
                early_stop = False
                min_val_loss = np.inf
                print('Enabling early stopping with patience of', early_stopping_patience)
            text = '\033[95m' + 'Start training feature predictor: {} '.format(prediction_order) + '\033[0m'            
            print('#' * (len(text)-5))
            print('# ' + text + ' #')
            print('#' * (len(text)-5))
            pbar = tqdm.tqdm(range(max_epochs))
            for epoch in pbar:
                training_mode(encoder, decoder)
                if trainLib is not None:
                    if cov is not None:
                        for idx, (x, b, lib, y) in enumerate(DataLoader_train):
                            train_net(x, y, lib, encoder, decoder, optimizer, likelihood_type=modalities[output_modality], add_cov=True, input_batch=b, output_batch=b)
                    else:
                        for idx, (x, lib, y) in enumerate(DataLoader_train):
                            train_net(x, y, lib, encoder, decoder, optimizer, likelihood_type=modalities[output_modality])
                else:
                    if cov is not None:
                        for idx, (x, b, y) in enumerate(DataLoader_train):
                            train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=modalities[output_modality], add_cov=True, input_batch=b, output_batch=b)
                    else:
                        for idx, (x, y) in enumerate(DataLoader_train):
                            train_net(x, y, None, encoder, decoder, optimizer, likelihood_type=modalities[output_modality])
                evaluating_mode(encoder, decoder)
                with torch.no_grad():
                    if valLib is not None:
                        if cov is not None:
                            val_total_loss = eval_net(valData, valLabel, valLib, encoder, decoder, likelihood_type=modalities[output_modality], add_cov=True, input_batch=valBatch, output_batch=valBatch)
                        else:
                            val_total_loss = eval_net(valData, valLabel, valLib, encoder, decoder, likelihood_type=modalities[output_modality])
                    else:
                        if cov is not None:
                            val_total_loss = eval_net(valData, valLabel, None, encoder, decoder, likelihood_type=modalities[output_modality], add_cov=True, input_batch=valBatch, output_batch=valBatch)
                        else:
                            val_total_loss = eval_net(valData, valLabel, None, encoder, decoder, likelihood_type=modalities[output_modality])
                    pbar.set_postfix({"Epoch": epoch+1, "validation Loss": val_total_loss.item()})
                    # tqdm.tqdm.write("Epoch [{}/{}], validation: Loss: {:.4f}".format(epoch+1, max_epochs, val_total_loss.item()))
                if optimizer.param_groups[0]['lr'] > 1e-7:
                    # print(optimizer.param_groups[0]['lr'])
                    if weight_decay == 'ExponentialLR':
                        scheduler.step()
                    elif weight_decay == 'StepLR':
                        scheduler.step()
                    elif weight_decay == 'ReduceLROnPlateau':
                        scheduler.step()
                    else:
                        pass
                if val_total_loss.item() < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = val_total_loss.item()
                    encoder_ckpt = copy.deepcopy(encoder)
                    decoder_ckpt = copy.deepcopy(decoder)
                else:
                    epochs_no_improve += 1
                    pbar.set_postfix({'Message': 'Early stopping triggered!', 'Patience Left': early_stopping_patience - epochs_no_improve})
                    # tqdm.tqdm.write("Early stopping triggered!", early_stopping_patience - epochs_no_improve)
                if epoch > 1 and epochs_no_improve == early_stopping_patience:
                    print("Early Stopped!")
                    early_stop = True
                    break
                else:
                    continue
            encoder_dict[input_modality + ' to ' + output_modality] = copy.deepcopy(encoder_ckpt.eval())
            decoder_dict[input_modality + ' to ' + output_modality] = copy.deepcopy(decoder_ckpt.eval())
            if save_model is True:
                if save_path is None:
                    save_path = os.getcwd()
                torch.save(encoder_dict[input_modality + ' to ' + output_modality], save_path + '/encoder_' + input_modality + '_to_' + output_modality + '.pt')
                torch.save(decoder_dict[input_modality + ' to ' + output_modality], save_path + '/decoder_' + input_modality + '_to_' + output_modality + '.pt')
        return encoder_dict, decoder_dict
    def train_bidirectional_mapping(self):
        embedding_dim = self.hidden_layer[-1]
        modality_names = self.modality_names
        mapping_dict = {}
        for input_modality in modality_names:
            output_modality = [mn for mn in modality_names if mn != input_modality][0]
            print('mapping direction:', input_modality, 'to', output_modality)
            input_embedding_train = self.embeddings[input_modality + '_train']
            output_embedding_train = self.embeddings[output_modality + '_train']
            input_embedding_val = self.embeddings[input_modality + '_val']
            output_embedding_val = self.embeddings[output_modality + '_val']
            trainData = TensorDataset(input_embedding_train.to(self.device), output_embedding_train.to(self.device))
            mapping_network = self.mapping_module(origin_module_dim = embedding_dim,
                                                target_module_dim = embedding_dim,
                                                translational_hidden_nodes = self.mapping_hidden_nodes,
                                                non_neg = self.mapping_non_neg,
                                                activation = self.mapping_activation,
                                                layernorm = self.mapping_layernorm,
                                                batchnorm = self.mapping_batchnorm,
                                                dropout_rate = self.mapping_dropout_rate)
            mapping_network.apply(init_weights)
            mapping_network.to(self.device)
            decay_param, nodecay_param, decay_name, nodecay_name = add_weight_decay(mapping_network, output_layer="bias")
            optimizer = torch.optim.AdamW([{'params': decay_param, 'weight_decay': self.L2_lambda, 'lr': self.mapping_learning_rate},
                                        {'params': nodecay_param, 'weight_decay':0, 'lr': self.mapping_learning_rate}])
            if self.weight_decay == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1)
            elif self.weight_decay == 'StepLR':
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=25, gamma=0.1, last_epoch=-1)
            elif self.weight_decay == 'ReduceLROnPlateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
            else:
                scheduler = None
                Warning('weight_decay should be one of the following: ExponentialLR, StepLR, ReduceLROnPlateau. No weight decay applied.')
            from torch.utils.data import DataLoader, Dataset
            set_seed(self.SEED)
            DataLoader_train = DataLoader(trainData, batch_size=self.batch_size, shuffle=True)
            if self.early_stopping_activation is True:
                early_stopping_patience = self.early_stopping_patience
                epochs_no_improve = 0
                early_stop = False
                min_val_loss = np.inf
                print('Enabling early stopping with patience of', early_stopping_patience)
            # print('#### Start mapping {} embeddings: ####'.format(input_modality + ' to ' + output_modality))
            text = '\033[95m' + 'Start mapping {} embeddings'.format(input_modality + ' to ' + output_modality) + '\033[0m'            
            print('#' * (len(text)-5))
            print('# ' + text + ' #')
            print('#' * (len(text)-5))
            pbar = tqdm.tqdm(range(self.max_epochs))
            for epoch in pbar:
                mapping_network.train();
                for idx, (x, y) in enumerate(DataLoader_train):
                    pred_emb = mapping_network(x)
                    training_loss = scaled_loss(y, pred_emb)
                    optimizer.zero_grad()
                    training_loss.backward()
                    optimizer.step()  
                mapping_network.eval();
                with torch.no_grad():
                    val_pred_emb = mapping_network(input_embedding_val.to(self.device))
                    val_total_loss = scaled_loss(output_embedding_val.to(self.device), val_pred_emb)
                    pbar.set_postfix({"Epoch": epoch+1, "validation Loss": val_total_loss.item()})
                    # tqdm.tqdm.write("Epoch [{}/{}], validation: Loss: {:.4f}".format(epoch+1, self.max_epochs, val_total_loss.item()))
                if optimizer.param_groups[0]['lr'] > 1e-7:
                    # print(optimizer.param_groups[0]['lr'])
                    if self.weight_decay == 'ExponentialLR':
                        scheduler.step()
                    elif self.weight_decay == 'StepLR':
                        scheduler.step()
                    elif self.weight_decay == 'ReduceLROnPlateau':
                        scheduler.step()
                    else:
                        pass
                if val_total_loss.item() < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = val_total_loss.item()
                    mapping_network_ckpt = copy.deepcopy(mapping_network)
                else:
                    epochs_no_improve += 1
                    pbar.set_postfix({'Message': 'Early stopping triggered!', 'Patience Left': early_stopping_patience - epochs_no_improve})
                    # tqdm.tqdm.write("Early stopping triggered!", early_stopping_patience - epochs_no_improve)
                if epoch > 1 and epochs_no_improve == early_stopping_patience:
                    print("Early Stopped!")
                    early_stop = True
                    break
                else:
                    continue
            mapping_dict[input_modality + '_to_' + output_modality] = copy.deepcopy(mapping_network_ckpt)
            if self.save_model is True:
                if self.save_path is None:
                    self.save_path = os.getcwd()
                torch.save(mapping_dict[input_modality + '_to_' + output_modality], self.save_path + '/mapping_' + input_modality + '_to_' + output_modality + '.pt')
        return mapping_dict
    def reference_embeddings(self):
        data_loader_dict_train, data_loader_dict_val, data_loader_dict_test = self.data_loader_dict_train, self.data_loader_dict_val, self.data_loader_dict_test
        modality_names = self.modality_names
        cov = self.cov
        embeddings = {}
        df_embeddings = {}
        for modality in modality_names:
            # print('predicting from', modality)
            encoder_cal = self.encoder_dict[modality + ' to ' + [mn for mn in modality_names if mn != modality][0]]
            encoder_cal.eval()
            encoder_cal.to(self.device)
            embeddings[modality + "_train"] = encoder_cal(data_loader_dict_train[modality + '_train_mtx'].to(self.device), data_loader_dict_train[modality + '_train_cov'].to(self.device))[0].cpu().detach() if cov is not None else encoder_cal(data_loader_dict_train[modality + '_train_mtx'].to(self.device), None)[0].cpu().detach()
            embeddings[modality + "_val"] = encoder_cal(data_loader_dict_val[modality + '_val_mtx'].to(self.device), data_loader_dict_val[modality + '_val_cov'].to(self.device))[0].cpu().detach() if cov is not None else encoder_cal(data_loader_dict_val[modality + '_val_mtx'].to(self.device), None)[0].cpu().detach()
            if data_loader_dict_test is not None:
                embeddings[modality + "_test"] = encoder_cal(data_loader_dict_test[modality + '_test_mtx'].to(self.device), data_loader_dict_test[modality + '_test_cov'].to(self.device))[0].cpu().detach() if cov is not None else encoder_cal(data_loader_dict_test[modality + '_test_mtx'].to(self.device), None)[0].cpu().detach()
            df_embeddings[modality + "_train"] = pd.DataFrame(embeddings[modality + "_train"].numpy(), index = self.train_metadata.index)
            df_embeddings[modality + "_val"] = pd.DataFrame(embeddings[modality + "_val"].numpy(), index = self.val_metadata.index)
            if data_loader_dict_test is not None:
                df_embeddings[modality + "_test"] = pd.DataFrame(embeddings[modality + "_test"].numpy(), index = self.test_metadata.index)
        self.embeddings = embeddings
        return embeddings, df_embeddings
    def mapped_embeddings(self):
        modality_names = self.modality_names
        mapping_dict = self.mapping_dict
        embeddings = self.embeddings
        mapped_embeddings = {}
        df_mapped_embeddings = {}
        for input_modality in modality_names:
            output_modality = [mn for mn in modality_names if mn != input_modality][0]
            # print('mapping direction:', input_modality, 'to', output_modality)
            input_embedding_train = embeddings[input_modality + '_train']
            output_embedding_train = embeddings[output_modality + '_train']
            input_embedding_val = embeddings[input_modality + '_val']
            output_embedding_val = embeddings[output_modality + '_val']
            input_embedding_test = embeddings[input_modality + '_test'] if self.data_loader_dict_test is not None else None
            output_embedding_test = embeddings[output_modality + '_test'] if self.data_loader_dict_test is not None else None
            mapping_network = mapping_dict[input_modality + '_to_' + output_modality]
            mapping_network.eval()
            mapping_network.to(self.device)
            mapped_embeddings[input_modality + ' to ' + output_modality + '_train'] = mapping_network(input_embedding_train.to(self.device)).cpu().detach()
            mapped_embeddings[input_modality + ' to ' + output_modality + '_val'] = mapping_network(input_embedding_val.to(self.device)).cpu().detach()
            if self.data_loader_dict_test is not None:
                mapped_embeddings[input_modality + ' to ' + output_modality + '_test'] = mapping_network(input_embedding_test.to(self.device)).cpu().detach()
            df_mapped_embeddings[input_modality + ' to ' + output_modality + '_train'] = pd.DataFrame(mapped_embeddings[input_modality + ' to ' + output_modality + '_train'].numpy(), index = self.train_metadata.index)
            df_mapped_embeddings[input_modality + ' to ' + output_modality + '_val'] = pd.DataFrame(mapped_embeddings[input_modality + ' to ' + output_modality + '_val'].numpy(), index = self.val_metadata.index)
            if self.data_loader_dict_test is not None:
                df_mapped_embeddings[input_modality + ' to ' + output_modality + '_test'] = pd.DataFrame(mapped_embeddings[input_modality + ' to ' + output_modality + '_test'].numpy(), index = self.test_metadata.index)
        return mapped_embeddings, df_mapped_embeddings
    def predict(self):
        data_loader_dict_train, data_loader_dict_val, data_loader_dict_test = self.data_loader_dict_train, self.data_loader_dict_val, self.data_loader_dict_test
        modality_names = self.modality_names
        embeddings = self.embeddings
        decoder_dict = self.decoder_dict
        predictions = {}
        for input_modality in modality_names:
            output_modality = [mn for mn in modality_names if mn != input_modality][0]
            input_distribution = self.modalities[input_modality]
            output_distribution = self.modalities[output_modality]
            print('predicting from', input_modality, 'to', output_modality)
            print('input_distribution:', input_distribution, 'output_distribution:', output_distribution)
            decoder_cal = decoder_dict[input_modality + ' to ' + output_modality]
            decoder_cal.eval()
            decoder_cal.to(self.device)
            if output_distribution == 'ber' and output_modality == 'Peaks' and self.infer_library_size_atac is True:
                encoder_cal = self.encoder_dict[input_modality + ' to ' + output_modality]
                encoder_cal.eval()
                encoder_cal.to(self.device)
                _, latent_train = encoder_cal(data_loader_dict_train[input_modality + '_train_mtx'].to(self.device), 
                                              data_loader_dict_train[input_modality + '_train_cov'].to(self.device) if self.cov is not None else None)
                _, latent_val = encoder_cal(data_loader_dict_val[input_modality + '_val_mtx'].to(self.device),
                                            data_loader_dict_val[input_modality + '_val_cov'].to(self.device) if self.cov is not None else None)
                if data_loader_dict_test is not None:
                    _, latent_test = encoder_cal(data_loader_dict_test[input_modality + '_test_mtx'].to(self.device),
                                                 data_loader_dict_test[input_modality + '_test_cov'].to(self.device) if self.cov is not None else None)
            elif output_distribution == 'nb' or output_distribution == 'zinb':
                if output_modality == 'Gene Expression' and self.infer_library_size_rna is True:
                    encoder_cal = self.encoder_dict[input_modality + ' to ' + output_modality]
                    encoder_cal.eval()
                    encoder_cal.to(self.device)
                    _, latent_train = encoder_cal(data_loader_dict_train[input_modality + '_train_mtx'].to(self.device),
                                                  data_loader_dict_train[input_modality + '_train_cov'].to(self.device) if self.cov is not None else None)
                    _, latent_val = encoder_cal(data_loader_dict_val[input_modality + '_val_mtx'].to(self.device),
                                                data_loader_dict_val[input_modality + '_val_cov'].to(self.device) if self.cov is not None else None)
                    if data_loader_dict_test is not None:
                        _, latent_test = encoder_cal(data_loader_dict_test[input_modality + '_test_mtx'].to(self.device),
                                                     data_loader_dict_test[input_modality + '_test_cov'].to(self.device) if self.cov is not None else None)
            else:
                latent_train = None
                latent_val = None
                latent_test = None
            if output_distribution == 'zinb' or output_distribution == 'nb' or output_distribution == 'ber':
                predictions[output_modality + "_train"] = decoder_cal(latent_rep = embeddings[input_modality + '_train'].to(self.device),
                                                                    output_batch = data_loader_dict_train[output_modality + '_train_cov'].to(self.device) if self.cov is not None else None,
                                                                    lib = torch.FloatTensor([1]).to(self.device),
                                                                    latent_lib = latent_train)[0].cpu().detach().numpy()
                predictions[output_modality + "_val"] = decoder_cal(latent_rep = embeddings[input_modality + '_val'].to(self.device),
                                                                    output_batch = data_loader_dict_val[output_modality + '_val_cov'].to(self.device) if self.cov is not None else None,
                                                                    lib = torch.FloatTensor([1]).to(self.device),
                                                                    latent_lib = latent_val)[0].cpu().detach().numpy()
                if data_loader_dict_test is not None:
                    predictions[output_modality + "_test"] = decoder_cal(latent_rep = embeddings[input_modality + '_test'].to(self.device),
                                                                    output_batch = data_loader_dict_test[output_modality + '_test_cov'].to(self.device) if self.cov is not None else None,
                                                                    lib = torch.FloatTensor([1]).to(self.device),
                                                                    latent_lib = latent_test)[0].cpu().detach().numpy()
            elif output_distribution == 'gau':
                print('output_distribution is Normal Distribution')
                predictions[output_modality + "_train"] = decoder_cal(latent_rep = embeddings[input_modality + '_train'].to(self.device),
                                                                    output_batch = data_loader_dict_train[output_modality + '_train_cov'].to(self.device) if self.cov is not None else None,
                                                                    lib = None,
                                                                    latent_lib = None).cpu().detach().numpy()
                predictions[output_modality + "_val"] = decoder_cal(latent_rep = embeddings[input_modality + '_val'].to(self.device),
                                                                    output_batch = data_loader_dict_val[output_modality + '_val_cov'].to(self.device) if self.cov is not None else None,
                                                                    lib = None,
                                                                    latent_lib = None).cpu().detach().numpy()
                if data_loader_dict_test is not None:
                    predictions[output_modality + "_test"] = decoder_cal(latent_rep = embeddings[input_modality + '_test'].to(self.device),
                                                                    output_batch = data_loader_dict_test[output_modality + '_test_cov'].to(self.device) if self.cov is not None else None,
                                                                    lib = None,
                                                                    latent_lib = None).cpu().detach().numpy()
        return predictions
    def augment(self, unimodal_scobj, unimodal_modalities, unimodal_cov = None):
        self.unimodal_scobj = unimodal_scobj
        self.unimodal_cov = unimodal_cov
        self.unimodal_modalities = unimodal_modalities
        self.unimodal_modality_names = list(unimodal_modalities.keys())
        self.unimodal_modality_distributions = list(unimodal_modalities.values())
        if set(self.unimodal_modality_names).issubset(self.modalities.keys()):
            print('Found matched modality and corresponding distribution for unimodal data...')
            print('Modality:', self.unimodal_modality_names, 'Distribution:', self.unimodal_modality_distributions)
        else:
            print('Paired multimodal data:', self.modalities)
            print('Unimodal data:', self.unimodal_modalities)
            raise ValueError('Unimodal data should have the same modality as one of the multimodal data')
        if self.unimodal_cov != self.cov:
            if self.unimodal_cov is not None and self.cov is not None:
                raise ValueError('Unimodal data have totally different covariate matrix compared to multimodal data')
            elif self.unimodal_cov is None and self.cov is not None:
                self.unimodal_cov = self.cov
                self.unimodal_cov_dummy = pd.DataFrame(data=np.zeros((self.unimodal_scobj.shape[0], self.cov_dummy.shape[1])), columns=self.cov_dummy.columns, index=self.unimodal_scobj.obs.index)
                print('Using multimodal cov for unimodal data, and set unimodal cov to all zeros during scPair_augment training')
            elif self.unimodal_cov is not None and self.cov is None:
                raise ValueError('Unimodal data have covariate matrix while multimodal data do not')
            #     print('Using unimodal cov = None for unimodal data during inference, and set multimodal cov to all zeros to match the dimension during scPair_augment training')
            #     self.unimodal_cov_dummy = pd.get_dummies(self.unimodal_scobj.obs[self.unimodal_cov])
            #     self.cov_dummy_ = pd.DataFrame(data=np.zeros((self.scobj.shape[0], self.unimodal_cov_dummy.shape[1])), columns=self.unimodal_cov_dummy.columns, index=self.scobj.obs.index)
            else:
                self.unimodal_cov = None
                self.unimodal_cov_dummy = None
        unimodal_train_input = torch.FloatTensor(self.unimodal_scobj.X.toarray()) if scipy.sparse.issparse(self.unimodal_scobj.X) else torch.FloatTensor(self.unimodal_scobj.X)
        unimodal_train_batch = torch.FloatTensor(self.unimodal_cov_dummy.to_numpy()) if self.unimodal_cov is not None else None
        bimodal_train_input = self.data_loader_dict_train[self.unimodal_modality_names[0] + '_train_mtx']
        bimodal_train_batch = self.data_loader_dict_train[self.unimodal_modality_names[0] + '_train_cov'] if self.cov is not None else None
        bimodal_val_input = self.data_loader_dict_val[self.unimodal_modality_names[0] + '_val_mtx']
        bimodal_val_batch = self.data_loader_dict_val[self.unimodal_modality_names[0] + '_val_cov'] if self.cov is not None else None
        bimodal_test_input = self.data_loader_dict_test[self.unimodal_modality_names[0] + '_test_mtx'] if self.data_loader_dict_test is not None else None
        bimodal_test_batch = self.data_loader_dict_test[self.unimodal_modality_names[0] + '_test_cov'] if self.cov is not None and self.data_loader_dict_test is not None else None
        encoder_to_be_updated = copy.deepcopy(self.encoder_dict[self.unimodal_modality_names[0] + ' to ' + [mn for mn in self.modalities.keys() if mn != self.unimodal_modality_names[0]][0]])
        encoder_to_be_updated.to(self.device)
        encoder_to_be_updated.eval()
        unimodal_train_label = encoder_to_be_updated(unimodal_train_input.to(self.device), unimodal_train_batch.to(self.device) if self.unimodal_cov is not None else None)[0].detach()
        bimodal_train_label = encoder_to_be_updated(bimodal_train_input.to(self.device), bimodal_train_batch.to(self.device) if self.cov is not None else None)[0].detach()
        bimodal_val_label = encoder_to_be_updated(bimodal_val_input.to(self.device), bimodal_val_batch.to(self.device) if self.cov is not None else None)[0].detach()
        if bimodal_test_input is not None:
            bimodal_test_label = encoder_to_be_updated(bimodal_test_input.to(self.device), bimodal_test_batch.to(self.device) if self.cov is not None else None)[0].detach()
        else:
            bimodal_test_label = None
        if self.unimodal_cov is not None:
            trainData = TensorDataset(unimodal_train_input.to(self.device), unimodal_train_batch.to(self.device), unimodal_train_label.to(self.device))
        else:
            trainData = TensorDataset(unimodal_train_input.to(self.device), unimodal_train_label.to(self.device))
        # unimodal_val_input/batch/label here will be used for early stopping: concat bimodal train and val data
        unimodal_val_input = torch.cat((bimodal_train_input, bimodal_val_input), 0).to(self.device)
        unimodal_val_batch = torch.cat((bimodal_train_batch, bimodal_val_batch), 0).to(self.device) if self.unimodal_cov is not None else None
        unimodal_val_label = torch.cat((bimodal_train_label, bimodal_val_label), 0).to(self.device)
        pbar = tqdm.tqdm(range(self.max_epochs))
        optimizer_unimodal = torch.optim.AdamW(encoder_to_be_updated.parameters(), lr=self.learning_rate_prediction, weight_decay=self.L2_lambda)
        if self.early_stopping_activation is True:
            early_stopping_patience = self.early_stopping_patience
            epochs_no_improve = 0
            early_stop = False
            min_val_loss = np.inf
            print('Enabling early stopping with patience of', early_stopping_patience)
        augment_encoder_dict = {}
        for epoch in pbar:
            if self.unimodal_cov is not None:
                encoder_to_be_updated.train()
                for idx, (x, b, y) in enumerate(DataLoader(trainData, batch_size=self.batch_size, shuffle=True)):
                    pred = encoder_to_be_updated(x, b if self.unimodal_cov is not None else None)[0]
                    loss = scaled_loss(y, pred)
                    optimizer_unimodal.zero_grad()
                    loss.backward()
                    optimizer_unimodal.step()
            else:
                for idx, (x, y) in enumerate(DataLoader(trainData, batch_size=self.batch_size, shuffle=True)):
                    pred = encoder_to_be_updated(x, None)[0]
                    loss = scaled_loss(y, pred)
                    optimizer_unimodal.zero_grad()
                    loss.backward()
                    optimizer_unimodal.step()
            encoder_to_be_updated.eval()
            with torch.no_grad():
                if self.unimodal_cov is not None:
                    val_total_loss = scaled_loss(unimodal_val_label, encoder_to_be_updated(unimodal_val_input, unimodal_val_batch)[0])
                else:
                    val_total_loss = scaled_loss(unimodal_val_label, encoder_to_be_updated(unimodal_val_input, None)[0])
            if val_total_loss.item() < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_total_loss.item()
                self.augment_encoder_ckpt = copy.deepcopy(encoder_to_be_updated)
            else:
                epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print("Early Stopped!")
                early_stop = True
                break
            else:
                continue
        self.augment_encoder_ckpt.eval()
        if self.save_model is True:
            if self.save_path is None:
                self.save_path = os.getcwd()
            torch.save(self.augment_encoder_ckpt, self.save_path + '/augment_encoder_' + self.unimodal_modality_names[0] + '_to_' + [mn for mn in self.modalities.keys() if mn != self.unimodal_modality_names[0]][0] + '.pt')
        augment_encoder_dict['scPair_augment: ' + self.unimodal_modality_names[0] + ' to ' + [mn for mn in self.modalities.keys() if mn != self.unimodal_modality_names[0]][0]] = copy.deepcopy(self.augment_encoder_ckpt)
        self.augment_encoder_dict = augment_encoder_dict
        augment_emb = {}
        unimodal_predictions = self.augment_encoder_ckpt(unimodal_train_input.to(self.device), unimodal_train_batch.to(self.device) if self.unimodal_cov is not None else None)[0].cpu().detach().numpy()
        augment_emb[self.unimodal_modality_names[0] + '_train_unimodal'] = unimodal_predictions
        bimodal_train_predictions = self.augment_encoder_ckpt(bimodal_train_input.to(self.device), bimodal_train_batch.to(self.device) if self.cov is not None else None)[0].cpu().detach().numpy()
        augment_emb[self.unimodal_modality_names[0] + '_train_bimodal'] = bimodal_train_predictions
        bimodal_val_predictions = self.augment_encoder_ckpt(bimodal_val_input.to(self.device), bimodal_val_batch.to(self.device) if self.cov is not None else None)[0].cpu().detach().numpy()
        augment_emb[self.unimodal_modality_names[0] + '_val_bimodal'] = bimodal_val_predictions
        if bimodal_test_input is not None:
            bimodal_test_predictions = self.augment_encoder_ckpt(bimodal_test_input.to(self.device), bimodal_test_batch.to(self.device) if self.cov is not None else None)[0].cpu().detach().numpy()
            augment_emb[self.unimodal_modality_names[0] + '_test_bimodal'] = bimodal_test_predictions
        else:
            bimodal_test_predictions = None
        return augment_encoder_dict, augment_emb
