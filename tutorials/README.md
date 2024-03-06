### Tutorials
1. example of prediction network training
```python
# import necessary packages for single cell analysis and visualization
# such as: scipy, scanpy, sklearn, numpy, pandas, umap, matplotlib, seaborn, etc
from scpair import *

# assuming you already have the (raw count) RNA cellxgene and the paired (binarized) ATAC cellxpeak matrices in hand
# as well as split sets (train, validation, and held-out test sets)
####################################################################
# tensor prep
# EXAMPLE -->
# input: atac + batch
trainData = torch.FloatTensor(np.array(atac_train)).to(device)
valData = torch.FloatTensor(np.array(atac_val)).to(device)
trainData_batch = torch.FloatTensor(np.array(batch_train)).to(device)
valData_batch = torch.FloatTensor(np.array(batch_val)).to(device)
# output: rna
trainLabel = torch.FloatTensor(np.array(rna_train)).to(device) 
valLabel = torch.FloatTensor(np.array(rna_val)).to(device)
# rna library size
trainLabel_lib = torch.FloatTensor(np.array(rna_size_train)).to(device).reshape(trainLabel.size()[0],-1)
valLabel_lib = torch.FloatTensor(np.array(rna_size_val)).to(device).reshape(valLabel.size()[0],-1)

train_dataset = TensorDataset(trainData, trainData_batch, trainLabel_lib, trainLabel, trainData_batch)
val_dataset = TensorDataset(valData, valData_batch, valLabel_lib, valLabel, valData_batch)
input_batch_num, output_batch_num =  trainData_batch.shape[1], trainLabel_batch.shape[1]
input_dim, output_dim = trainData.shape[1], trainLabel.shape[1]

####################################################################
#                            TRAINING                              #
####################################################################
# encoder + decoder (ATAC > RNA)
# reproducibility
SEED = 12
# hyper-param
hidden_layer = [800, 30]
dropout_rate = 0.1
batchnorm = False
layernorm = True
learning_rate = 1e-3
batch_size = 120
L2_lambda = 1e-8
epochs = 1000
activation = nn.GELU()
# data loader prep
from torch.utils.data import DataLoader, Dataset
set_seed(1234)
DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# framework setup
# encoder
set_seed(SEED)
encoder = Input_Module(input_dim=input_dim, 
                       input_batch_num=input_batch_num, 
                       hidden_layer=hidden_layer, 
                       layernorm=layernorm,
                       activation=activation, 
                       batchnorm=batchnorm,
                       dropout_rate=dropout_rate, 
                       infer_library_size=False, 
                       add_linear_layer=True)
encoder.apply(init_weights)
encoder.to(device)
# decoder
set_seed(SEED)
decoder = Output_Module_Raw(output_dim=output_dim, 
                            output_batch_num=output_batch_num, 
                            hidden_layer=hidden_layer, 
                            infer_library_size=False, 
                            sample_factor=True,
                            feature_factor=False, 
                            zero_inflated=False, 
                            dispersion="feature")
decoder.apply(init_weights)
decoder.to(device)
print(encoder, decoder)
# weight decay, optimization, and learning rate decay setup
output_layers = ["output"]
decay_param_encoder, nodecay_param_encoder, decay_name_encoder, nodecay_name_encoder = add_weight_decay(encoder, output_layer=output_layers)
decay_param_decoder, nodecay_param_decoder, decay_name_decoder, nodecay_name_decoder = add_weight_decay(decoder, output_layer=output_layers)
optimizer = torch.optim.AdamW([{'params': decay_param_encoder, 'weight_decay':L2_lambda, 'lr': learning_rate},
                               {'params': decay_param_decoder, 'weight_decay':L2_lambda, 'lr': learning_rate},
                               {'params': nodecay_param_encoder, 'weight_decay':0, 'lr': learning_rate}, 
                               {'params': nodecay_param_decoder, 'weight_decay':0, 'lr': learning_rate}])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1)
# early-stopping setup
min_val_loss = np.Inf
epochs_no_improve = 0
early_stop = False
patience = 25
# start training
for epoch in range(epochs):
        training_mode(encoder, decoder)
        for idx, (x, x_, lib, y, y_) in enumerate(DataLoader):  
           train_net(x, y, lib, # None, # 
                     encoder, decoder, optimizer, likelihood_type="nb", add_cov=True, input_batch=x_)#, output_batch=y_)#None)#
        evaluating_mode(encoder, decoder)
        with torch.no_grad():
           train_total_loss = eval_net(trainData, trainLabel, trainLabel_lib, # None, # 
                                       encoder, decoder, likelihood_type="nb", add_cov=True, input_batch=trainData_batch)#, output_batch=trainLabel_batch)#None)#
           print("Epoch [{}/{}], train     : Loss: {:.4f}".format(epoch+1, epochs, train_total_loss.item()))
           val_total_loss = eval_net(valData, valLabel, valLabel_lib, # None, #
                                     encoder, decoder, likelihood_type="nb", add_cov=True, input_batch=valData_batch)#, output_batch=valLabel_batch)#None)#
           print("Epoch [{}/{}], validation: Loss: {:.4f}".format(epoch+1, epochs, val_total_loss.item()))
        if optimizer.param_groups[0]['lr'] > 1e-4:
           print(optimizer.param_groups[0]['lr'])
         #   scheduler.step()
        if val_total_loss.item() < min_val_loss:
           epochs_no_improve = 0
           min_val_loss = val_total_loss.item()
           atac_rna_encoder_ckpt = copy.deepcopy(encoder)
           atac_rna_decoder_ckpt = copy.deepcopy(decoder)
        else:
           epochs_no_improve += 1
           print("Early stopping triggered!", patience - epochs_no_improve)
        if epoch > 1 and epochs_no_improve == patience:
           print("Stopped!")
           early_stop = True
           break
        else:
           continue


```

2. example of mapping network training
```python

```
