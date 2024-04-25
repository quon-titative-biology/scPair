## Tutorials
### Basic example for running scPair
```python
# import necessary packages for single cell analysis and visualization
# such as: scipy, scanpy, sklearn, numpy, pandas, umap, matplotlib, seaborn, etc
import anndata
import scanpy as sc
from scpair import *

# assuming you already have the (raw count) RNA cellxgene and the paired (binarized) ATAC cellxpeak matrices in hand
# as well as split sets (train, validation, and held-out test sets)
####################################################################
"""
build modality-specific scanpy object
"""
rna_adata = sc.AnnData(X=rna_mtx, obs=meta)
atac_adata = sc.AnnData(X=atac_mtx, obs=meta)

"""
make paired multi-modal object
"""
adata_paired = merge_paired_data([rna_adata, atac_adata])

"""
predefined split
"""
train_id = pd.read_table(index_path + "train_id.txt",  header=None)
val_id = pd.read_table(index_path + "val_id.txt",  header=None)
test_id = pd.read_table(index_path + "test_id.txt",  header=None)
train_id = train_id.iloc[:,0].tolist()
val_id = val_id.iloc[:,0].tolist()
test_id = test_id.iloc[:,0].tolist()
pre_split = [train_id, val_id, test_id]

"""
perform the split operation for the multi-modal object
"""
adata_paired = training_split(adata_paired, pre_split=[train_id, val_id, test_id])
adata_paired.obs.scPair_split.value_counts()
# scPair_split
# train    2700
# test      842
# val       674

"""
set up scPair object
"""
scpair_setup = scPair_object(scobj = adata_paired, cov=['batch'], modalities = {'Gene Expression': 'zinb', 'Peaks': 'ber'},
                         sample_factor_rna=True, sample_factor_atac=False, infer_library_size_rna=False, infer_library_size_atac=True,
                         batchnorm=True, layernorm=True, SEED=0, hidden_layer=[800, 30], dropout_rate=0.1, learning_rate_prediction=1e-3, max_epochs=1000)

"""
start running optimization for scPair framework
"""
res = scpair_setup.run()


"""
extrct the learned embeddings
"""
e, e_df = scpair_setup.reference_embeddings()
me, me_df = scpair_setup.mapped_embeddings()
e_df.keys() # dict_keys(['Gene Expression_train', 'Gene Expression_val', 'Gene Expression_test', 'Peaks_train', 'Peaks_val', 'Peaks_test'])
me_df.keys() # dict_keys(['Gene Expression to Peaks_train', 'Gene Expression to Peaks_val', 'Gene Expression to Peaks_test', 'Peaks to Gene Expression_train', 'Peaks to Gene Expression_val', 'Peaks to Gene Expression_test'])

"""
extrct the cross-modality predictions 
"""
predictions = scpair_setup.predict()
predictions.keys() # dict_keys(['Peaks_train', 'Peaks_val', 'Peaks_test', 'Gene Expression_train', 'Gene Expression_val', 'Gene Expression_test'])


"""
optional: adding unimodal data
"""
augment = scpair_setup.augment(unimodal_scobj = unimodal_rna_adata, unimodal_modalities = {'Gene Expression': 'zinb'}, unimodal_cov = None)
augment_encoder_dict, augment_emb = augment # return the updated modality encoder and embeddings





