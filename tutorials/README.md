## Tutorials
### Basic example for running scPair


Let's assume you already have the (raw count) RNA cellxgene and the paired (binarized) ATAC cellxpeak matrices in hand
```python
# import necessary packages for single cell analysis
import os
import copy
import scipy
import random
import numpy as np
import pandas as pd
from scipy import sparse

import anndata
import scanpy as sc
import scvi
from scpair import *

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
if you have predefined split
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
# this is an example using sciCAR cell line dataset
# scPair_split
# train    2700
# test      842
# val       674
```

You can also try the preprocessed multimodal AnnData object we have prepared: [figshare link](https://figshare.com/s/ea98335f6f8a0abc8ae5), DOI: 10.6084/m9.figshare.26143948

```python
import os
import copy
import scipy
import random
import numpy as np
import pandas as pd
from scipy import sparse

import anndata
import scanpy as sc
import scvi
from scpair import *

"""
load multimodal data
"""
adata_paired = sc.read_h5ad(data_path + "./data/sciCAR/paired_full_features.h5ad")

"""
set up scPair object
"""
scpair_setup = scPair_object(scobj = adata_paired, cov=None, modalities = {'Gene Expression': 'zinb', 'Peaks': 'ber'},
                         sample_factor_rna=True, sample_factor_atac=False, infer_library_size_rna=False, infer_library_size_atac=True,
                         batchnorm=True, layernorm=True, SEED=0, hidden_layer=[800, 30], dropout_rate=0.1, learning_rate_prediction=1e-3, max_epochs=1000)
```

In this specific case, we do not need to provide covariates. But if in your own data, you want to correct the batch effect or other covariates, please set the `cov` argument to  `cov=['batch']`, or `cov=['batch0', 'batch1']` if you have more than one covariates to be corrected. 

Please make sure the names of covariates exist in the column names of your metadata `adata_paired.obs`.

Additionally, please ensure that the type of categorical covariates is set properly using `.astype('category')` so that scPair will automatically generate one-hot encoded covariates. Otherwise, scPair will consider them as continuous data if they are numeric.


```python
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

```

This step below is optional. 

If you have an extra unimodal data object from either modality, make sure for the specific modality, you have the same set of features with the same order.

```python

"""
optional: adding unimodal data
"""
augment = scpair_setup.augment(unimodal_scobj = unimodal_rna_adata, unimodal_modalities = {'Gene Expression': 'zinb'}, unimodal_cov = None)
augment_encoder_dict, augment_emb = augment # return the updated modality encoder and embeddings
```

### For more detailed scPair argument explanation, please go to this [link](https://github.com/quon-titative-biology/scPair/blob/main/tutorials/understand_the_argument.md)

---
