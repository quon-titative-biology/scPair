### scPair_object()

- `scobj`: This is the input anndata for the model, in the form of a single-cell (scanpy) object.

- `modalities`: This is a dictionary that maps the name of each modality to its type. For example: `{'Gene Expression': 'zinb','Peaks': 'ber'}`

- `cov`: This is a list of covariates to include in the model. For example: `['sex', 'donor_id']`.

- `SEED`: This is the seed for the random number generator, used to ensure reproducibility.

- `hidden_layer`: This is a list that specifies the number of nodes in each hidden layer of the neural network. For example: `[900, 40]`.

- `dropout_rate`: This is the dropout rate for the dropout layers in the neural network.

- `batchnorm`: This is a boolean that indicates whether to use batch normalization.

- `layernorm`: This is a boolean that indicates whether to use layer normalization.

- `learning_rate_prediction`: This is the learning rate for the optimizer during the prediction phase.

- `batch_size`: This is the size of the batches for training the model.
  
- `weight_decay`: This is the weight decay (L2 penalty) for the optimizer.

- `L2_lambda`: This is the lambda parameter for L2 regularization.

- `max_epochs`: This is the maximum number of epochs for training the model.

- `activation`: This is the activation function to use in the neural network.

- `early_stopping_activation`: This is a boolean that indicates whether to use early stopping.

- `early_stopping_patience`: This is the number of epochs with no improvement after which training will be stopped.

- `device`: This is the device to use for computations (CPU or GPU).

- `save_model`: This is a boolean that indicates whether to save the trained model.

- `save_path`: This is the path where the trained model should be saved, the defualt path is the current working directory.

- `input_module`: This is the module to use for the input layer of the neural network, please keep it as default.

- `output_module`: This is a dictionary that maps the type of each modality to the module to use for its output layer, please keep it as default.

- `add_linear_layer`: This is a boolean that indicates whether to add a linear layer to the encoding networks.

- `sample_factor_rna`, `feature_factor_rna`, `sample_factor_atac`, `feature_factor_atac`: These are booleans that indicate whether to include sample and feature factors for RNA and ATAC data.

- `zero_inflated`: This is a boolean that indicates whether to use a zero-inflated model.

- `dispersion`: This is the type of dispersion to use in the model. For example: `feature` or `feature-cell`.

- `infer_library_size_rna`, `infer_library_size_atac`: These are booleans that indicate whether to infer the library size for RNA and ATAC data.

- `mapping_module`: This is the module to use for the mapping layer of the neural network, please keep it as default.

- `mapping_hidden_nodes`: This is the number of hidden nodes in the mapping layer, `None` as default.

- `mapping_learning_rate`: This is the learning rate for the optimizer during the mapping phase.

- `mapping_non_neg`: This is a boolean that indicates whether to enforce non-negativity constraints in the mapping layer, which depends on the activation function if infer_library_size is deactivated.

- `mapping_layernorm`, `mapping_batchnorm`: These are booleans that indicate whether to use layer normalization and batch normalization in the mapping layer. Only need to adjust them if you prefer MLP mappings rather than single-layer linear mappings.

- `mapping_activation`: This is the activation function to use in the mapping layer.

- `mapping_dropout_rate`: This is the dropout rate for the dropout layers in the mapping layer.
