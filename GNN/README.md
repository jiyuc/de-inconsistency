# Instruction for pre-training on existing GO annotation patterns with GNN-HeteroGraphSAGE

This directory includes the source code for pre-training the GNN-HeteroGraphSAGE model for the integration of existing GO annotation patterns into node embeddings 🤖️.

### How to generate pre-trained node embeddings with our prepared dataset, derived from NCBI-*gene2go*?

In terminal, execute `python train_graph.py` (or `python3 train_graph.py`), the node embeddings will be saved as `gnn_model.pt`. It will take a while to generate the node embeddings.

### How to customize the dataset?
You can easily extend existing dataset with addition of new instances or removal of old instances 😊.\
The `*.csv` file contains the positive and negative edges which will be leveraged to train the model.\
positive edges are consistent GOA, named as 'CO' in file\
negative edges are inconistent GOA, named as 'IC' in file\
You can add more consistent GOA into the `train_positive_*.csv` files and inconsistent GOA into the `train_negative_*.csv` file, following the format indicated in each `*.csv` file.\
⚠️ We encourage you to retain the `dev_*` and `test_*` files.

### How to conduct pre-training after dataset customization?
In terminal, execute `python train_graph.py` (or `python3 train_graph.py`) to pre-train node embeddings -- `gnn_model.pt`, which can be leveraged to combine with *PubMedBERT* for downstream task, i.e., end-to-end GOA inconsistency detection.