# Instruction for pre-training on existing GO annotation patterns with GNN-HeteroGraphSAGE ###
This directory includes the source code for pre-training the GNN-HeteroGraphSAGE model for the integration of existing GO annotation patterns into node embeddings, namely `exGOA` 🤖️.

### Direct use of pre-trained exGOA embedding ###
The pre-trained exGOA embedding is stored as `out/gnn_model.pt`. If you want to expand or train another exGOA embedding, please follow the instructions below:

### How to customize the dataset for pre-training a exGOA embeeding? ###
You can easily extend existing dataset with addition of new instances or removal of old instances 😊.\
The `*.csv` files within this folder contain pair-wised positive and negative edges on the existing GOA graph.\
positive edges are consistent GOA, named as 'CO' in `*_positives.csv`\
negative edges are inconistent GOA, named as 'IC' in `*_negatives.csv`\
You can customize your own consistent GOA instances into the `train_positive_*.csv` files and inconsistent GOA into the `train_negative_*.csv` file.\
The `dev_*` and `test_*` files are only supplied for hyper-parameters tuning and testing purpose.

### How to pre-train your customized exGOA embeddings ###
Open a terminal and execute `python train_graph.py` (or `python3 train_graph.py`), the exGOA will be encoded as `gnn_model.pt`. It will take a while to generate the exGOA embeddings.

### Required library ###
The Deep Graph Learning https://www.dgl.ai/
