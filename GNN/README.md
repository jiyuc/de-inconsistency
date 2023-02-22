# Instruction for GNN pre-training

This directory includes the source code for pre-training GNN to learn existing GO annotation patterns.

### How to generate pre-trained node embeddings with our prepared dataset?

Open a terminal, execute `python train_graph.py`, the node embeddings will be saved as `gnn_model.pt`

### How to customize the dataset?
`*.csv` files contains the positive and negative edges which will be leveraged to train the model.
positive edges are consistent GOA, named as 'CO' in file
negative edges are inconistent GOA, named as 'IC' in file
You can add more consistent GOA into the `train_positive_*.csv` files and inconsistent GOA into the `train_negative_*.csv` file, following the format indicated in each `*.csv` file.

### How to conduct pre-training after dataset customization?
Open a terminal, execute `python train_graph.py` to pre-train node embeddings `gnn_model.pt`, which can be leveraged to combine with *PubMedBERT*