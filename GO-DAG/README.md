
# Instruction for pre-training on GO-DAG with GNN-GraphSAGE

This directory includes the source code for pre-training the GNN-GraphSAGE model for the integration of GO specificity knowledge into 16-dim node embeddings 🤖️. A pre-trained node embedding with integration of GO specificity knowledge is saved in `out/go_hier_16dim.pt`

### How to generate pre-trained node embeddings with our prepared dataset, derived from NCBI-*gene2go*?

In terminal, execute `python generate_node_embedding.py` (or `python3 generate_node_embedding.py`), the node embeddings will be saved as `go_hier_16dim.pt`. It will take a while to generate the node embeddings.

### How to customize the dataset?
You can easily extend existing dataset with addition of new instances or removal of old instances 😊.\
The `go_dag.csv` file contains the 'parent_of' and 'children_of' edges which will be leveraged to train the model.\
'children_of' edges are the merge of 'is_a' and 'part_of' relation\
'parent_of' are the reverse of 'children_of'\
You can add more paired GO terms into the `go_dag.csv` files with corresponding new GO info into `go_info.csv` file.\
⚠️ A prompt will be given during pre-training if any GO info is missing in the `go_info.csv`, notifying you to retrieve the required info.

### How to conduct pre-training after dataset customization?
In terminal, execute `python generate_node_embedding.py` (or `python3 generate_node_embedding.py`) to pre-train node embeddings -- `go_hier_16dim.pt`, which can be leveraged to combine with *PubMedBERT* for downstream task, i.e., end-to-end GOA inconsistency detection.