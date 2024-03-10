
# Instruction for pre-training on GO-DAG with GNN-GraphSAGE
This directory includes the source code for pre-training the GNN-GraphSAGE model for the integration of GO specificity knowledge into 16-dim node embeddings 🤖️. A pre-trained node embedding with integration of GO specificity knowledge is saved in `out/go_hier_16dim.pt`

### Direct use of pre-trained GoDg embedding ###
A pre-trained 16-dim GoDg embedding is stored in `out/go_hier_16dim.pt`

### How to customize the dataset?
You can customize dataset to pre-training your own GoDg node embedding by modifying the `go_dag.csv` and `go_info.csv`.\
The `go_dag.csv` file contains the 'parent_of' and 'children_of' edges which will be leveraged to train the model.\
'children_of' edges are the merge of 'is_a' and 'part_of' relation of GO DAG\
'parent_of' edges are the reverse of 'children_of'\
You can customize paired GO terms into the `go_dag.csv` files with corresponding new GO info saved within `go_info.csv` file.\


### How to pre-trained customized node embeddings ###
Open a terminal, execute `python generate_node_embedding.py` (or `python3 generate_node_embedding.py`), the node embeddings will be saved as `go_hier_16dim.pt`. It will take a while to generate the node embeddings.
