### How to run +GoDg & exGOA with GNN-PubMedBERT ###
The core implementation of GoDg & exGOA can be evaluated by execute `bash g_evaluation.sh`.
The test data can be found in `data/test.txt`.

### How to apply the inconsistency detection model on customized datset ###
- Modifying the `data/test.txt` in following format for each line:

| PMID     | GeneID | a list of GO ids as existing annotation to the GeneID |  GOId(The one that requires inconsistency detection)| <label: optional> |
|----------|--------|-----------------------------------|--------------------------------|-------|
| 20421391 | 53314  | GO:0045190, GO:0045064     | GO:2000320                     | os    |

- Add resource indexed by those Ids by following the instructions in `res/README.md`

### Where to find the technical implementation for the joint modelling of GNN-PubMedBERT ###
The flat concatenation of pre-trained 16-dim GoDg embedding and 16-dim exGOA embedding can be found in the `joint.py` and `graph_preprocessing.py`