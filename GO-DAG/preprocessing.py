import torch
import glob
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
import pandas as pd
from corpus_path import go_info, go_hier
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import dgl



class MakeGODAG:
    GO_HIER_MAP = {'children_of': 1, # merge is_a and part_of
                   'parent_of': 0, # merge parent_is_a and parent_part_of
                  }

    def __init__(self, go_hier_csv):
        print("constructing GO DAG")
        
        # the go hierarchical relation is stored in go_network.csv file
        go_hier_df = pd.read_csv(go_hier_csv)
        go_src = go_hier_df['src'].tolist()
        go_des = go_hier_df['des'].tolist()

        # encode nodes to unique node id
        self.le = LabelEncoder().fit(go_src + go_des)

        # created directed hierarchical edges
        src = self.le.transform(go_src)
        des = self.le.transform(go_des)
        self.g = dgl.graph((src, des))
        go_hier_elabels = [self.GO_HIER_MAP[l] for l in go_hier_df['edge_type'].tolist()]

        # splitting dataset
        train_mask = torch.zeros(len(go_hier_elabels), dtype=torch.bool).bernoulli(0.7)  # 70%
        #for idx in range(len(go_hier_elabels)):
            #if go_hier_elabels[idx] == 2:
                #train_mask[idx] = False  # the reverse of is_a and part_of relation is masked during training
        
        test_mask = ~train_mask
        #for idx in range(len(go_hier_elabels)):
            #if go_hier_elabels[idx] == 2:
                #test_mask[idx] = False  # the reverse of is_a and part_of relation is masked during training                

        self.g.edata['label'] = torch.tensor(go_hier_elabels, dtype=torch.long)  # edge label
        self.g.edata['train_mask'] = train_mask
        self.g.edata['test_mask'] = test_mask


    def forward(self, max_length=16):


        self.g.ndata['feature'] = ExtractNodeFeature(self.g, self.le, max_length=max_length).forward()
        # torch.randn(self.g.num_nodes(), 16) # alternative random intialization
        return self.g

    def get_node_encoder(self):
        return self.le


class ExtractNodeFeature:
    def __init__(self, g, le, max_length=16):
        nodes = le.inverse_transform(g.nodes())
        print("extracting node text attribute")

        # making lookup table go_dict[go id] -> go term
        go_df = pd.read_csv(go_info).drop_duplicates()
        go_dict = go_df.set_index('go_id').T.to_dict('list')
        node_features = []
        for n in nodes:
            try:
                go_term = go_dict[n][0]
            except:
                print(f'{go_term} cannot be found in go_info.csv, please retrieve the GOid and GO term from QuickGO')
                continue

                # encode
            node_features.append(' '.join((n, go_term)))
                #node_features.append((go_term))

        # initialising encoding of input textural attribute
        self.node_encodings = torch.tensor(Encode(node_features, max_length=max_length).forward()["input_ids"],
                                           dtype=torch.float)
        print('Shape of intial GO node embedding:', self.node_encodings.shape)
        # self.g.ndata['feature'] = node_encodings

    def forward(self):
        return self.node_encodings

class Encode:
    def __init__(self, x, max_length=16):
        print("encoding text attribute")
        self.tokenizers = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.encodings = self.tokenizers(x,
                                         max_length=max_length,
                                         padding='max_length',
                                         truncation=True)
    def forward(self):
        # dataset = DatasetBert(self.encodings, self.labels)
        return self.encodings


