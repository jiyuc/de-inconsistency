import torch
from transformers import AutoTokenizer
import pandas as pd
from corpus_path import go_text_attr
from sklearn.preprocessing import LabelEncoder
import dgl
import dgl.nn as dglnn
from torch import nn
import torch.nn.functional as F


class MakeGODAG:
    GO_HIER_MAP = {'children_of': 0,
                   'parent_of': 1,
                   }

    def __init__(self, go_hier_csv, flag='tune'):
        print("build GO DAG")
        # the go hierarchical relation is stored in go_network.csv file
        go_hier_df = pd.read_csv(go_hier_csv)
        go_src = go_hier_df['src'].tolist()
        go_des = go_hier_df['des'].tolist()

        # encode nodes to unique node id
        self.le = LabelEncoder().fit(go_src + go_des)

        # directional go hierarchy
        src = self.le.transform(go_src)
        des = self.le.transform(go_des)
        self.g = dgl.graph((src, des))
        # print(go_hier_df['edge_type'].tolist())
        go_hier_elabels = [self.GO_HIER_MAP[l] for l in go_hier_df['edge_type'].tolist()]
        if flag == 'train':  # only mask during pre-training
            # split dataset
            train_mask = torch.zeros(len(go_hier_elabels), dtype=torch.bool).bernoulli(0.7)  # 70%
            test_mask = ~train_mask
            self.g.edata['train_mask'] = train_mask
            self.g.edata['test_mask'] = test_mask
        self.g.edata['label'] = torch.tensor(go_hier_elabels, dtype=torch.long)  # edge label

    def forward(self, max_length=16):
        self.g.ndata['feature'] = ExtractNodeFeature(self.g, self.le, max_length=max_length).forward()
        return self.g, self.le


class ExtractNodeFeature:
    def __init__(self, g, le, max_length=16):
        nodes = le.inverse_transform(g.nodes())
        print("extracting node text attribute")

        # go term & definition -> go_dict[go id] = (go term, go definition)
        go_df = pd.read_csv(go_text_attr).drop_duplicates()
        go_dict = go_df.set_index('go_id').T.to_dict('list')
        """
        # gene symbol and gene description -> gene_dict[gene_id] = (gene symbol, gene description)
        gene_df = pd.read_csv(gene_info).drop_duplicates()
        gene_dict = gene_df.set_index('gene_id').T.to_dict('list')

        # evidence info -> textpr[pmid] = (title, abstract)
        textpr = ParsePassage()
        """
        # encode either PMIDgene info pair or go info pair with BERT tokenizer and encoder
        node_features = []
        for n in nodes:
            try:
                go_term = go_dict[n][0]
                # go_def = go_dict[n][1]
            except:
                print(n)
                continue
            node_features.append(' '.join((n, go_term)))

        # initial input textural features
        self.node_encodings = torch.tensor(Encode(node_features, max_length=max_length).forward()["input_ids"],
                                           dtype=torch.float)
        print('Shape of GO node feature:', self.node_encodings.shape)
        # self.g.ndata['feature'] = node_encodings

    def forward(self):
        return self.node_encodings


class Encode:
    def __init__(self, x, max_length=16):
        print("encoding text")
        self.tokenizers = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

        self.encodings = self.tokenizers(x, max_length=max_length, padding='max_length', truncation=True)
        # self.labels = labels

    def forward(self):
        return self.encodings


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='pool')

        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='pool')

        self.dropout_1 = nn.Dropout(p=0.2)
        self.dropout_2 = nn.Dropout(p=0.2)

    def forward(self, graph, inputs):
        # inputs are features of nodes
        # first layer
        h = self.conv1(graph, inputs)
        h = F.relu(h)  # activation of 1st layer
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)  # normalisation
        h = self.dropout_1(h)

        # second layer
        h = self.conv2(graph, h)
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)
        # h = self.dropout_2(h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(self.dropout(torch.cat([h_u, h_v], 1)))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class GOKGModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_class):
        super().__init__()
        # node embedding
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.h = in_features
        # edge classification
        self.pred = MLPPredictor(out_features, num_class)

    def forward(self, g, x):
        self.h = self.sage(g, x)
        return self.pred(g, self.h)

    def get_node_representation(self, g, x):
        return self.sage(g, x)
