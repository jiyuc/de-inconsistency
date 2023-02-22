import pandas as pd

import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
import dgl.function as fn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


"""class Model(nn.Module):
    def __init__(self, in_features, hidden_features_1, hidden_features_2, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features_1, hidden_features_2, out_features, rel_names)
        self.pred = HeteroMLPPredictor(out_features)

    def forward(self, g, neg_g, x, etype):
        h = self.sage(g, x)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype)"""


class Model(nn.Module):
    def __init__(self, in_features, hidden_features_1, hidden_features_2, out_features, rel_names):
        super().__init__()
        self.sage = RGCN(in_features, hidden_features_1, hidden_features_2, out_features, rel_names)
        self.pred = HeteroMLPPredictor(out_features, len(rel_names))  # exist or not

    def forward(self, g, x, dec_graph):
        h = self.sage(g, x)
        return self.pred(dec_graph, h)


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W = nn.Linear(in_dims * 2, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = self.W(x)
        return {'score': y}

    def forward(self, graph, h):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


# Define a Heterograph Conv model
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats_1, hid_feats_2, out_feats, rel_names):
        super().__init__()

        # three GraphConv layer
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_feats, hid_feats_1, aggregator_type='pool')
            for rel in rel_names}, 'mean')


        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats_1, hid_feats_2, aggregator_type='pool')
            for rel in rel_names}, 'mean')

        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats_2, out_feats, aggregator_type='pool')
            for rel in rel_names}, 'mean')

        """self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hid_feats_2, out_feats, aggregator_type='pool')
            for rel in rel_names}, 'sum')"""

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)  # 1st layer output
        h = {k: F.relu(v) for k, v in h.items()}  # activation
        h = self.conv2(graph, h)  # 2nd layer output
        h = {k: F.relu(v) for k, v in h.items()}  # activation
        h = self.conv3(graph, h)
        return h


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats_1, hid_feats_2, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats_1, aggregator_type='pool')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats_1, out_feats=hid_feats_2, aggregator_type='pool')
        self.conv3 = dglnn.SAGEConv(
            in_feats=hid_feats_2, out_feats=out_feats, aggregator_type='pool')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        # batch normalisation before activation, reference: https://arxiv.org/pdf/1502.03167.pdf
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)
        h = F.relu(h)

        # second layer
        h = self.conv2(graph, h)
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)
        h = F.relu(h)

        # output layer
        h = self.conv3(graph, h)
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)  # batch normalisation
        h = F.relu(h)  # activation of 3rd layer
        return h

class LabelEncoder:
    def __init__(self):
        self.label2num = dict()
        self.num2label = dict()
        pass
        
    def fit(self, items):
        count = 0
        for item in items:
            if item not in self.label2num:
                self.label2num[item] = count
                self.num2label[count] = item
                count += 1
            continue

    def transform(self, items):
        return [self.label2num[i] for i in items]

    def inverse_transform(self, nums):
        return [self.num2label[i] for i in nums]

class MakeGraph:
    def __init__(self, positive_csv, negative_csv, is_train=False):
        print("making GOA graph")
        pos = pd.read_csv(open(positive_csv,'r'), dtype=str)
        neg = pd.read_csv(open(negative_csv,'r'), dtype=str)

        # make positive heterograph with 2 node types and 1 edge type.
        g, self.go_le, self.pmid_gene_le = self.make_graph(pos, neg, is_train)
        self.g = self.node_encoder(g)

        #self.neg_g = dgl.to_homogeneous(self.make_graph(neg))
        #self.neg_g.ndata['h'] = torch.ones(self.neg_g.num_nodes(), 3)

        # make negative heterograph with 2 node types and 1 edge type
        #print(self.pos_g)
        #print(self.neg_g)


    def make_graph(self, pos_csv, neg_csv, is_train=False):
        # Create heterograph with 2 node types and undirected edge.
        go_nodes = pos_csv['GO_ID'].tolist() + neg_csv['GO_ID'].tolist()
        pmid_gene_nodes = pos_csv['PMID_GeneID'].tolist() + neg_csv['PMID_GeneID'].tolist()

        # encode node ids
        go_le, pmid_gene_le = LabelEncoder(), LabelEncoder()
        go_le.fit(go_nodes)
        pmid_gene_le.fit(pmid_gene_nodes)

        # encode
        pos_src = go_le.transform(pos_csv['GO_ID'].tolist())
        pos_des = pmid_gene_le.transform(pos_csv['PMID_GeneID'].tolist())
        neg_src = go_le.transform(neg_csv['GO_ID'].tolist())
        neg_des = pmid_gene_le.transform(neg_csv['PMID_GeneID'].tolist())
        print(len(pos_src), len(pos_des))
        print(len(neg_src), len(neg_des))

        print(len(set(pos_src+neg_src)))

        """g = dgl.graph((u,v))
        edges = ([1] * len(pos_csv['GO_ID'].tolist())
                + [0] * len(neg_csv['GO_ID'].tolist())) * 2  # edge types
        g.edata['label'] = torch.tensor(edges, dtype=torch.long)
        g.ndata['h'] = torch.ones(g.num_nodes(), 6)

        # mask edges
        train_mask = torch.zeros(len(edges), dtype=torch.bool).bernoulli(0.7)  # 70%
        test_mask = ~train_mask
        g.edata['train_mask'] = train_mask
        g.edata['test_mask'] = test_mask
        return g"""
        #return dgl.graph((u,v))
        

        graph_data = {
            ('go_id', 'annotate', 'pmid_gene'): (pos_src, pos_des),  # go_id -> pmid_gene
            ('pmid_gene', 'rev', 'go_id'): (pos_des, pos_src),  # pmid_gene -> go_id
            ('go_id', 'not', 'pmid_gene'): (neg_src, neg_des),  # go_id -> pmid_gene
            ('pmid_gene', 'rev', 'go_id'): (neg_des, neg_src)  # pmid_gene -> go_id
        }

        """neg_graph_data = {
            ('go_id', 'NOT', 'pmid_gene'): (neg_src, neg_des),  # go_id -> pmid_gene
            ('pmid_gene', 'NOT_by', 'go_id'): (neg_des, neg_src)  # pmid_gene -> go_id
        }"""

        return dgl.heterograph(graph_data), go_le, pmid_gene_le  #, dgl.heterograph(neg_graph_data)

    def node_encoder(self, g):
        # initialise node features in 16dim vector
        g.nodes['go_id'].data['h'] = torch.randn(g.num_nodes('go_id'), 8)
        g.nodes['pmid_gene'].data['h'] = torch.ones(g.num_nodes('pmid_gene'), 8)
        return g

    def forward(self):
        return self.g, self.go_le, self.pmid_gene_le


"""def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores, labels)"""

def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).detach().numpy()
    return roc_auc_score(labels, scores)

def construct_negative_graph(graph, k):
    src, dst = graph.edges()
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

def evaluation_summary(y_test, y_pred):
    """
    summary of accuracy, macro presicion,
    recall, f1 score
    """
    print("Accuracy:")
    print(metrics.accuracy_score(y_test, y_pred))

    print("\n Micro Average precision:")
    print(metrics.precision_score(y_test, y_pred, average='micro'))

    print("\n Micro Average recall:")
    print(metrics.recall_score(y_test, y_pred, average='micro'))

    print("\n Micro Average f1:")
    print(metrics.f1_score(y_test, y_pred, average='micro'))

    print("\n Classification report:")
    print(metrics.classification_report(y_test, y_pred))


def evaluate(model, g, node_features, dec_graph, labels):
    with torch.no_grad():
        logits = model(g, node_features, dec_graph)
        _, indices = torch.max(logits, dim=1)
        # print([x for x in indices])
        y_test = [int(x) for x in labels]
        y_pred = [int(x) for x in indices]
        evaluation_summary(y_test, y_pred)
        # correct = torch.sum(indices == labels)
        # return correct.item() * 1.0 / len(labels)



if __name__ == '__main__':

    g,_,_ = MakeGraph('./train_positives.csv', './train_negatives.csv',is_train=True).forward()
    dec_graph = g['go_id',:,'pmid_gene']
    edge_label = dec_graph.edata[dgl.ETYPE]
    model = Model(8, 16, 16, 8, g.etypes)
    go_feats = g.nodes['go_id'].data['h']
    pmid_gene_feats = g.nodes['pmid_gene'].data['h']
    node_features = {'go_id': go_feats, 'pmid_gene': pmid_gene_feats}
    opt = torch.optim.Adam(model.parameters())
    losses = list()
    for epoch in tqdm(range(300)):
        model.train()
        logits = model(g, node_features, dec_graph)
        g_loss = F.cross_entropy(logits, edge_label)  # compute_loss(pos_score, neg_score)
        opt.zero_grad()
        g_loss.backward()
        opt.step()
        losses.append(g_loss.item())
        #print(loss.item())
    plt.plot(losses)
    plt.show()
    model.eval()
    evaluate(model, g, node_features, dec_graph, edge_label)
    torch.save(model.state_dict(), './gnn_model.pt')


    g,_,_ = MakeGraph('./dev_positives.csv', './dev_negatives.csv').forward()
    dec_graph = g['go_id', :, 'pmid_gene']
    edge_label = dec_graph.edata[dgl.ETYPE]
    #model = Model(8, 16, 8, g.etypes)
    go_feats = g.nodes['go_id'].data['h']
    pmid_gene_feats = g.nodes['pmid_gene'].data['h']
    node_features = {'go_id': go_feats, 'pmid_gene': pmid_gene_feats}
    model = Model(8, 16, 16, 8, g.etypes)
    model.load_state_dict(torch.load('./gnn_model.pt'))
    model.eval()
    evaluate(model, g, node_features, dec_graph, edge_label)

    g,_,_ = MakeGraph('./test_positives.csv', './test_negatives.csv').forward()
    dec_graph = g['go_id', :, 'pmid_gene']
    edge_label = dec_graph.edata[dgl.ETYPE]
    # model = Model(8, 16, 8, g.etypes)
    go_feats = g.nodes['go_id'].data['h']
    pmid_gene_feats = g.nodes['pmid_gene'].data['h']
    node_features = {'go_id': go_feats, 'pmid_gene': pmid_gene_feats}
    model.eval()
    evaluate(model, g, node_features, dec_graph, edge_label)

    """node_features = train_graph.ndata['h']
    n_features = node_features.shape[1]
    print(n_features)
    model = Model(n_features, 10, 10, 3)
    model.train()
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(100):
        neg_graph = construct_negative_graph(train_graph, k=3)
        pos_score, neg_score = model(train_graph, neg_graph, node_features)
        print("pass")
        loss = compute_loss(pos_score, neg_score)
        print(compute_auc(pos_score, neg_score))
        opt.zero_grad()
        loss.backward()
        opt.step()
        #print(loss.item())"""
