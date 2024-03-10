import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import MakeGODAG
from corpus_path import go_hier
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()

        # 1st hidden layer
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='pool')

        # 2nd layer
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='pool')

        # dropout
        self.dropout_1 = nn.Dropout(p=0.2)


    def forward(self, graph, inputs):
        # inputs are features of nodes

        # first layer
        h = self.conv1(graph, inputs)
        # batch normalisation before activation
        # reference: https://arxiv.org/pdf/1502.03167.pdf
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)
        h = F.relu(h)  # activation of 1st layer
        self.dropout_1 = nn.Dropout(p=0.2)

        # second layer
        h = self.conv2(graph, h)
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)  # batch normalisation
        h = F.relu(h)  # activation of 2nd layer
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.dropout = nn.Dropout(p=0.2)
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


class GOKGModel(nn.Module):
    def __init__(self, in_features=16, hidden_features=32, out_features=16, num_class=2):
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

if __name__ == '__main__':
    graph_maker = MakeGODAG(go_hier)
    g = graph_maker.forward(max_length=16)
    # le = graph_maker.get_node_encoder()
    print(g)
    node_features = g.ndata['feature']
    edge_label = g.edata['label']
    train_mask = g.edata['train_mask']
    losses = []
    model = GOKGModel(16, 32, 16, 2)

    # train
    """
    decoment this block to re-train node embedding
    opt = torch.optim.Adam(model.parameters())
    for epoch in tqdm(range(1000)):
        model.train()
        logits = model(g, node_features)
        loss = F.cross_entropy(logits[train_mask], edge_label[train_mask])
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    plt.plot(losses)
    plt.savefig('out/training_loss.pdf')
    plt.show()
    """

    # saving the pre-trained GO specificity embedding
    torch.save(model.state_dict(), 'out/go_hier_16dim.pt')
    test_mask = g.edata['test_mask']

    # evaluation
    model.load_state_dict(torch.load('out/go_hier_16dim.pt'))
    model.eval()
    with torch.no_grad():
        logits = model(g, node_features)
        logits = logits[test_mask]
        labels = edge_label[test_mask]
        _, indices = torch.max(logits, dim=1)
        y_test = [int(x) for x in labels]
        y_pred = [int(x) for x in indices]
        evaluation_summary(y_test, y_pred)

