import torch
import pickle
from transformers import AutoTokenizer, BertConfig
from corpus_path import go_info, gene_info, generif_info, abstract_info, num_cls, go_hier
from transformers import Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
from joint import JointModelForSequenceClassification
from train_graph import Model, MakeGraph
import pandas as pd
from go_graph import MakeGODAG, GOKGModel


class Integration:
    if num_cls == 2:
        labels = {
            'co': {'co': 1, 'os': 0, 'ob': 0, 'im': 0, 'ig': 0},
            'os': {'co': 0, 'os': 1, 'ob': 0, 'im': 0, 'ig': 0},
            'ob': {'co': 0, 'os': 0, 'ob': 1, 'im': 0, 'ig': 0},
            'im': {'co': 0, 'os': 0, 'ob': 0, 'im': 1, 'ig': 0},
            'ig': {'co': 0, 'os': 0, 'ob': 0, 'im': 0, 'ig': 1}
        }
    else:
        labels = {
            'co': {'co': 0, 'os': 1, 'ob': 2, 'im': 3, 'ig': 4},
            'os': {'co': 0, 'os': 1, 'ob': 2, 'im': 3, 'ig': 4},
            'ob': {'co': 0, 'os': 1, 'ob': 2, 'im': 3, 'ig': 4},
            'im': {'co': 0, 'os': 1, 'ob': 2, 'im': 3, 'ig': 4},
            'ig': {'co': 0, 'os': 1, 'ob': 2, 'im': 3, 'ig': 4},
            'mclass': {'co': 0, 'os': 1, 'ob': 2, 'im': 3, 'ig': 4},
        }

    def __init__(self, goa_record, bg=False, generif=False, type='co', flag='train'):
        print("integrating dataset")
        label_map = self.labels[type]
        goa_records = self.load_records(goa_record)

        go_index = pickle.load(open(go_info, 'rb'))
        gene_index = pickle.load(open(gene_info, 'rb'))

        # load evidence
        if generif:  # generif as evidence
            generif_index = pickle.load(open(generif_info, 'rb'))
        else:  # title/abstract as evidence
            generif_index = pickle.load(open(abstract_info, 'rb'))

        # extract evidence
        if generif:  # generif as evidence
            self.evidences = [generif_index[(r.split('\t')[0], r.split('\t')[1])] for r in goa_records]
        else:  # title/abstract as evidence
            self.evidences = [generif_index[r.split('\t')[0]] for r in goa_records]

        # extract gene info
        self.gene_symbols = [r.split('\t')[1] + ' ' + gene_index[r.split('\t')[1]] for r in goa_records]
        # [gene_index[r.split('\t')[1]] for r in goa_records]

        self.labels = [label_map[r.split('\t')[-1]] for r in goa_records]

        # flat concatentation of GO terms
        print("integrating gene ontology info")

        # using extra GO terms as supportive background knowledge
        if bg:

            # extract node embedding from pre-trained GNN network
            node2embedding = GBGEmbedding(flag=flag).encode(goa_record)
            gohier2embedding = GODAGEmbedding().encode(go_hier)
            go_node_embedding = torch.stack([node2embedding['go_id'][r.split('\t')[-2]] for r in goa_records])
            temp = []
            for r in goa_records:
                go_id = r.split('\t')[-2]
                try:
                    temp.append(gohier2embedding[go_id])
                except KeyError:
                    temp.append(torch.zeros(gohier2embedding['GO:0000001'].shape))
            #go_hier_embedding = torch.stack(temp)
            go_hier_embedding = torch.stack([gohier2embedding[r.split('\t')[-2]] for r in goa_records])
            pmid_gene_node_embedding = torch.stack([node2embedding['pmid_gene'][r.split('\t')[0]+'_'+r.split('\t')[1]] for r in goa_records])

            print(f'shape of encoded go node {go_node_embedding.shape}')
            self.node_embedding = torch.cat([go_node_embedding, pmid_gene_node_embedding, go_hier_embedding], dim=1)

            print('shape of node embedding')
            print(self.node_embedding.shape)

            self.go_terms = [' [SEP] '.join([r.split('\t')[-2] + ' ' + go_index[r.split('\t')[-2]]] +
                                            [v + ' ' + go_index[v] for v in r.split('\t')[2:-2] if v])
                             for r in goa_records]
        else:
            # no extra GO term as supportive background knowledge
            try:
                self.go_terms = [r.split('\t')[-2] + ' ' + go_index[r.split('\t')[-2]] + '[MASK]' for r in goa_records]
            except KeyError:
                for r in goa_records:
                    try:
                        r.split('\t')[-2] + ' ' + go_index[r.split('\t')[-2]]
                    except KeyError:
                        print(r)



    def forward(self):
        return list(
            zip(self.evidences, self.gene_symbols, self.go_terms)), self.labels, self.node_embedding

    def load_records(self, file):
        records = set()
        with open(file, 'r') as fp:
            for r in fp:
                r = r.strip('\n')
                records.add(r)  # remove potential duplication
        return list(records)

class GODAGEmbedding:

    def __init__(self, model_path='./data/go_hier_16dim.pt'):
        self.model_path = model_path
        self.node_embedding = dict()


    def encode(self, go_hier='./data/go_network.csv'):
        graph_maker = MakeGODAG(go_hier)
        g, go_le = graph_maker.forward(max_length=16)
        # le = graph_maker.get_node_encoder()
        print(g)
        node_features = g.ndata['feature']

        model = GOKGModel(16, 32, 16, 2)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        go_node_embeddings = model.sage(g, node_features)
        print(go_node_embeddings.shape)
        go_node_ids = go_le.inverse_transform([int(i) for i in g.nodes()])
        print(len(go_node_ids))
        #node_embeddings = torch.cat([go_node_embeddings, pmid_gene_node_embeddings])
        #vertices_id = go_node_ids + pmid_gene_node_ids

        self.build_id2embedding_dict(go_node_ids, go_node_embeddings)
        return self.node_embedding

    def build_id2embedding_dict(self, node_ids, embeddings):
        singleton_count = 0
        for node_id, embed in list(zip(node_ids, embeddings)):
            self.node_embedding[node_id] = embed.clone().detach()




class GBGEmbedding:

    def __init__(self, flag, model_path='./data/gnn_model.pt'):
        self.flag = flag
        self.model_path = model_path
        self.node_embedding = {'go_id':{}, 'pmid_gene':{}}


    def encode(self, file):
        print(f'build goa bg graph from {file}')
        Pipeline().build_negative_edges(file, self.flag)
        g, go_le, pmid_gene_le = MakeGraph(f'./data/{self.flag}_positives.csv', f'./data/{self.flag}_negatives.csv').forward()

        go_feats = g.nodes['go_id'].data['h']
        pmid_gene_feats = g.nodes['pmid_gene'].data['h']
        node_features = {'go_id': go_feats, 'pmid_gene': pmid_gene_feats}

        print(f'load pre-trained bg embedding from {self.model_path}')
        model = Model(8, 16, 16, 8, g.etypes)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()



        go_node_embeddings = model.sage(g, node_features)['go_id']
        print(go_node_embeddings.shape)
        go_node_ids = go_le.inverse_transform([int(i) for i in g.nodes('go_id')])
        pmid_gene_node_embeddings = model.sage(g, node_features)['pmid_gene']
        pmid_gene_node_ids = pmid_gene_le.inverse_transform([int(i) for i in g.nodes('pmid_gene')])

        #node_embeddings = torch.cat([go_node_embeddings, pmid_gene_node_embeddings])
        #vertices_id = go_node_ids + pmid_gene_node_ids

        self.build_id2embedding_dict('go_id',go_node_ids, go_node_embeddings)
        self.build_id2embedding_dict('pmid_gene',pmid_gene_node_ids, pmid_gene_node_embeddings)
        return self.node_embedding

    def build_id2embedding_dict(self, node_type, node_ids, embeddings):
        for node_id, embed in list(zip(node_ids, embeddings)):
            self.node_embedding[node_type][node_id] = embed.clone().detach()


class Preprocessing:
    def __init__(self, evidence, gene_symbol, go_terms):
        self.evidence = str(evidence)
        self.gene_symbol = str(gene_symbol)
        self.go_terms = str(go_terms)
        if not self.go_terms:
            print(self.evidence, self.gene_symbol)

    def forward(self):
        """
        extract text attribute for each GOA annotation
        :return: tuple of (gene_conditioned_evidence, go_info)
        """

        # flat concatenation with linguistic marker
        evidence = self.evidence
        gene_info = '[SEP] ' + self.gene_symbol

        # if num_of_cls == 4:  # term inconsistency
        gene_conditioned_evidence = evidence + ' ' + gene_info
        go_info = self.go_terms
        return gene_conditioned_evidence, go_info


class DatasetBert(torch.utils.data.Dataset):
    def __init__(self, encodings, node_embedding, labels):
        print("converting to BERT dataset format")
        self.encodings = encodings
        self.node_embedding = node_embedding
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['node_embedding'] = self.node_embedding[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Encode:
    def __init__(self, x):
        print("encoding text")
        padding_length = 500
        self.tokenizers = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.encodings = self.tokenizers([Preprocessing(*i).forward() for i in x],
                                         max_length=padding_length,
                                         padding=True,
                                         truncation=True)
        # self.labels = labels

    def forward(self):
        # dataset = DatasetBert(self.encodings, self.labels)
        return self.encodings

    def decode(self):
        curr_max = 0
        for ids in self.encodings["input_ids"]:
            curr_max = max(curr_max, len([x for x in self.tokenizers.decode(ids).split(' ') if x != '[PAD]']))

        print(f'the maximum input length after encoding: {curr_max}')

    def demo_decode(self):
        for ids in self.encodings["input_ids"]:
            print([x for x in self.tokenizers.decode(ids).split(' ')])
            break

    def get_encoding(self):
        for i, v in self.encodings.items():
            print(i, v[0])


def load_pmids(filename):
    """
    find pmid_gene_goid in the synthesized data set
    :param filename: filename of test.txt
    :return: list of pmid_gene pairs as string type
    """
    contents = [l.split('\t') for l in open(filename, 'r').readlines()]
    pmids = set(r[0] for r in contents)
    return pmids


def no_pmid(x):
    """
    Evaluate if an input GOA misses biolink to PMID;
    Return True if bio link to PMID is missing. Otherwise, return False.
    """
    if x.isdigit():
        return False
    return True


class GOA:
    def __init__(self):
        self.record = []

    def __int__(self, records):
        self.record = records

    def insert(self, x):
        """
        insert a GOA record
        :return: None
        """
        self.record.append(x)

    def __len__(self):
        return len(self.record)

    def drop_duplication(self):
        self.record = list(set(self.record))

    def to_csv(self, filename):
        cols = ['PMID_GeneID', 'GO_ID', 'edge_type']
        filtered_annotations = pd.DataFrame(self.record, columns=cols)
        filtered_annotations = filtered_annotations.drop_duplicates()
        filtered_annotations.to_csv(filename, index=False)
        print(len(filtered_annotations))


class Pipeline:
    def __init__(self):
        return

    def build_negative_edges(self, filename, csv_flag):
        negatives = GOA()
        positives = GOA()
        contents = [l.strip().split('\t') for l in open(filename, 'r').readlines()]
        negatives.record = [[f'{r[0]}_{r[1]}', r[-2], 'IC'] for r in contents if r[-1] != 'co']
        positives.record = [[f'{r[0]}_{r[1]}', r[-2], 'CO'] for r in contents if r[-1] == 'co']
        for r in contents:
            #if r[-1] != 'co' and len(r[2:-2]):
                for go_id in r[2:-2]:
                    positives.insert([f'{r[0]}_{r[1]}', go_id, 'CO'])

        negatives.to_csv(f'./data/{csv_flag}_negatives.csv')
        positives.to_csv(f'./data/{csv_flag}_positives.csv')
        return
