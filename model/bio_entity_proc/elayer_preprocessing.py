import torch
import pickle
from transformers import AutoTokenizer, BertConfig
from corpus_path import go_info, gene_info, generif_info, abstract_info, num_cls
from transformers import Trainer, TrainingArguments, AdamW, get_linear_schedule_with_warmup
from entity_layer import EntityTagModelForSequenceClassification, EntityEncoder


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

    def __init__(self, goa_record, bg=False, generif=False, type='co'):
        print("integrating dataset")
        label_map = self.labels[type]
        goa_records = self.load_records(goa_record)

        go_index = pickle.load(open(go_info, 'rb'))
        gene_index = pickle.load(open(gene_info, 'rb'))

        entity_encoder = EntityEncoder()

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

        self.evidence_entity_encodings = [entity_encoder.encode_evidence(r.split('\t')[0], text) for r, text in
                                          list(zip(goa_records, self.evidences))]

        # extract gene info
        self.gene_symbols = [r.split('\t')[1] + ' ' + gene_index[r.split('\t')[1]] for r in goa_records]
        self.gene_symbol_encodings = [entity_encoder.encode_gene(gene) for gene in self.gene_symbols]
        # [gene_index[r.split('\t')[1]] for r in goa_records]

        self.labels = [label_map[r.split('\t')[-1]] for r in goa_records]

        # flat concatentation of GO terms, connecting with special markers [br]
        print("integrating gene ontology info")

        # using extra GO terms as supportive background knowledge
        if bg:
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
        self.go_term_encodings = [entity_encoder.encode_go(go_info) for go_info in self.go_terms]

    def forward(self):
        return list(
            zip(self.evidences, self.gene_symbols, self.go_terms)), self.labels

    def forward_entity_encoding(self):
        entity_tag_encodings = list(
            zip(self.evidence_entity_encodings,
                self.gene_symbol_encodings,
                self.go_term_encodings)
        )

        matrix = [[0, 0] + evidence + gene + [0] + go + [0] for evidence, gene, go in entity_tag_encodings]
        # [2, 36] for [CLS] [
        # [3] for [SEP]

        return matrix

    def load_records(self, file):
        records = set()
        with open(file, 'r') as fp:
            for r in fp:
                r = r.strip('\n')
                records.add(r)  # remove potential duplication
        return list(records)


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
        # print(gene_conditioned_evidence, go_info)
        return gene_conditioned_evidence, go_info

        # elif num_of_cls == 2:  # gene inconsistency
        # go_conditioned_evidence = evidence + ' ' + self.go_terms
        # print(go_conditioned_evidence, gene_info)
        # return go_conditioned_evidence, gene_info


class DatasetBert(torch.utils.data.Dataset):
    def __init__(self, encodings, entity_tag_encodings, labels):
        print("converting to BERT dataset format")
        self.encodings = encodings
        self.entity_tag_encodings = entity_tag_encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['entity_tag_ids'] = self.entity_tag_encodings[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Encode:
    def __init__(self, x, entity_x):
        print("encoding text")
        padding_length = 500
        self.tokenizers = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.encodings = self.tokenizers([Preprocessing(*i).forward() for i in x],
                                         max_length=padding_length,
                                         padding=True,
                                         truncation=True)

        self.entity_encodings = torch.tensor([x + [0] * (padding_length - len(x))
                                              if len(x) < padding_length else x[:padding_length]
                                              for x in entity_x])

        # self.labels = labels

    def forward(self):
        # dataset = DatasetBert(self.encodings, self.labels)
        return self.encodings, self.entity_encodings

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

    def mask_by_mention(self, evidence, go, alpha=0.7):
        """
        mask out GO term mentioned as keywords within evidence text
        :param evidence: str: evidence text
        :param go: str: go term text
        :param alpha: default 1, fraction of evidence text that need to be masked
        :return: go mention masked evidence and go
        """
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        go_tokens = set(self.tokenizers.tokenize(go))
        evidence_tokens = self.tokenizers.tokenize(evidence)
        overlap_indice = []
        for idx in range(len(evidence_tokens)):
            if evidence_tokens[idx] in go_tokens:
                overlap_indice.append(idx)
                # evidence_tokens[idx] = ''

        # mask output overlaps
        for i in range(int(len(overlap_indice) * alpha // 1)):
            evidence_tokens[overlap_indice[i]] = ''

        evidence = ' '.join(evidence_tokens).replace(' ##', '')
        return evidence, go


if __name__ == '__main__':
    """
    The main is implemented for code-testing.
    """

    train_path = './data/synthesized/temp/ensemble/train_os.txt'
    go_info = './data/go_info/go_lookup.pkl'
    gene_info = './data/gene_info/gene_lookup.pkl'
    abstract_info = './data/abstract_lookup.pkl'

    # train dataset
    dataset_integrator = Integration(train_path, bg=False, generif=False, type='os')  # integrate text attributes

    train_x, train_labels = dataset_integrator.forward()
    entity_tag_encodings = dataset_integrator.forward_entity_encoding()
    train_encodings, entity_tag_encodings = Encode(train_x, entity_tag_encodings).forward()  # encode text
    # train_encodings.get_encoding()
    # train_encodings.demo_decode()  # show example of encoded input
    # train_dataset = DatasetBert(train_encodings.forward(), train_labels)  # formatting dataset
    """tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    train_x = [("Test code x", "gene", "term"), ("testing with i", "protein", "term terminal")]
    train_labels = [1, 0]
    train_encodings = Encode(train_x).forward()
    print(train_encodings.items())
    train_entity_tag_encodings = torch.tensor([[0, 0, 0, 3, 2, 3, 0, 0, 0, 0], [0, 0, 3, 2, 3, 3, 0, 0, 0, 0]])"""
    train_dataset = DatasetBert(train_encodings, entity_tag_encodings, train_labels)

    model = EntityTagModelForSequenceClassification. \
        from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_labels=2)

    # set training args
    training_args = TrainingArguments(
        output_dir='./checkpoints',  # output directory
        do_train=True,  # run training
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=1,  # batch size per device during training
    )
    # create BERT fine-tuner
    entity_weight = ['bert.embeddings.entity_tag_embeddings.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in entity_weight],
         'lr': 2e-5},
        {'params': [p for n, p in model.named_parameters() if n in entity_weight],
         'lr': 5e-5}
    ]
    optim = AdamW(optimizer_grouped_parameters, lr=1e-5)
    lr_scheduler = get_linear_schedule_with_warmup(optim,
                                                   num_warmup_steps=training_args.warmup_steps,
                                                   num_training_steps=training_args.max_steps)
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        optimizers=(optim, lr_scheduler)
    )

    # begin fine-tuning
    trainer.train()
