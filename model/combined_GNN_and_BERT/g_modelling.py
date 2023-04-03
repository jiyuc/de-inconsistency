from transformers import Trainer, TrainingArguments
from joint import JointModelForSequenceClassification
from graph_preprocessing import Integration, Encode, DatasetBert
from corpus_path import model_path, has_bg, use_generif, num_cls, suffix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import sys
import wandb


def type_mapper(path):
    if 'co' in path:
        return 'co'
    elif 'os' in path:
        return 'os'
    elif 'ob' in path:
        return 'ob'
    elif 'im' in path:
        return 'im'
    elif 'ig' in path:
        return 'ig'
    else:
        return 'mclass'


def sin_mul(path):
    if 'slim' in path:
        return 'slim'
    elif 'rich' in path:
        return 'rich'
    else:
        return ''


if __name__ == '__main__':
    train_path, dev_path = sys.argv[1], sys.argv[2]
    model_path += type_mapper(train_path) + suffix
    print(model_path)

    # train dataset
    train_dataset_integrator = Integration(train_path, bg=has_bg, generif=use_generif, type=type_mapper(train_path))
    train_x, train_labels, train_node_embedding = train_dataset_integrator.forward()
    train_encodings = Encode(train_x).forward()
    train_dataset = DatasetBert(train_encodings, train_node_embedding, train_labels)
    print(f'{train_dataset.__len__()} instances in training set')

    # dev dataset
    dev_dataset_integrator = Integration(dev_path, bg=has_bg, generif=use_generif, type=type_mapper(dev_path), flag='dev')
    dev_x, dev_labels, dev_node_embedding = dev_dataset_integrator.forward()
    dev_encodings = Encode(dev_x).forward()
    dev_dataset = DatasetBert(dev_encodings, dev_node_embedding, dev_labels)
    print(f'{dev_dataset.__len__()} instances in dev set')

    # load pre_trained model
    model = JointModelForSequenceClassification.\
        from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_labels=num_cls)
    # "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    # define metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        """log = open(f'{model_path}_metrics.txt', 'w')
        log.write('\t'.join(['p','r','f1','acc\n']))
        with open(f'{model_path}_metrics.txt', 'a') as f:
            f.write('\t'.join((str(precision), str(recall), str(f1), str(acc))) + '\n')
        f.close()"""
        return {
            'accuracy': acc,
            'micro_f1': micro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'macro_f1': macro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall
        }

    wandb.init(project="bioc-1", entity="jiyuc", name=model_path)

    # set training args
    training_args = TrainingArguments(
        output_dir='./output/checkpoints',  # output directory
        run_name=model_path,
        do_train=True,
        overwrite_output_dir=True,
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=300,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,
        #save_steps=300,
        #eval_steps=300,
        save_strategy="steps",
        logging_steps=100,
        evaluation_strategy="steps",  # evaluate at the end of steps
        load_best_model_at_end=True,
        report_to="wandb"
    )

    # create BERT fine-tuner
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=dev_dataset,  # validation dataset
        compute_metrics = compute_metrics,  # evaluation score
    )

    # begin fine-tuning
    trainer.train()
    trainer.save_model(model_path)
