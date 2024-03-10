import numpy as np
from sklearn import metrics
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification
from corpus_path import has_bg, use_generif, model_path, num_cls, suffix
from preprocessing import Integration, Encode, DatasetBert
import sys
import wandb
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def type_mapper(path):
    if 'co' in path:
        return 'co'
    elif 'os' in path:
        return 'os'
    elif 'ob' in path:
        return 'ob'
    elif '_im' in path:
        return 'im'
    elif 'ig'in path:
        return 'ig'
    else:
        return 'mclass'

def slim_rich(path):
    if 'slim' in path:
        return 'slim'
    elif 'rich' in path:
        return 'rich'
    else:
        return ''

def evaluation_summary(y_pred, y_test):
    """
    summary of accuracy, macro presicion,
    recall, f1 score
    """
    print("Accuracy:")
    print(metrics.accuracy_score(y_test, y_pred))

    print("\n Macro Average precision:")
    print(metrics.precision_score(y_test, y_pred, average='macro'))

    print("\n Macro Average recall:")
    print(metrics.recall_score(y_test, y_pred, average='macro'))

    print("\n Macro Average f1:")
    print(metrics.f1_score(y_test, y_pred, average='macro'))

    print("\n Classification report:")
    print(metrics.classification_report(y_test, y_pred))

# define metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'micro_f1': micro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall
    }


if __name__ == '__main__':
    test_path = sys.argv[1]
    model_path += type_mapper(test_path) + suffix
    sr = slim_rich(test_path)
    print(f'{model_path} model is loaded')

    test_dataset_integrator = Integration(test_path, bg=has_bg, generif=use_generif)
    test_x, test_labels = test_dataset_integrator.forward()
    #entity_tag_encodings = test_dataset_integrator.forward_entity_encoding()
    test_encodings = Encode(test_x).forward()
    test_dataset = DatasetBert(test_encodings, test_labels)
    print(f'{test_dataset.__len__()} instances in training set')

    """test_x, test_labels = Integration(test_path, bg=has_bg, generif=use_generif).forward()  # integrate text attributes

    test_encodings = Encode(test_x)  # encode text
    # test_encodings.decode()
    test_encodings.decode()
    test_encodings.demo_decode()  # show example of encoded input
    test_dataset = DatasetBert(test_encodings.forward(), test_labels)  # formatting dataset
    print(f'{test_dataset.__len__()} instances in test set')
    """
    # load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(f"{model_path}", num_labels=num_cls)

    wandb.init(project="bioc-eval", entity="jiyuc", name=model_path+sr)
    # set test args
    test_args = TrainingArguments(
        output_dir='./output/checkpoints',  # output directory
        per_device_eval_batch_size=500,  # batch size for evaluation
        run_name=model_path,
        do_predict=True,
        report_to="wandb"
    )

    # create BERT fine-tuner
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=test_args,  # training arguments, defined above
        compute_metrics=compute_metrics,
    )

    # begin evaluation
    y_pred = trainer.predict(test_dataset=test_dataset)
    wandb.log(y_pred.metrics)
    preds = np.argmax(y_pred.predictions, axis=1)
    #if num_of_cls == 4:
        #compute_auc(test_labels, y_pred.predictions)
    evaluation_summary(preds, test_labels)
    #wandb.log(evaluation_summary(preds, test_labels))

    # write predicted class to txt file
    with open(f'{model_path+sr}.txt', 'w') as f:
        f.write('\n'.join(str(x) for x in preds))
        print(f'predictions saved to {model_path+sr}.txt')
    f.close()
    wandb.save(f'{model_path+sr}.txt')

