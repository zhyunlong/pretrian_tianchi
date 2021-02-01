from dataset import Mydataset,data_collator,OCNLI_dataset
from transformers import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

label_dict = {"0":0, "1":1, "2":2}

train = OCNLI_dataset("data/OCNLI_train1128.csv", label_dict)
eval = OCNLI_dataset("data/OCNLI_train1128.csv", label_dict)


model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=len(label_dict))
training_args = TrainingArguments(
    output_dir='exp/OCNLI/model',          # output directory
    num_train_epochs=25,              # total # of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_steps=5000,
    eval_steps=2000,
    logging_dir='exp/OCNLI/logs',            # directory for storing logs
    evaluation_strategy='steps',
    #load_best_model_at_end=True,
    #metric_for_best_model="marco_f1_score",
    #prediction_loss_only=True,
    do_eval=True,
)
logging.info(model)

from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix,classification_report

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    labels = labels.flatten()
    preds = preds.flatten()
    marco_f1_score = f1_score(labels, preds, average='macro')
    logging.info(marco_f1_score)
    logging.info(f"{'confusion_matrix':*^80}")
    logging.info(confusion_matrix(labels, preds, ))
    logging.info(f"{'classification_report':*^80}")
    logging.info(classification_report(labels, preds, ))
    res = {"marco_f1_score":marco_f1_score}
    return res


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train,         # training dataset
    eval_dataset=eval,          # evaluation dataset
    data_collator= data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()

