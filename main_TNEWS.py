from dataset import Mydataset,data_collator
from transformers import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "None"

label_dict = {}

for i in range(17):
    if i<10:
        label_dict["10%d"%i] = i
    else:
        label_dict["1%d"%i] = i
train = Mydataset("data/TNEWS_train1128.csv", label_dict)
eval = Mydataset("data/TNEWS_train1128.csv", label_dict)


model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=len(label_dict))
training_args = TrainingArguments(
    output_dir='exp/TNEWS/model',          # output directory
    num_train_epochs=25,              # total # of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_steps=20000,
    eval_steps=5,
    logging_dir='exp/TNEWS/logs',            # directory for storing logs
    evaluation_strategy='steps',
    #prediction_loss_only=True,
    do_eval=True,
)
logging.info(model)

from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix,classification_report

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    labels = labels.flatten()
    preds = preds.flatten()
    marco_f1_score = f1_score(labels, preds, average='macro')
    print(marco_f1_score)
    print(f"{'confusion_matrix':*^80}")
    print(confusion_matrix(labels, preds, ))
    print(f"{'classification_report':*^80}")
    print(classification_report(labels, preds, ))


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train,         # training dataset
    eval_dataset=eval,          # evaluation dataset
    data_collator= data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()

