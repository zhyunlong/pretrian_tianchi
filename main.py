from dataset import Mydataset,data_collator
from transformers import *

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "None"


task_name = ["OCEMOTION", "OCNLI", "TNEWS"]
label_len = [3,7,17]
task_label_dict = {}
task_label_dict["OCEMOTION"] = {0:"100", }
train = Mydataset(path="OCEMOTION_train1128.csv")


label_dict = {0:"O", 1:"，", 2:"。", 3:"：",  4:"、"}
model = BertForSequenceClassification.from_pretrained("bert-base-chinese")
training_args = TrainingArguments(
    output_dir='exp/bert_base/model',          # output directory
    num_train_epochs=25,              # total # of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    save_steps=20000,
    eval_steps=5,
    logging_dir='exp/bert_base/logs',            # directory for storing logs
    evaluation_strategy='steps',
    #prediction_loss_only=True,
    do_eval=True,
)
logging.info(model)

from sklearn.metrics import precision_recall_fscore_support


trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train,         # training dataset
    eval_dataset=eval,          # evaluation dataset
    data_collator= data_collator,
    #compute_metrics=compute_metrics,
)


trainer.train()

