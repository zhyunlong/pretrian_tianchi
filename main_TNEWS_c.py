from dataset import Mydataset,data_collator
from dataset_for_pretrain import PretrainDataset, pretrain_data_collator
from transformers import *
import os
import torch
import logging
from model import BertClassification
from utilst import *


def fine_tune_stage_1():
    label_dict = get_TNEWS_label_dict()
    train = Mydataset("data_split/TNEWS_train.csv", label_dict)
    eval = Mydataset("data_split/TNEWS_dev.csv", label_dict)
    model = BertClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=len(label_dict))
    training_args = TrainingArguments(
        output_dir='exp/TNEWS_stage1/model',          # output directory
        num_train_epochs=5,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=2,
        eval_steps=1000,
        learning_rate=1e-5,
        logging_dir='exp/TNEWS_stages1/logs',            # directory for storing logs
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model="marco_f1_score",
    )
    #logging.info(model)
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train,         # training dataset
        eval_dataset=eval,          # evaluation dataset
        data_collator= data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()


def fine_tune_stage_2():
    label_dict = get_TNEWS_label_dict()
    train = Mydataset("data_split/TNEWS_dev.csv", label_dict)
    eval = Mydataset("data_split/TNEWS_dev.csv", label_dict)
    checkpoint_dir = get_newest_checkpoit("TNEWS_stage1")
    model = BertClassification.from_pretrained(checkpoint_dir, num_labels=len(label_dict))
    training_args = TrainingArguments(
        output_dir='exp/TNEWS/model',          # output directory
        num_train_epochs=4,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=2,
        eval_steps=1000,
        learning_rate=1e-2,
        logging_dir='exp/TNEWS/logs',            # directory for storing logs
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model="marco_f1_score",
    )
    logging.info(model)

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train,         # training dataset
        eval_dataset=eval,          # evaluation dataset
        data_collator= data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train(checkpoint_dir)

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    logging.info(torch.cuda.is_available())

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--pretrian_model', default='hfl/chinese-roberta-wwm-ext-large')
    # args = parser.parse_args()
    # pretrian_model = args.pretrian_model
    fine_tune_stage_1()
    fine_tune_stage_2()


