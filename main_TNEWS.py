from dataset import Mydataset,data_collator
from dataset_for_pretrain import PretrainDataset, pretrain_data_collator
from transformers import *
import os
import torch
import logging
from model import BertClassification

def pretrain_stage():
    pretrained_model = BertLMHeadModel.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    train = PretrainDataset("data_split/TNEWS_train.csv")

    training_args = TrainingArguments(
        output_dir='exp/TNEWS_pretrain/model',  # output directory
        num_train_epochs=2,  # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        save_total_limit=1,
        save_steps=1000,
        #eval_steps=1000,
        logging_dir='exp/TNEWS_pretrain/logs',  # directory for storing logs
        #evaluation_strategy='steps',
        # load_best_model_at_end=True,
        # metric_for_best_model="marco_f1_score",
        prediction_loss_only=True,
        #do_eval=False,
    )

    trainer = Trainer(
        model=pretrained_model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train,  # training dataset
        data_collator=pretrain_data_collator,
    )
    trainer.train()

def fine_tune_stage():
    label_dict = {}
    for i in range(17):
        if i < 10:
            label_dict["10%d" % i] = i
        else:
            label_dict["1%d" % i] = i
    train = Mydataset("data_split/TNEWS_train.csv", label_dict)
    eval = Mydataset("data_split/TNEWS_dev.csv", label_dict)
    model = BertClassification("hfl/chinese-roberta-wwm-ext-large", num_labels=len(label_dict))
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(name)
    #model = BertForSequenceClassification.from_pretrained("exp/TNEWS_pretrain/model/checkpoint-6000", num_labels=len(label_dict))
    #model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=len(label_dict))
    training_args = TrainingArguments(
        output_dir='exp/TNEWS/model',          # output directory
        num_train_epochs=1,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=1,
        save_steps=1000,
        eval_steps=1000,
        learning_rate=1e-5,
        logging_dir='exp/TNEWS/logs',            # directory for storing logs
        evaluation_strategy='steps',
        #load_best_model_at_end=True,
        #metric_for_best_model="marco_f1_score",
        #prediction_loss_only=True,
        do_eval=False,
    )
    #logging.info(model)

    from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix,classification_report


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        labels = labels.flatten()
        preds = preds.flatten()
        marco_f1_score = f1_score(labels, preds, average='macro')
        logging.info(marco_f1_score)
        #logging.info(f"{'confusion_matrix':*^80}")
        #logging.info(confusion_matrix(labels, preds, ))
        #logging.info(f"{'classification_report':*^80}")
        #logging.info(classification_report(labels, preds, ))
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


def fine_tune_stage_fix():
    label_dict = {}
    for i in range(17):
        if i < 10:
            label_dict["10%d" % i] = i
        else:
            label_dict["1%d" % i] = i
    train = Mydataset("data_split/TNEWS_train.csv", label_dict)
    eval = Mydataset("data_split/TNEWS_dev.csv", label_dict)
    model = BertClassification("exp/TNEWS/model/checkpoint-3000", num_labels=len(label_dict), freeze_bert=True)
    #model = BertClassification("hfl/chinese-roberta-wwm-ext-large", num_labels=len(label_dict), freeze_bert=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(name)
    #model = BertForSequenceClassification.from_pretrained("exp/TNEWS_pretrain/model/checkpoint-6000", num_labels=len(label_dict))
    training_args = TrainingArguments(
        output_dir='exp/TNEWS_fix/model',          # output directory
        num_train_epochs=4,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=1,
        save_steps=1000,
        eval_steps=1000,
        learning_rate=1e-2,
        logging_dir='exp/TNEWS_fix/logs',            # directory for storing logs
        evaluation_strategy='steps',
        #load_best_model_at_end=True,
        #metric_for_best_model="marco_f1_score",
        #prediction_loss_only=True,
        do_eval=False,
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
        #logging.info(f"{'confusion_matrix':*^80}")
        #logging.info(confusion_matrix(labels, preds, ))
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

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    logging.info(torch.cuda.is_available())
    #pretrain_stage()
    #fine_tune_stage()
    fine_tune_stage_fix()


