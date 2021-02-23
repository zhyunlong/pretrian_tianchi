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
    train = Mydataset("data_split/OCEMOTION_train.csv", label_dict)
    eval = Mydataset("data_split/OCEMOTION_dev.csv", label_dict)
    model = BertClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=len(label_dict))
    training_args = TrainingArguments(
        output_dir='exp/OCEMOTION_stage1/model',          # output directory
        num_train_epochs=5,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=2,
        eval_steps=1000,
        learning_rate=1e-5,
        logging_dir='exp/OCEMOTION_stages1/logs',            # directory for storing logs
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
    return trainer.model

def fine_tune_stage_2(model):
    label_dict = get_OCEMOTION_label_dict()
    train = Mydataset("data_split/OCEMOTION_dev.csv", label_dict)
    eval = Mydataset("data_split/OCEMOTION_train_small.csv", label_dict)
    training_args = TrainingArguments(
        output_dir='exp/OCEMOTION/model',          # output directory
        num_train_epochs=5,              # total # of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        save_total_limit=2,
        eval_steps=200,
        learning_rate=1e-5,
        logging_dir='exp/OCEMOTION/logs',            # directory for storing logs
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
    trainer.train()
    return trainer.model


def inference(model):
    import json
    index = 0
    task_name = "OCEMOTION"
    task_label_dict = get_OCEMOTION_label_dict()
    file = open("%s_predict.json" % task_name.lower(), "w+")
    test_dataset = Mydataset("/tcdata/%s_test_B.csv" % task_name.lower(), task_label_dict, is_test=True)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    inverse_label_dict = {v: k for k, v in task_label_dict.items()}
    for batch in dataloader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))[0]
        predict_label = logits.argmax(-1)
        labels = batch["labels"].flatten().cpu().numpy()
        predict_label = predict_label.flatten().cpu().numpy()
        for i in range(labels.shape[0]):
            json.dump({"id": int(index), "label": inverse_label_dict[int(predict_label[i])]}, file)
            file.write("\n")
            index += 1
    print("task  %s complete " % task_name)
    file.close()


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
    model = fine_tune_stage_1()
    model = fine_tune_stage_2(model)
    inference(model)



