import torch
from transformers import *
import os

if(os.path.exists("tokenizer/")):
    tokenizer = BertTokenizer.from_pretrained("tokenizer/")
else:
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    tokenizer.save_pretrained("tokenizer/")

def pretrain_data_collator(features):
    batch = {}
    input_ids = [sample["input_seq"] for sample in features]
    out = tokenizer(input_ids,  add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
    batch["input_ids"] = out["input_ids"]
    batch["attention_mask"] = out["attention_mask"]
    batch["labels"] = out["input_ids"]
    return batch

class PretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.input_seq_all = []
        self.label_all = []
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            while (line != ""):
                split_line = line.split("\t")
                #assert len(split_line)== 3
                self.input_seq_all.append(split_line[1])
                self.label_all.append(split_line[2].strip())
                line = f.readline()

    def __getitem__(self, idx):
        sample = dict()
        sample["input_seq"] = self.input_seq_all[idx]
        return sample

    def __len__(self):
        assert len(self.label_all) == len(self.input_seq_all)
        return len(self.label_all)

