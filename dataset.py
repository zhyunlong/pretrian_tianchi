import torch
from transformers import *
import os

if(os.path.exists("tokenizer/")):
    tokenizer = BertTokenizer.from_pretrained("tokenizer/")
else:
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    tokenizer.save_pretrained("tokenizer/")

def data_collator(features):
    batch = {}
    input_ids = [sample["input_seq"] for sample in features]
    labels = [sample["labels"] for sample in features]
    out = tokenizer(input_ids,  add_special_tokens=True, padding=True, truncation=True, return_tensors="pt")
    batch["input_ids"] = out["input_ids"]
    batch["attention_mask"] = out["attention_mask"]
    batch["labels"] = torch.tensor(labels)
    return batch

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, path, label_dict, is_test=False):
        self.input_seq_all = []
        self.label_all = []
        self.label_dict = label_dict
        self.is_test = is_test
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            while (line != ""):
                split_line = line.split("\t")
                #assert len(split_line)== 3
                self.input_seq_all.append(split_line[1])
                if is_test==False:
                    self.label_all.append(split_line[2].strip())
                else:
                    self.label_all.append(-1)
                line = f.readline()

    def __getitem__(self, idx):
        sample = dict()
        sample["input_seq"] = self.input_seq_all[idx]
        if self.is_test:
            sample["labels"] = self.label_all[idx]
        else:
            sample["labels"] = self.label_dict[self.label_all[idx]]
        return sample

    def __len__(self):
        assert len(self.label_all) == len(self.input_seq_all)
        return len(self.label_all)



class OCNLI_dataset(torch.utils.data.Dataset):
    def __init__(self, path, label_dict, is_test=False):
        self.first_input_seq_all = []
        self.second_input_seq_all = []
        self.label_all = []
        self.label_dict = label_dict
        self.is_test = is_test
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            while (line != ""):
                split_line = line.split("\t")
                #assert len(split_line)== 4
                self.first_input_seq_all.append(split_line[1])
                self.second_input_seq_all.append(split_line[2])
                if is_test==False:
                    self.label_all.append(split_line[3].strip())
                else:
                    self.label_all.append(-1)
                line = f.readline()

    def __getitem__(self, idx):
        sample = dict()
        sample["input_seq"] = [self.first_input_seq_all[idx],self.second_input_seq_all[idx]]
        if self.is_test:
            sample["labels"] = self.label_all[idx]
        else:
            sample["labels"] = self.label_dict[self.label_all[idx]]
        return sample

    def __len__(self):
        assert len(self.label_all) == len(self.first_input_seq_all)
        return len(self.label_all)
