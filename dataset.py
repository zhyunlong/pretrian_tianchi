import torch
from transformers import *


tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

def data_collator(features):
    batch = {}
    input_ids = [sample["input_ids"] for sample in features]
    labels = [sample["labels"] for sample in features]
    batch["input_ids"] = tokenizer(input_ids,  add_special_tokens=True, return_tensors="pt")["input_ids"]
    batch["labels"] = torch.tensor(labels)
    return batch

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, path, label_dict):
        self.input_seq_all = []
        self.label_all = []
        self.label_dict = label_dict
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            while (line != ""):
                split_line = line.split("\t")
                assert len(split_line)== 3
                self.input_seq_all.append(split_line[1])
                self.label_all.append(split_line[2].strip())
                line = f.readline()

    def __getitem__(self, idx):
        sample = dict()
        sample["input_ids"] = self.input_seq_all[idx]
        sample["labels"] = self.label_dict[self.label_all[idx]]
        return sample

    def __len__(self):
        assert len(self.label_all) == len(self.input_seq_all)
        return len(self.label_all)
