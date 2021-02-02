from dataset import Mydataset,data_collator, OCNLI_dataset
from transformers import BertForSequenceClassification
import torch
import logging
import json
import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


task_name_list = ["OCEMOTION", "OCNLI", "TNEWS"]
task_label_dict = {}
label_dict = {}
for i in range(17):
    if i<10:
        label_dict["10%d"%i] = i
    else:
        label_dict["1%d"%i] = i
task_label_dict["TNEWS"] = label_dict
task_label_dict["OCNLI"] = {"0":0, "1":1, "2":2}
task_label_dict["OCEMOTION"] = {'sadness': 0, 'happiness': 1, 'disgust': 2, 'anger': 3, 'like': 4, 'surprise': 5, 'fear': 6}
model_dict = {}
model_dict["TNEWS"] = "bert-base-chinese"
model_dict["OCNLI"] = "bert-base-chinese"
model_dict["OCEMOTION"] = "bert-base-chinese"

file = open("result.json", "w+")

for task_name in task_name_list:
    index = 0
    if task_name == "OCNLI":
        test_dataset = OCNLI_dataset("data/OCNLI_train1128.csv", task_label_dict["OCNLI"],is_test=True)
    else:
        test_dataset = Mydataset("data/%s_train1128.csv" % task_name, task_label_dict[task_name], is_test=True)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, collate_fn=data_collator)
    model = BertForSequenceClassification.from_pretrained(model_dict[task_name], num_labels=len(task_label_dict[task_name]))
    model.eval()
    inverse_label_dict = {v:k for k,v in task_label_dict[task_name].items()}
    for batch in dataloader:
        logits = model(batch["input_ids"], batch["attention_mask"]).logits
        predict_label = logits.argmax(-1)
        labels = batch["labels"].flatten().numpy()
        predict_label = predict_label.flatten().numpy()
        for i in range(labels.shape[0]):
            json.dump({"id":int(index), "predict": inverse_label_dict[int(predict_label[i])]}, file)
            file.write("\n")
            index+=1
        if(index>64):
            break
    print("task  %s complete " %task_name)

