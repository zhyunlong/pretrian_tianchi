from dataset import Mydataset,data_collator, OCNLI_dataset
from transformers import BertForSequenceClassification
import torch
import logging
import json
import tqdm
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s' )
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

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
file_name = os.walk("exp/TNEWS/model/").__next__()[1][-1]
model_dict["TNEWS"] = "exp/TNEWS/model/%s"%file_name
file_name = os.walk("exp/OCNLI/model/").__next__()[1][-1]
model_dict["OCNLI"] = "exp/OCNLI/model/%s"%file_name
file_name = os.walk("exp/OCEMOTION/model/").__next__()[1][-1]
model_dict["OCEMOTION"] = "exp/OCEMOTION/model/%s"%file_name


for task_name in task_name_list:
    index = 0
    file = open("%s_predict.json"%task_name.lower(), "w+")
    if task_name == "OCNLI":
        test_dataset = OCNLI_dataset("/tcdata/ocnli_test_B.csv", task_label_dict["OCNLI"],is_test=True)
    else:
        test_dataset = Mydataset("/tcdata/%s_test_B.csv" % task_name.lower(), task_label_dict[task_name], is_test=True)
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, collate_fn=data_collator)
    print("logging model from %s"%model_dict[task_name])
    model = BertForSequenceClassification.from_pretrained(model_dict[task_name], num_labels=len(task_label_dict[task_name]))
    model.to(device)
    model.eval()
    inverse_label_dict = {v:k for k,v in task_label_dict[task_name].items()}
    for batch in dataloader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).logits
        predict_label = logits.argmax(-1)
        labels = batch["labels"].flatten().cpu().numpy()
        predict_label = predict_label.flatten().cpu().numpy()
        for i in range(labels.shape[0]):
            json.dump({"id":int(index), "label": inverse_label_dict[int(predict_label[i])]}, file)
            file.write("\n")
            index+=1
    print("task  %s complete " %task_name)
    file.close()

import zipfile
zip_file = zipfile.ZipFile('result.zip','a')
for task_name in task_name_list:
    zip_file.write("%s_predict.json"%task_name.lower(),compress_type=zipfile.ZIP_DEFLATED)
zip_file.close()

