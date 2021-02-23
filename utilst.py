from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix,classification_report
import logging
import os
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

def get_TNEWS_label_dict():
    label_dict = {}
    for i in range(17):
        if i < 10:
            label_dict["10%d" % i] = i
        else:
            label_dict["1%d" % i] = i
    return label_dict

def get_OCNLI_label_dict():
    return {"0": 0, "1": 1, "2": 2}

def get_OCEMOTION_label_dict():
    return {'sadness': 0, 'happiness': 1, 'disgust': 2, 'anger': 3, 'like': 4, 'surprise': 5, 'fear': 6}

def get_newest_checkpoit(task_name, exp_dir="exp"):
    model_dir = os.path.join(exp_dir, task_name, "model")
    checkpoint_dir = os.walk(model_dir).__next__()[1][-1]
    return checkpoint_dir

