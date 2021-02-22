import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel

class BertClassification(nn.Module):
    def __init__(self, bert_pretrained, num_labels, freeze_bert=False):
        super(BertClassification, self).__init__()
        self.hidden_size = 1024
        self.lstm_hidden_size = 256
        self.hidden_dropout_prob = 0.1
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(bert_pretrained)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.bilstm = nn.LSTM(self.hidden_size, self.lstm_hidden_size, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(self.lstm_hidden_size*2, self.num_labels)
        #self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        #pooled_output = outputs[1]
        hidden_states = outputs[0]
        #pooled_output = hidden_states.mean(-2)
        #pooled_output = self.dropout(pooled_output)
        hidden_states = self.dropout(hidden_states)
        lstm_hidden_states, _ = self.bilstm(hidden_states)
        #lstm_hidden_states = self.dropout(lstm_hidden_states)
        pooled_output = lstm_hidden_states.mean(-2)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
