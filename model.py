import torch
import torch.nn as nn

from torchcrf import CRF
from transformers import BertModel, BertConfig

class ModelOutput:
  def __init__(self, 
         logits=None, 
         labels=None, 
         loss=None, 
         start_logits=None,
         end_logits=None,
         start_labels=None,
         end_labels=None):
    self.logits = logits
    self.labels = labels
    self.loss = loss
    self.start_logits = start_logits
    self.end_logits = end_logits
    self.start_labels = start_labels
    self.end_labels = end_labels


class BertNer(nn.Module):
  def __init__(self, args):
    super(BertNer, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    hidden_size = self.bert_config.hidden_size
    self.lstm_hiden = 128
    self.max_seq_len = args.max_seq_len
    self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
               dropout=0.1)
    self.linear = nn.Linear(self.lstm_hiden * 2, args.num_labels)
    self.crf = CRF(args.num_labels, batch_first=True)

  def forward(self, input_ids, attention_mask, labels=None):
    bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    seq_out = bert_output[0]  # [batchsize, max_seq_Len, 768]
    batch_size = seq_out.size(0)
    seq_out, _ = self.bilstm(seq_out)
    seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
    seq_out = seq_out.contiguous().view(batch_size, self.max_seq_len, -1)
    seq_out = self.linear(seq_out)
    logits = self.crf.decode(seq_out, mask=attention_mask.bool())
    loss = None
    if labels is not None:
      loss = -self.crf(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
    model_output = ModelOutput(logits=logits, labels=labels, loss=loss)
    return model_output

class BertRe(nn.Module):
  def __init__(self, args):
    super(BertRe, self).__init__()
    self.bert = BertModel.from_pretrained(args.bert_dir)
    self.bert_config = BertConfig.from_pretrained(args.bert_dir)
    self.hidden_size = self.bert_config.hidden_size
    self.start_linear = nn.Linear(self.hidden_size, 2)
    self.end_linear = nn.Linear(self.hidden_size, 2)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, 
         input_ids,
         attention_mask,
         token_type_ids,
         start_labels=None,
         end_labels=None):
    bert_output = self.bert(input_ids=input_ids)
    seq_out = bert_output[0]  # [batchsize, max_seq_Len, 768]
    start_logits = self.start_linear(seq_out)
    end_logits = self.end_linear(seq_out)
    loss = None


    if start_labels is not None:
      start_ = start_logits.view(-1, 2)
      start_labels_ = start_labels.view(-1)
      end_ = end_logits.view(-1, 2)
      end_labels_ = end_labels.view(-1)

      start_loss = self.criterion(start_, start_labels_)
      end_loss = self.criterion(end_, end_labels_)
      loss = start_loss + end_loss
    model_output = ModelOutput(start_logits=start_logits, 
                   end_logits=end_logits,
                   start_labels=start_labels,
                   end_labels=end_labels,
                   loss=loss)
    return model_output
