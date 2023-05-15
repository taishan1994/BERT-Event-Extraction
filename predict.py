import os
import json
import torch
import numpy as np

from collections import namedtuple
from model import BertNer, BertRe
from seqeval.metrics.sequence_labeling import get_entities
from transformers import BertTokenizer


def get_args(args_path, args_name=None):
    with open(args_path, "r") as fp:
        args_dict = json.load(fp)
    # 注意args不可被修改了
    args = namedtuple(args_name, args_dict.keys())(*args_dict.values())
    return args


class DueePredictor:
    def __init__(self, data_name):
        self.data_name = data_name
        self.ner_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "ner_args.json"), "ner_args")
        self.re_args = get_args(os.path.join("./checkpoint/{}/".format(data_name), "re_args.json"), "re_args")
        self.ner_id2label = {int(k): v for k, v in self.ner_args.id2label.items()}
        self.tokenizer = BertTokenizer.from_pretrained(self.ner_args.bert_dir)
        self.max_seq_len = self.ner_args.max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ner_model = BertNer(self.ner_args)
        self.ner_model.load_state_dict(torch.load(os.path.join(self.ner_args.output_dir, "pytorch_model_ner.bin")))
        self.ner_model.to(self.device)
        self.re_model = BertRe(self.re_args)
        self.re_model.load_state_dict(torch.load(os.path.join(self.re_args.output_dir, "pytorch_model_re.bin")))
        self.re_model.to(self.device)
        with open(os.path.join(self.re_args.data_path, "rels.txt"), "r") as fp:
            self.rels = json.load(fp)
            print(self.rels)

    def ner_tokenizer(self, text):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        text = text[:self.max_seq_len - 2]
        text = ["[CLS]"] + [i for i in text] + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(text)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        input_ids = torch.tensor(np.array([input_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask

    def re_tokenizer(self, text, prompt):
        # print("文本长度需要小于：{}".format(self.max_seq_len))
        pre_length = 3 + len(prompt)
        text = text[:self.max_seq_len - pre_length]
        text = list(text)
        prompt = list(prompt)
        tmp_input_ids = ["[CLS]"] + prompt + ["[SEP]"] + text + ["[SEP]"]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = [1] * len(tmp_input_ids) + [0] * (self.max_seq_len - len(tmp_input_ids))
        token_type_ids = [0] * self.max_seq_len
        input_ids = torch.tensor(np.array([input_ids]))
        token_type_ids = torch.tensor(np.array([token_type_ids]))
        attention_mask = torch.tensor(np.array([attention_mask]))
        return input_ids, attention_mask, token_type_ids

    def re_predict(self, text, ner_result):
        res = {}
        for event_type, v in ner_result.items():
            event_rols = self.rels[event_type]
            v = list(set([i[0] for i in v]))
            res[event_type] = []
            for role in event_rols:
                prompt = event_type + "的" + role
                # print(prompt)
                input_ids, attention_mask, token_type_ids = self.re_tokenizer(text, prompt)
                input_ids = input_ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                output = self.re_model(input_ids, token_type_ids, attention_mask)
                start_logits = output.start_logits
                end_logits = output.end_logits
                start_logits = start_logits.detach().cpu().numpy()
                end_logits = end_logits.detach().cpu().numpy()
                start_logits = np.argmax(start_logits, -1)
                end_logits = np.argmax(end_logits, -1)
                ind = 2 + len(prompt)
                attention_mask = attention_mask.detach().cpu().numpy().tolist()[0]
                end_ind = attention_mask.index(0) if 0 in attention_mask else len(attention_mask)
                start_logits = start_logits[0]
                end_logits = end_logits[0]
                start_logits = start_logits[ind:end_ind]
                end_logits = end_logits[ind:end_ind]
                # for i,s in enumerate(start_logits):
                #   for j,e in enumerate(end_logits):
                #     if s == e and s == 1:
                #       t = text[i:j+1]
                #       res.append((role, text[i:j+1], i, j))
                i = 0
                j = 0
                tmp = []  # 可能存在多个答案
                while i < len(start_logits) and j < len(end_logits):
                    while i < len(start_logits) and start_logits[i] != 1:
                        i += 1
                    if i == len(start_logits) or j == len(end_logits):
                        break
                    if j == 0:
                        j = i
                    for k in range(j, len(end_logits)):
                        if end_logits[k] == 1:
                            if i <= k:
                                res[event_type].append((role, text[i:k + 1], i, k))
                            j = k + 1
                            i += k + 1
                            flag = True
                            break
                    i += 1
        return res

    def ner_predict(self, text):
        input_ids, attention_mask = self.ner_tokenizer(text)

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.ner_model(input_ids, attention_mask)
        attention_mask = attention_mask.detach().cpu().numpy()
        length = sum(attention_mask[0])
        logits = output.logits
        logits = logits[0][1:length - 1]
        logits = [self.ner_id2label[i] for i in logits]
        entities = get_entities(logits)
        result = {}
        for ent in entities:
            ent_name = ent[0]
            ent_start = ent[1]
            ent_end = ent[2]
            if ent_name not in result:
                result[ent_name] = [("".join(text[ent_start:ent_end + 1]), ent_start, ent_end)]
            else:
                result[ent_name].append(("".join(text[ent_start:ent_end + 1]), ent_start, ent_end))
        return result


if __name__ == "__main__":
    data_name = "duee"
    dueePredictor = DueePredictor(data_name)

    with open("./data/duee/ori_data/duee_test2.json", "r") as fp:
        data = fp.read().strip().split("\n")
    for i, d in enumerate(data):
        d = eval(d)
        text = d["text"]
        ner_result = dueePredictor.ner_predict(text)
        re_result = dueePredictor.re_predict(text, ner_result)
        print("文本>>>>>", text)
        print("实体>>>>>", ner_result)
        print("关系>>>>>", re_result)
        print("=" * 100)
        if i > 10:
            break
