import torch
import numpy as np

from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = self.data[item]["text"]
        labels = self.data[item]["labels"]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data


class ReDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class ReCollate:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_len = args.max_seq_len

    def collate(self, batch_data):
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_start_labels = []
        batch_end_labels = []
        for d in batch_data:
            text = d["text"]
            start_labels = d["start"]
            end_labels = d["end"]
            prompt = d["prompt"]
            prompt = [i for i in prompt]
            pre_length = 3 + len(prompt)
            if len(text) > self.max_seq_len - pre_length:
                text = text[:self.max_seq_len - pre_length]
                start_labels = start_labels[:self.max_seq_len - pre_length]
                end_labels = end_labels[:self.max_seq_len - pre_length]

            tmp_input_ids = ["[CLS]"] + prompt + ["[SEP]"] + text + ["[SEP]"]
            tmp_input_ids = self.tokenizer.convert_tokens_to_ids(tmp_input_ids)
            attention_mask = [1] * len(tmp_input_ids)
            input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
            attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
            token_type_ids = [0] * self.max_seq_len
            start_labels = [-100] * (pre_length - 1) + start_labels + [-100] + [-100] * (
                        self.max_seq_len - len(tmp_input_ids))
            end_labels = [-100] * (pre_length - 1) + end_labels + [-100] + [-100] * (
                        self.max_seq_len - len(tmp_input_ids))

            # print(len(input_ids) , len(attention_mask) , len(token_type_ids) , len(start_labels) , len(end_labels))

            # assert len(input_ids) == len(attention_mask) == len(token_type_ids) == len(start_labels) == len(end_labels)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_start_labels.append(start_labels)
            batch_end_labels.append(end_labels)

        input_ids = torch.tensor(batch_input_ids)
        attention_mask = torch.tensor(batch_attention_mask)
        token_type_ids = torch.tensor(batch_token_type_ids)
        start_labels = torch.tensor(batch_start_labels)
        end_labels = torch.tensor(batch_end_labels)

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "start_labels": start_labels,
            "end_labels": end_labels
        }
        return data
