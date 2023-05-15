import os
import re
import json
import codecs
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class ProcessDueeData:
    def __init__(self):
        self.data_path = "./data/duee/"
        self.train_file = self.data_path + "ori_data/duee_train.json"
        self.dev_file = self.data_path + "ori_data/duee_dev.json"
        self.test_file = self.data_path + "ori_data/duee_test2.json"
        self.schema_file = self.data_path + "ori_data/duee_event_schema.json"

    def get_ner_data(self, in_file, out_file, mode=""):
        with open(in_file, "r") as fp:
            data = fp.read().strip().split("\n")
        res = []
        labels = set()
        for d in data:
            d = eval(d)
            text = d["text"]
            event_list = d["event_list"]
            tmp = {}
            tmp["id"] = d["id"]
            tmp["text"] = [i for i in text]
            tmp["labels"] = ["O"] * len(text)
            for event in event_list:
                trigger = event["trigger"]
                event_type = event["event_type"]
                event_type = event_type.replace("-", "_")
                labels.add(event_type)
                trigger_start = event["trigger_start_index"]
                trigger_end = trigger_start + len(trigger)
                tmp["labels"][trigger_start] = "B-" + event_type
                for i in range(trigger_start + 1, trigger_end):
                    tmp["labels"][i] = "I-" + event_type
            res.append(tmp)

        with open(out_file, "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in res]))

        if mode == "train":
            with open(self.data_path + "ner_data/labels.txt", "w") as fp:
                fp.write("\n".join(list(labels)))

    def get_re_data(self, in_file, out_file, mode=""):
        with open(in_file, "r") as fp:
            data = fp.read().strip().split("\n")
        res = []
        labels = set()
        for d in data:
            d = eval(d)
            text = d["text"]
            event_list = d["event_list"]
            for event in event_list:
                trigger = event["trigger"]
                event_type = event["event_type"]
                event_type = event_type.replace("-", "_")
                arguments = event["arguments"]
                event_tmp = {}
                for argument in arguments:
                    role = argument["role"]
                    prompt = event_type + "的" + role
                    tmp = {}
                    tmp["id"] = d["id"]
                    tmp["text"] = [i for i in text]
                    argument_text = argument["argument"]
                    argument_start = argument["argument_start_index"]
                    argument_end = argument_start + len(argument_text) - 1
                    if prompt not in event_tmp:
                        event_tmp[prompt] = {}
                        tmp["start"] = [0] * len(text)
                        tmp["end"] = [0] * len(text)
                        tmp["start"][argument_start] = 1
                        tmp["end"][argument_end] = 1
                        tmp["prompt"] = prompt
                        event_tmp[prompt] = tmp
                    else:
                        event_tmp[prompt]["start"][argument_start] = 1
                        event_tmp[prompt]["end"][argument_end] = 1
                for k, v in event_tmp.items():
                    res.append(v)

        with open(out_file, "w") as fp:
            fp.write("\n".join([json.dumps(d, ensure_ascii=False) for d in res]))

        if mode == "train":
            with open(self.schema_file, "r") as fp:
                schema = fp.read().strip().split("\n")
            rels = {}
            for s in schema:
                s = eval(s)
                event_type = s["event_type"].replace("-", "_")
                rels[event_type] = []
                role_list = s["role_list"]
                for role in role_list:
                    rels[event_type].append(role["role"])

            with open(os.path.join(self.data_path, "re_data/rels.txt"), "w") as fp:
                json.dump(rels, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    processGdcqData = ProcessDueeData()
    processGdcqData.get_ner_data(processGdcqData.train_file, "./data/duee/ner_data/train.txt", "train")
    processGdcqData.get_ner_data(processGdcqData.dev_file, "./data/duee/ner_data/dev.txt", "dev")
    processGdcqData.get_re_data(processGdcqData.train_file, "./data/duee/re_data/train.txt", "train")
    processGdcqData.get_re_data(processGdcqData.dev_file, "./data/duee/re_data/dev.txt", "dev")
