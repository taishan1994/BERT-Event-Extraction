import os
import json
import torch
import numpy as np

from config import ReConfig
from model import BertRe
from data_loader import ReDataset, ReCollate

from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer


class Trainer:
    def __init__(self,
                 output_dir=None,
                 model=None,
                 train_loader=None,
                 dev_loader=None,
                 test_loader=None,
                 optimizer=None,
                 schedule=None,
                 epochs=1,
                 device="cpu",
                 save_step=None, ):
        self.output_dir = output_dir
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.schedule = schedule
        self.total_step = len(self.train_loader) * self.epochs
        self.save_step = save_step

    def train(self):
        global_step = 1
        for epoch in range(1, self.epochs + 1):
            for step, batch_data in enumerate(self.train_loader):
                self.model.train()
                for key, value in batch_data.items():
                    batch_data[key] = value.to(self.device)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                token_type_ids = batch_data["token_type_ids"]
                start_labels = batch_data["start_labels"]
                end_labels = batch_data["end_labels"]
                output = self.model(input_ids, attention_mask, token_type_ids, start_labels, end_labels)
                loss = output.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.schedule.step()
                print(f"【train】{epoch}/{self.epochs} {global_step}/{self.total_step} loss:{loss.item()}")
                if global_step % self.save_step == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_re.bin"))
                global_step += 1

        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model_re.bin"))

    def decode(self, start, end, start_inds, end_inds):
        res = []
        for st, en, start_ind, end_ind in zip(start, end, start_inds, end_inds):
            start = start_ind.index(1)
            end = end_ind.index(0) if 0 in end_ind else len(end_ind)
            flag = False
            st = st[start:end]
            en = en[start:end]
            i = 0
            j = 0
            tmp = []  # 可能存在多个答案
            while i < len(st) and j < len(en):
                while i < len(st) and st[i] != 1:
                    i += 1
                if i == len(st) or j == len(en):
                    break
                if j == 0:
                    j = i
                for k in range(j, len(en)):
                    if en[k] == 1:
                        if i <= k:
                            tmp.append((i, k))
                        j = k + 1
                        i += k + 1
                        flag = True
                        break
                i += 1
            if len(tmp) != 0:
                res.append(tmp)
            if not flag:
                res.append(["未识别"])
        return res

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "pytorch_model_re.bin")))
        self.model.eval()
        total_num = 0
        correct_num = 0
        for step, batch_data in enumerate(tqdm(self.test_loader)):
            for key, value in batch_data.items():
                batch_data[key] = value.to(self.device)
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            token_type_ids = batch_data["token_type_ids"]
            start_labels = batch_data["start_labels"]
            end_labels = batch_data["end_labels"]

            output = self.model(input_ids,
                                attention_mask,
                                token_type_ids,
                                start_labels,
                                end_labels)
            start_logits = output.start_logits
            start_logits = start_logits.detach().cpu().numpy()
            start_logits = np.argmax(start_logits, -1).tolist()
            end_logits = output.end_logits
            end_logits = end_logits.detach().cpu().numpy()
            end_logits = np.argmax(end_logits, -1).tolist()

            start_labels = output.start_labels
            start_labels = start_labels.detach().cpu().numpy()

            end_labels = output.end_labels
            end_labels = end_labels.detach().cpu().numpy()

            start_inds = np.where(start_labels != -100, 1, 0).tolist()
            end_inds = attention_mask.detach().cpu().numpy().tolist()

            start_labels = start_labels.tolist()
            end_labels = end_labels.tolist()

            pred = self.decode(start_logits, end_logits, start_inds, end_inds)
            true = self.decode(start_labels, end_labels, start_inds, end_inds)
            print(pred)
            print(true)
            print("=" * 100)
            assert len(pred) == len(true)

            correct_num += sum([1 if pred[i] == true[i] else 0 for i in range(len(true))])
            total_num += len(true)

        rersult = {
            'total': total_num,
            "correct": correct_num,
            "accuracy": correct_num / total_num
        }

        return rersult


def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.learning_rate},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler


def main(data_name):
    args = ReConfig(data_name)

    with open(os.path.join(args.output_dir, "re_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.data_path, "train.txt"), "r") as fp:
        train_data = fp.read().split("\n")
    train_data = [json.loads(d) for d in train_data]

    with open(os.path.join(args.data_path, "dev.txt"), "r") as fp:
        dev_data = fp.read().split("\n")
    dev_data = [json.loads(d) for d in dev_data]

    re_collate = ReCollate(args, tokenizer)
    train_dataset = ReDataset(train_data)
    dev_dataset = ReDataset(dev_data)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=2,
                              collate_fn=re_collate.collate)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.dev_batch_size, num_workers=2,
                            collate_fn=re_collate.collate)

    model = BertRe(args)

    # for name,_ in model.named_parameters():
    #   print(name)

    model.to(device)
    t_toal = len(train_loader) * args.epochs
    optimizer, schedule = build_optimizer_and_scheduler(args, model, t_toal)

    train = Trainer(
        output_dir=args.output_dir,
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=dev_loader,
        optimizer=optimizer,
        schedule=schedule,
        epochs=args.epochs,
        device=device,
        save_step=args.save_step,
    )

    train.train()

    result = train.test()
    print("总共：{}，正确：{}，准确率：{}".format(result["total"], result["correct"], result["accuracy"]))


if __name__ == "__main__":
    data_name = "duee"
    main(data_name)
