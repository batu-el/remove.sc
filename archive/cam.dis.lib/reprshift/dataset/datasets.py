
### Based on https://github.com/YyzHarry/SubpopBench ###

import os
import pandas as pd
import torch
from transformers import BertTokenizer

class SubpopDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, train_attr='yes', subsample_type=None, duplicates=None):
        df = pd.read_csv(metadata)
        df = df[df["split"] == (self.SPLITS[split])]
        self.idx = list(range(len(df)))
        self.x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = df["y"].tolist()
        self.a = df["a"].tolist() if train_attr == 'yes' else [0] * len(df["a"].tolist())
        self.transform_ = transform
        self._count_groups()

        if subsample_type is not None:
            self.subsample(subsample_type)
        if duplicates is not None:
            self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.num_attributes = len(set(self.a))
        self.num_labels = len(set(self.y))
        self.group_sizes = [0] * self.num_attributes * self.num_labels
        self.class_sizes = [0] * self.num_labels

        for i in self.idx:
            self.group_sizes[self.num_attributes * self.y[i] + self.a[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[self.num_attributes * self.y[i] + self.a[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.num_attributes * self.num_labels
        counts_y = [0] * self.num_labels
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.a[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.num_attributes * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.num_attributes * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = self.idx[index]
        x = self.transform(self.x[i])
        y = torch.tensor(self.y[i], dtype=torch.long)
        a = torch.tensor(self.a[i], dtype=torch.long)
        return i, x, y, a

    def __len__(self):
        return len(self.idx)
    
class MultiNLI(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000
    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        root = os.path.join(data_path, "multinli", "glue_data", "MNLI")
        metadata = os.path.join(data_path, "multinli", "metadata_multinli.csv")
        
        self.features_array = []
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features = torch.load(os.path.join(root, feature_file))
            self.features_array += features

        self.all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor( [f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)
        self.x_array = torch.stack((self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2)
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, i):
        return self.x_array[int(i)]

class CivilComments(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 granularity="coarse"):
        text = pd.read_csv(os.path.join(data_path, "civilcomments/civilcomments_{}.csv".format(granularity)))
        metadata = os.path.join(data_path, "civilcomments", "metadata_civilcomments_{}.csv".format(granularity))

        self.text_array = list(text["comment_text"])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        self.data_type = "text"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"]
                ), dim=2
                ), dim=0
            )
        else:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"]
                ), dim=2
                ), dim=0
            )