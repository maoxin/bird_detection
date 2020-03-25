from pathlib import Path
import json

from PIL import Image
import torch
from torch.utils.data import Dataset

import torch
import numpy as np
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

RootDir = Path("/media/data1/mx_dataset")

class BirdDatasetCls(Dataset):
    def __init__(self, name, transforms=None, train=True, small_set=False):
        self.name = name
        self.train = train
        self.small_set = small_set
        self.transforms = transforms

        if self.name == 'real':
            self.dataset_path = RootDir/"bird_dataset/bird_dataset_real_cam20_Jun-Aug"
        elif self.name == 'synthesized':
            self.dataset_path = RootDir/"bird_dataset/bird_dataset_synthesized"
        else:
            raise Exception("name should be either 'real' or 'synthesized")

        self.imgs_path = self.dataset_path/"images"
        if self.train:
            self.record_path = self.dataset_path/"train_records_cls.json"
        else:
            self.record_path = self.dataset_path/"test_records_cls.json"

        with open(self.record_path) as f:
            self.records = [json.loads(l.strip().replace("\'", "\"")) for l in f.readlines()]

        self.records = [r for r in self.records if r['label_index'] <= 3]


        if self.name == 'synthesized' and self.small_set and self.train:
            self.records = self.records[:1000]
        
    def __getitem__(self, idx):
        record = self.records[idx]

        img_path = str(self.imgs_path/record['img_name'])
        img = Image.open(img_path).convert("RGB")
        bbox = record['bbox']
        img = img.crop(bbox)

        label = record['label_index'] - 1

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.records)

class BirdDataset(Dataset):
    def __init__(self, name, transforms, train=True, small_set=False, only_instance=False):
        self.name = name
        self.train = train
        self.small_set = small_set
        self.transforms = transforms
        self.only_instance = only_instance

        if self.name == 'real':
            # self.dataset_path = RootDir/"bird_dataset/bird_dataset_real_cam20_Jun-Aug"
            self.dataset_path = RootDir/"bird_dataset/bird_dataset_real"
        elif self.name == 'synthesized':
            self.dataset_path = RootDir/"bird_dataset/bird_dataset_synthesized"
        else:
            raise Exception("name should be either 'real' or 'synthesized")

        self.imgs_path = self.dataset_path/"images"
        if self.train:
            self.record_path = self.dataset_path/"train_records.json"
        else:
            self.record_path = self.dataset_path/"test_records.json"

        with open(self.record_path) as f:
            self.records = [json.loads(l.strip().replace("\'", "\"")) for l in f.readlines()]

        if self.train and self.name != "synthesized":
            self.records = self.records[:60]
        if self.name == 'synthesized' and self.small_set and self.train:
            # self.records = self.records[:1000]
            self.records = self.records[:120]

        records = []
        for r in self.records:
            labels = torch.as_tensor(r['list_label_index'], dtype=torch.int64)
            if len(labels[labels <= 3]) > 0:
                records.append(r)
        self.records = records
        
    def __getitem__(self, idx):
        record = self.records[idx]

        img_path = str(self.imgs_path/record['img_name'])
        img = Image.open(img_path).convert("RGB")
        
        boxes = torch.as_tensor(record['list_bbox'], dtype=torch.float32)
        labels = torch.as_tensor(record['list_label_index'], dtype=torch.int64)
        boxes = boxes[labels <= 3]
        labels = labels[labels <= 3]
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((labels.size(0),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels_real"] = labels
        if self.only_instance:
            target["labels"] = torch.ones_like(target["labels_real"], dtype=target["labels_real"].dtype)
            target["labels"][:] = 1
        else:
            target["labels"] = target["labels_real"]
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.records)