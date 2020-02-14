from pathlib import Path
import json

from PIL import Image
import torch
from torch.utils.data import Dataset

RootDir = Path("/media/data1/mx_dataset")

class BirdDataset(Dataset):
    def __init__(self, name, transforms, train=True, small_set=False):
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
            self.record_path = self.dataset_path/"train_records.json"
        else:
            self.record_path = self.dataset_path/"test_records.json"

        with open(self.record_path) as f:
            self.records = [json.loads(l.strip().replace("\'", "\"")) for l in f.readlines()]

        if self.name == 'synthesized' and self.small_set and self.train:
            self.records = self.records[:1000]

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
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.records)