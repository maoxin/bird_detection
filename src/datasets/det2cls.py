import numpy as np
from pathlib import Path
import json

RootDir = Path("/media/data1/mx_dataset")

def det2cls(name='real'):
    if name == 'real':
        dataset_path = RootDir/"bird_dataset/bird_dataset_real_cam20_Jun-Aug"
    elif name == 'synthesized':
        dataset_path = RootDir/"bird_dataset/bird_dataset_synthesized"
    else:
        raise Exception("name should be either 'real' or 'synthesized")

    train_record_path = dataset_path/"train_records.json"
    test_record_path = dataset_path/"test_records.json"

    with open(train_record_path) as f:
        train_records = [json.loads(l.strip().replace("\'", "\"")) for l in f.readlines()]
        train_records = [r for r in train_records if (np.array(r['list_label_index']) <= 3).sum() > 0]
    with open(test_record_path)as f:
        test_records = [json.loads(l.strip().replace("\'", "\"")) for l in f.readlines()]
        test_records = [r for r in test_records if (np.array(r['list_label_index']) <= 3).sum() > 0]

    train_records_cls = []
    for r in train_records:
        img_name = r['img_name']
        img_size = r['img_size']

        for bbox, label_index, label in zip(r['list_bbox'], r['list_label_index'], r['list_label']):
            train_records_cls.append({
                'img_name': img_name,
                'bbox': bbox,
                'label_index': label_index,
                'label': label
            })

    test_records_cls = []
    for r in test_records:
        img_name = r['img_name']
        img_size = r['img_size']

        for bbox, label_index, label in zip(r['list_bbox'], r['list_label_index'], r['list_label']):
            test_records_cls.append({
                'img_name': img_name,
                'bbox': bbox,
                'label_index': label_index,
                'label': label
            })

    print(f"train cls set size: {len(train_records_cls)}")
    print(f"test cls set size: {len(test_records_cls)}")

    with open(dataset_path/"train_records_cls.json", 'w') as f:
        for r in train_records_cls:
            f.write(f"{r}\n")
    with open(dataset_path/"test_records_cls.json", 'w') as f:
        for r in test_records_cls:
            f.write(f"{r}\n")

if __name__ == "__main__":
    print("real")
    det2cls('real')

    print('synthesized')
    det2cls("synthesized")

    
