# data/data_loader.py
import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class BotDataset(Dataset):
    def __init__(self, data_dir, split='train', num_steps=5):
        self.data_dir = data_dir
        self.split = split
        self.num_steps = num_steps
        self._load_features()

    def _load_features(self):
        prefix = self.split if self.split != 'test' else 'test'

        des_path = os.path.join(self.data_dir, 'descriptions', f'{prefix}_des_tensor.pt')
        self.des_tensor = torch.load(des_path)

        num_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_num_properties_tensor.pt')
        self.num_prop = torch.load(num_path)

        llm_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_llm_features.pt')
        self.llm_features = torch.load(llm_path)

        label_path = os.path.join(self.data_dir, f'{prefix}_labels.pt')
        self.labels = torch.load(label_path)

        self.tweets_list = []
        self.amrs_list = []
        self.edge_indices = []

        for step in range(1, self.num_steps + 1):
            text_path = os.path.join(self.data_dir, f'{prefix}_text_tensor{step}.pt')
            if os.path.exists(text_path):
                text_tensor = torch.load(text_path)
            else:
                text_tensor = torch.zeros((len(self.labels), 768))
            self.tweets_list.append(text_tensor)
            amr_path = os.path.join(self.data_dir, f'{prefix}_amr_tensor{step}.pt')
            if os.path.exists(amr_path):
                amr_tensor = torch.load(amr_path)
            else:
                amr_tensor = torch.zeros((len(self.labels), 768))
            self.amrs_list.append(amr_tensor)
            edge_path = os.path.join(self.data_dir, f'{prefix}_edge_index_part{step}.pt')
            if os.path.exists(edge_path):
                edge_index = torch.load(edge_path)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_indices.append(edge_index)

        self.num_samples = len(self.labels)
        logger.info(f"Loaded {self.num_samples} samples for {self.split} split")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'des': self.des_tensor[idx],
            'tweets': [tweet[idx] for tweet in self.tweets_list],
            'amrs': [amr[idx] for amr in self.amrs_list],
            'num_prop': self.num_prop[idx],
            'llm_features': self.llm_features[idx],
            'edge_indices': self.edge_indices,
            'label': self.labels[idx]
        }


class FoldDataLoader:
    def __init__(self, cv_dir, fold_id, device='cuda', num_steps=5):
        self.cv_dir = cv_dir
        self.fold_id = fold_id
        self.device = device
        self.num_steps = num_steps

        self.train_dir = os.path.join(cv_dir, f'fold_{fold_id}', 'encoded', 'train', 'temporal')
        self.val_dir = os.path.join(cv_dir, f'fold_{fold_id}', 'encoded', 'val', 'temporal')

    def get_train_loader(self, batch_size=32):
        dataset = BotDataset(self.train_dir, 'train', self.num_steps)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def get_val_loader(self, batch_size=32):
        dataset = BotDataset(self.val_dir, 'val', self.num_steps)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def collate_fn(batch):
    des = torch.stack([item['des'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    num_prop = torch.stack([item['num_prop'] for item in batch])
    llm_features = torch.stack([item['llm_features'] for item in batch])
    tweets = []
    amrs = []
    for step in range(len(batch[0]['tweets'])):
        tweets.append(torch.stack([item['tweets'][step] for item in batch]))
        amrs.append(torch.stack([item['amrs'][step] for item in batch]))
    edge_indices = batch[0]['edge_indices']

    return des, tweets, amrs, num_prop, llm_features, edge_indices, labels