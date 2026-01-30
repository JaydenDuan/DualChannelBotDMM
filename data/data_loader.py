import torch
import torch.nn.functional as F
import json
import os
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


def preprocess_tensor(tensor, replace_nan=True, replace_inf=True, normalize=True, clamp=True):
    if replace_nan and torch.isnan(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0)
    if replace_inf and torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, posinf=5.0, neginf=-5.0)
    if normalize and tensor.dim() > 1:
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True) + 1e-8
        tensor = (tensor - mean) / std
    if clamp:
        tensor = torch.clamp(tensor, -5.0, 5.0)
    if torch.isnan(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0)
    return tensor


class Twibot20Dataset(Dataset):
    def __init__(self, data_dir, split='train', num_steps=5):
        self.data_dir = data_dir
        self.split = split
        self.num_steps = num_steps
        self._load_features()

    def _load_features(self):
        prefix = self.split
        map_loc = torch.device('cpu')

        des_path = os.path.join(self.data_dir, 'descriptions', f'{prefix}_des_tensor.pt')
        self.des_tensor = torch.load(des_path, map_location=map_loc, weights_only=True)
        self.des_tensor = preprocess_tensor(self.des_tensor)

        num_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_num_properties_tensor.pt')
        self.num_prop = torch.load(num_path, map_location=map_loc, weights_only=True)
        self.num_prop = preprocess_tensor(self.num_prop)

        llm_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_llm_features.pt')
        self.llm_features = torch.load(llm_path, map_location=map_loc, weights_only=True)
        self.llm_features = preprocess_tensor(self.llm_features)

        label_path = os.path.join(self.data_dir, f'{prefix}_labels.pt')
        self.labels = torch.load(label_path, map_location=map_loc, weights_only=True)
        self.labels[self.labels == 3] = 2

        self.tweets_list = []
        self.amrs_list = []
        self.edge_indices = []

        for step in range(1, self.num_steps + 1):
            text_path = os.path.join(self.data_dir, f'{prefix}_text_tensor{step}.pt')
            if os.path.exists(text_path):
                text_tensor = torch.load(text_path, map_location=map_loc, weights_only=True)
            else:
                text_tensor = torch.zeros((len(self.labels), 768))
            self.tweets_list.append(preprocess_tensor(text_tensor))

            amr_path = os.path.join(self.data_dir, f'{prefix}_amr_tensor{step}.pt')
            if os.path.exists(amr_path):
                amr_tensor = torch.load(amr_path, map_location=map_loc, weights_only=True)
            else:
                amr_tensor = torch.zeros((len(self.labels), 768))
            self.amrs_list.append(preprocess_tensor(amr_tensor))

            edge_path = os.path.join(self.data_dir, f'{prefix}_edge_index_part{step}.pt')
            if os.path.exists(edge_path):
                edge_index = torch.load(edge_path, map_location=map_loc, weights_only=True)
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


class BotSimDataset(Dataset):
    def __init__(self, data_dir, split='train', num_steps=5):
        self.data_dir = data_dir
        self.split = split
        self.num_steps = num_steps
        self._load_features()

    def _load_features(self):
        prefix = self.split
        map_loc = torch.device('cpu')

        des_path = os.path.join(self.data_dir, 'descriptions', f'{prefix}_des_tensor.pt')
        self.des_tensor = torch.load(des_path, map_location=map_loc, weights_only=True)
        self.des_tensor = preprocess_tensor(self.des_tensor)

        num_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_num_properties_tensor.pt')
        self.num_prop = torch.load(num_path, map_location=map_loc, weights_only=True)
        self.num_prop = preprocess_tensor(self.num_prop)

        llm_path = os.path.join(self.data_dir, 'numerical', f'{prefix}_llm_features.pt')
        self.llm_features = torch.load(llm_path, map_location=map_loc, weights_only=True)
        self.llm_features = preprocess_tensor(self.llm_features)

        label_file = os.path.join(self.data_dir, 'labels', f'{prefix}.json')
        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            self.labels = torch.tensor([d.get('label', 0) for d in data.values()], dtype=torch.long)
        else:
            self.labels = torch.tensor([d.get('label', 0) for d in data], dtype=torch.long)

        min_samples = min(self.des_tensor.shape[0], self.num_prop.shape[0],
                          self.llm_features.shape[0], self.labels.shape[0])

        self.des_tensor = self.des_tensor[:min_samples]
        self.num_prop = self.num_prop[:min_samples]
        self.llm_features = self.llm_features[:min_samples]
        self.labels = self.labels[:min_samples]

        self.tweets_list = []
        self.amrs_list = []
        self.edge_indices = []

        for step in range(1, self.num_steps + 1):
            text_path = os.path.join(self.data_dir, f'{prefix}_text_tensor{step}.pt')
            if os.path.exists(text_path):
                text_tensor = torch.load(text_path, map_location=map_loc, weights_only=True)[:min_samples]
            else:
                text_tensor = torch.zeros((min_samples, 768))
            self.tweets_list.append(preprocess_tensor(text_tensor))

            amr_path = os.path.join(self.data_dir, f'{prefix}_amr_tensor{step}.pt')
            if os.path.exists(amr_path):
                amr_tensor = torch.load(amr_path, map_location=map_loc, weights_only=True)[:min_samples]
            else:
                amr_tensor = torch.zeros((min_samples, 768))
            self.amrs_list.append(preprocess_tensor(amr_tensor))

            edge_path = os.path.join(self.data_dir, f'{prefix}_edge_index_part{step}.pt')
            if os.path.exists(edge_path):
                edge_index = torch.load(edge_path, map_location=map_loc, weights_only=True)
                if edge_index.size(1) > 0:
                    valid_edges = (edge_index[0] < min_samples) & (edge_index[1] < min_samples)
                    edge_index = edge_index[:, valid_edges]
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_indices.append(edge_index)

        self.num_samples = min_samples
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
