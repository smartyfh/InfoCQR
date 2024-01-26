from tqdm import tqdm

import random
from copy import deepcopy
from itertools import chain
from collections import Counter, defaultdict

from torch.utils.data import Dataset
import torch
import h5py
import json


def batch_to_device(batch, device):
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue

        batch[k] = v.to(device)
    return batch


class RetrievalDataset(Dataset):
    def __init__(self, dataset,
                 sampler=None,
                 n_negative=0,
                 pad_token_id=0,
                 tokenizer=None,
                 rng=None):
        self.dataset = dataset
        self.length = len(dataset)
        self.sampler = sampler
        self.n_negative = n_negative
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.rng = rng
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item
    
    def __len__(self):
        return self.length
    
    def pad_cands(self, arrays, padding):
        max_length = max(list(map(len, arrays)))
        max_length_per_cand = len(arrays[0][0])
        padding_array = [padding] * max_length_per_cand
        arrays = [
            array + [padding_array] * (max_length - len(array))
            for array in arrays
        ]
        return arrays

    def collate_fn(self, batch):
        ids = [b.guid for b in batch]

        input_ids = torch.LongTensor([b.input_id for b in batch])
        input_masks = torch.LongTensor([b.input_mask for b in batch])

        if self.n_negative and self.sampler is not None and hasattr(batch[0], 'hard_negative_ids'):
            cand_ids = []
            for b in batch:
                pos = b.cand_ids[b.label]
                negative_candidates = b.hard_negative_ids
                negs = self.sampler.negative_sampling(negative_candidates, self.n_negative)
                cands = [pos] + negs
                cand_ids.append(cands)
            cand_ids = torch.tensor(self.pad_cands(cand_ids, self.pad_token_id))
            cand_masks = cand_ids.ne(self.pad_token_id)
            labels = torch.LongTensor([0 for b in batch]) # first
            
            hard_array = []
            for i in range(labels.size(0)):
                hard_array.append(range(1, self.n_negative + 1))
            hards = torch.tensor(hard_array)
        else:
            cand_ids = torch.LongTensor(self.pad_cands([b.cand_ids for b in batch], self.pad_token_id))
            cand_masks = torch.LongTensor(self.pad_cands([b.cand_mask for b in batch], self.pad_token_id))
            labels = torch.LongTensor([b.label for b in batch])
            hards = torch.tensor([b.hard_negative_idx for b in batch]).unsqueeze(-1)

        if self.n_negative:
            labels = torch.cat([labels.unsqueeze(-1), hards], -1)
        
        result = {
            'ids': ids,
            'input_ids': input_ids,
            'input_masks': input_masks,
            'cand_ids': cand_ids,
            'cand_masks': cand_masks,
            'labels': labels
        }
        return result
