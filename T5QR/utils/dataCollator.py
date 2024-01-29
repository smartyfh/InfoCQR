import os
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import shift_tokens_right
from typing import Dict, Iterable, List, Optional, Tuple


class T5DataCollator:
    def __init__(self, tokenizer, decoder_start_token_id, max_source_len, max_target_len):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_len
        self.max_target_length = max_target_len
        self.pad_token_id = tokenizer.pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, batch):
        contexts = [x['source'] for x in batch]
        responses = [x['target'] for x in batch]
        
        self.tokenizer.truncation_side = 'left'
        results = self.tokenizer(contexts, padding=True, max_length=self.max_source_length, truncation=True, return_tensors='pt')
        
        self.tokenizer.truncation_side = 'right'
        labels = self.tokenizer(text_target=responses, padding=True, max_length=self.max_target_length, truncation=True, return_tensors='pt')['input_ids']
        decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)
        # mask padded labels to not count the loss
        labels.masked_fill_(labels==self.pad_token_id, -100)
        
        results['decoder_input_ids'] = decoder_input_ids
        results['labels'] = labels
        return results
    
    