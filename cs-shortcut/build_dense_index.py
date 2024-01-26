import torch
import argparse
from tqdm import tqdm, trange
import numpy as np
import random
import json
import time
import os
import shutil
import sys
import logging
from utils import set_seed, get_logger
from utils.indexing_utils import DenseIndexer, DocumentCollection
from sentence_transformers import SentenceTransformer


logger = get_logger(__name__)


def main(args):
    docs_path = os.path.join(args.data_path, args.task, "test_collections")
    docs_collection = DocumentCollection(f"{docs_path}/data.h5")
    num_docs = docs_collection.length
    print(f"Total number of passages: {num_docs}")
    
    model = SentenceTransformer('sentence-transformers/gtr-t5-base')
    model.max_seq_length = 384
    
    docs_per_split = num_docs // args.num_splits
    print(f"Number of passages per split: {docs_per_split}")
    
    for i in range(5, args.num_splits):
        st = i * docs_per_split
        ed = (i+1) * docs_per_split
        if i == args.num_splits - 1:
            ed = num_docs # the last split can have more docs
        
        # get the docs and their ids of each split
        docs_i_text = []
        docs_i_ids = []
        for j in trange(st, ed):
            docs_i_text.append(docs_collection.get_text(j))
            docs_i_ids.append(docs_collection.get_pid(j))
            
        indexer = DenseIndexer(batch_size=args.index_batch_size, 
                               max_buffer_size=args.max_buffer_size, 
                               logger=logger)
        indexer.passage_inference(model,
                                  docs_i_text,
                                  docs_i_ids,
                                  i,
                                  os.path.join(args.output_path, f"index_test_{i}.faiss"))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default="/home/ye/CQR/datasets/preprocessed")
    parser.add_argument('--output_path', type=str, default="/home/ye/CQR/datasets/preprocessed/qrecc/dense_index")
    parser.add_argument('--task', type=str, default='qrecc')
    parser.add_argument('--model_name_or_path', type=str, default='sentence-transformers/gtr-t5-base')
    
    parser.add_argument('--index_batch_size', type=int, default=256)
    parser.add_argument('--max_buffer_size', type=int, default=592000)
    parser.add_argument('--num_splits', type=int, default=8)
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    fileHandler = logging.FileHandler(f"{args.output_path}/log.out", "a")
    formatter = logging.Formatter('%(asctime)s > %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.info(args.output_path)
    logger.info("logging start!")
    main(args)
