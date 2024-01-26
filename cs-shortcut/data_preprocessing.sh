#!/bin/bash

TASK=qrecc
DATA_PATH=/home/ye/CQR/datasets
OUTPUT_PATH=/home/ye/CQR/datasets/preprocessed

python data_preprocessing.py \
  --task ${TASK} \
  --data_path ${DATA_PATH} \
  --output_path ${OUTPUT_PATH} \
  --max_passage_length 384 \
  --pyserini_index_path ${DATA_PATH}/${TASK}/pyserini_index \
  --test_collection_path ${DATA_PATH}/${TASK}
  