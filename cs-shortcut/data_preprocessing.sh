#!/bin/bash

TASK=qrecc
DATA_PATH=/home/user/InfoCQR/datasets
OUTPUT_PATH=/home/user/InfoCQR/datasets/preprocessed

python data_preprocessing.py \
  --task ${TASK} \
  --data_path ${DATA_PATH} \
  --output_path ${OUTPUT_PATH} \
  --max_passage_length 384 \
  --test_collection_path ${DATA_PATH}/${TASK}
  
