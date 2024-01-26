#!/bin/bash


if [ -n "$DATA_PATH" ]; then
  :
else
  export DATA_PATH=dataset
fi

if [[ -d "$DATA_PATH" ]]; then
  :
else
  mkdir $DATA_PATH
  echo "make base data directory $DATA_PATH"
fi
 
if [ -n "$TASK" ]; then
  echo $TASK
  
  if [[ -d "$DATA_PATH/$TASK" ]]; then
    echo "set data directory $DATA_PATH/$TASK"
  else
    mkdir $DATA_PATH/$TASK
    echo "make data directory $DATA_PATH/$TASK"
  fi
  
else
  echo "Please set environment variable! TASK=[orconvqa|qrecc]"
  exit 125
fi


if [[ -f "$DATA_PATH/$TASK/train.json" ]]; then
    echo "file already exists!"
    exit 125
fi

if [ $TASK == 'orconvqa' ]; then
  # donwload orconvqa
  wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/train.txt -O $DATA_PATH/$TASK/train.json
  wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/dev.txt -O $DATA_PATH/$TASK/dev.json
  wget https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt -O $DATA_PATH/$TASK/test.json
  wget https://ciir.cs.umass.edu/downloads/ORConvQA/dev_blocks.txt.gz -O $DATA_PATH/$TASK/dev_blocks.txt.gz
  wget https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz -O $DATA_PATH/$TASK/all_blocks.txt.gz
  wget https://ciir.cs.umass.edu/downloads/ORConvQA/qrels.txt.gz -O $DATA_PATH/$TASK/qrels.txt.gz

  # unzip collections
  echo "Unzipping..."
  gzip -d $DATA_PATH/$TASK/qrels.txt.gz
  gzip -d $DATA_PATH/$TASK/dev_blocks.txt.gz
  gzip -d $DATA_PATH/$TASK/all_blocks.txt.gz
  
elif [ $TASK == 'qrecc' ]; then  
  cp assets/$TASK/train.json $DATA_PATH/$TASK/train.json
  cp assets/$TASK/dev.json $DATA_PATH/$TASK/dev.json
  cp assets/$TASK/test.json $DATA_PATH/$TASK/test.json
  cp assets/$TASK/qrels.txt $DATA_PATH/$TASK/qrels.txt
  cp assets/$TASK/qrels_dev.txt $DATA_PATH/$TASK/qrels_dev.txt
  cp assets/$TASK/test_question_types.json $DATA_PATH/$TASK/test_question_types.json
  cp assets/$TASK/dev_blocks.jsonl.tar.gz $DATA_PATH/$TASK/dev_blocks.jsonl.tar.gz
  
  wget https://zenodo.org/record/5115890/files/passages.zip?download=1 -O $DATA_PATH/$TASK/passages.zip
   # unzip collections
  echo "Unzipping..."
  unzip $DATA_PATH/$TASK/passages.zip
  tar -zxvf $DATA_PATH/$TASK/dev_blocks.jsonl.tar.gz
  
else
  :
fi