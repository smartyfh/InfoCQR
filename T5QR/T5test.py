from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    AutoModelForSeq2SeqLM,
)
from dataclasses import dataclass, field
from datasets import load_dataset, load_from_disk
from transformers.trainer_utils import EvaluationStrategy
from transformers.utils import logging
from typing import Optional, List
from pathlib import Path
import sys
import os
import torch
import math
from utils.util import check_output_dir, Metrics
from utils.dataCollator import T5DataCollator
from tqdm import tqdm, trange
import json

logger = logging.get_logger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to use.
    """

    model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models, e.g. t5-base"}
    )
    mode: str = field(
        metadata={"help": "mode, can be either train, eval or pred"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .csv files (or other data files) for the task."}
    )
    max_ctx_length: Optional[int] = field(
        default=384,
        metadata={
            "help": "The maximum number of dialogue context after tokenization."
        },
    )
    max_resp_length: Optional[int] = field(
        default=64,
        metadata={
            "help": "The maximum length for system responses after tokenization."
        },
    )
    source_column: Optional[str] = field(
        default='source',
        metadata={
            "help": "The column name of the dialog context."},
    )
    target_column: Optional[str] = field(
        default='target',
        metadata={
            "help": "The column name of the system response."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    set_seed(training_args.seed) # set training seed
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer.add_tokens(["<Que>", "<Ans>"])
    
    if model_args.model_name in ['t5-base', 't5-small', 't5-large']:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name,
            cache_dir=model_args.cache_dir,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        print("Starting from a finetuned model!")
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            cache_dir=model_args.cache_dir,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    
    config = model.config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
     
    # Get datasets
    eval_dataset = None
    if model_args.mode == 'dev':
        eval_dataset = load_from_disk(Path(data_args.data_dir).joinpath("dev"))
    elif model_args.mode == 'test' or model_args.mode == 'predict':
        eval_dataset = load_from_disk(Path(data_args.data_dir).joinpath("test"))
    
    print(f"loaded eval_dataset length is: {len(eval_dataset)}")
    
    data_collator=T5DataCollator(tokenizer, config.decoder_start_token_id, data_args.max_ctx_length, data_args.max_resp_length)
    model_cfg = model_args.model_name.split("/")[-2]
    gl = int(model_cfg[model_cfg.find("gl"):].split("_")[0][2:])
    training_args.generation_max_length = gl
    print(f'Max generation length: {gl}')
    
    results = {}
    num_steps = math.ceil(len(eval_dataset) / training_args.per_device_eval_batch_size)
    for i in trange(num_steps):
        st = i*training_args.per_device_eval_batch_size
        ed = min(st+training_args.per_device_eval_batch_size, len(eval_dataset))
        part_i = []
        for j in range(st, ed):
            part_i.append(eval_dataset[j])
        inputs = data_collator(part_i)
#         if i == 0:
#             print(part_i[:3])
        del inputs['decoder_input_ids']
        del inputs['labels']
        inputs = inputs.to(device)
        outputs = model.generate(**inputs, max_length=training_args.generation_max_length)
        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if i == 0:
            print(pred[:5])
        for j in range(st, ed):
            cid = eval_dataset[j]['conv_id']
            results[cid] = {}
            results[cid]['label'] = eval_dataset[j]['target']
            results[cid]['pred'] = pred[j-st]
            if i == 0 and j < 5:
                print(results[cid])
                
    os.makedirs('predictions', exist_ok=True)
    with open(os.path.join('predictions', model_cfg+"-pred.json"), 'w') as out:
        json.dump(results, out, indent=2)
        
        
if __name__ == "__main__":
    main()
    