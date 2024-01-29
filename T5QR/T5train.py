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
import wandb

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
    
    bsz = training_args.per_device_train_batch_size * torch.cuda.device_count() * training_args.gradient_accumulation_steps
    lr = training_args.learning_rate
    ep = training_args.num_train_epochs
    acc = training_args.gradient_accumulation_steps
    rw = data_args.data_dir.split("/")[-1]
    gl = data_args.max_resp_length
    training_args.output_dir = Path(training_args.output_dir).joinpath(f'_lr{lr}_bs{bsz}_ep{ep}_acc{acc}_gl{gl}_{rw}')
    training_args.logging_dir = training_args.output_dir
    training_args.run_name = training_args.run_name + f'_lr{lr}_bs{bsz}_ep{ep}_acc{acc}_gl{gl}_{rw}'
     
    # Get datasets
    train_dataset = None
    eval_dataset = None
    if model_args.mode == 'train':
#         print("Load training data from ", Path(data_args.data_dir).joinpath("train.csv"))
#         train_dataset = load_dataset('csv', data_files={"train": str(Path(data_args.data_dir).joinpath("train.csv"))}, split="train", keep_default_na=False)
        train_dataset = load_from_disk(Path(data_args.data_dir).joinpath("train"))
    if model_args.mode == 'train' and training_args.evaluation_strategy != EvaluationStrategy.NO:
#         eval_dataset = load_dataset('csv', data_files={"dev": str(Path(data_args.data_dir).joinpath("dev.csv"))}, split="dev", keep_default_na=False)
        eval_dataset = load_from_disk(Path(data_args.data_dir).joinpath("dev"))
    
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded dev_dataset length is: {len(eval_dataset)}")

    training_args.remove_unused_columns = False
    wandb_dir = Path('/home/ye/CQR/wandb/').joinpath(training_args.run_name)
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(project="T5QR_S",
               name=training_args.run_name,
               dir=wandb_dir
              )
    # Initialize the Trainer
    training_args.do_predict = False
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=T5DataCollator(tokenizer, config.decoder_start_token_id, data_args.max_ctx_length, data_args.max_resp_length),
        tokenizer=tokenizer,
        compute_metrics= Metrics(tokenizer).compute_metrics,
    )

    if model_args.mode == 'train':
        print('enter training')
        check_output_dir(training_args) # check if output_dir exists
        print('start training')

        train_result = trainer.train()
        trainer.save_model(Path(training_args.output_dir).joinpath("best-checkpoint")) # save best model checkpoint
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        
    else:
        raise RuntimeError("Does not support the specified mode.")

if __name__ == "__main__":
    main()
    