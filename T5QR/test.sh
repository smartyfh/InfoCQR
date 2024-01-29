#!/usr/bin/env bash
#train the model with multiple GPUs
gpus=0

DATE_WITH_TIME=$(date "+%Y%m%d-%H%M%S")

tokenizer_name=t5-base
# finetune from scracth, set model name to long t5 global base
# finetune from a checkpoint, set model name to the folder containing the checkpoint
model_name=outputs/_lr1e-05_bs32_ep10.0_acc2_gl64_Truth_rewrite/best-checkpoint
mode=test

max_ctx_length=384
max_resp_length=64

data_dir=datasets/qrecc/Truth_rewrite
output_dir=outputs
logging_dir=outputs/logs

overwrite_output_dir=true
do_train=true
  
evaluation_strategy=steps
save_strategy=steps
logging_strategy=steps
log_level=info
eval_steps=400
save_steps=400
logging_steps=400
warmup_steps=1600

per_device_train_batch_size=16
per_device_eval_batch_size=64
num_train_epochs=10
learning_rate=1e-5
gradient_accumulation_steps=2
weight_decay=0.01
fp16=false

save_total_limit=1
seed=42
load_best_model_at_end=true
metric_for_best_model=bleu
predict_with_generate=true
greater_is_better=true
generation_max_length=64
generation_num_beams=1
report_to=wandb
run_name=T5QR

CUDA_VISIBLE_DEVICES=1 python T5test.py \
  --tokenizer_name ${tokenizer_name} \
  --model_name ${model_name} \
  --mode ${mode} \
  --max_ctx_length ${max_ctx_length} \
  --max_resp_length ${max_resp_length} \
  --data_dir ${data_dir} \
  --output_dir ${output_dir} \
  --logging_dir ${logging_dir} \
  --overwrite_output_dir ${overwrite_output_dir} \
  --do_train ${do_train} \
  --evaluation_strategy ${evaluation_strategy} \
  --save_strategy ${save_strategy} \
  --logging_strategy ${logging_strategy} \
  --log_level ${log_level} \
  --eval_steps ${eval_steps} \
  --save_steps ${save_steps} \
  --logging_steps ${logging_steps} \
  --warmup_steps ${warmup_steps} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_eval_batch_size} \
  --num_train_epochs ${num_train_epochs} \
  --learning_rate ${learning_rate} \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --weight_decay ${weight_decay} \
  --fp16 ${fp16} \
  --save_total_limit ${save_total_limit} \
  --seed ${seed} \
  --load_best_model_at_end ${load_best_model_at_end} \
  --metric_for_best_model ${metric_for_best_model} \
  --predict_with_generate ${predict_with_generate} \
  --greater_is_better ${greater_is_better} \
  --generation_max_length ${generation_max_length} \
  --generation_num_beams ${generation_num_beams} \
  --report_to ${report_to} \
  --run_name ${run_name} 