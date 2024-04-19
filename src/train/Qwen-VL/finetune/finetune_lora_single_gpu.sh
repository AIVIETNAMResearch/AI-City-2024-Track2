#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`


MODEL="Qwen/Qwen-VL-Chat" #"Qwen/Qwen-VL-Chat"/"Qwen/Qwen-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.

# Specify path to your train and evaluate data
DATA="path_to_train_data"
EVAL_DATA="path_to_eval_data"

export CUDA_VISIBLE_DEVICES=0

python finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --eval_data_path $EVAL_DATA \
    --bf16 True \
    --fix_vit True \
    --output_dir "path_to_save_output" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 10 \
    --eval_steps 0.05 \
    --metric_for_best_model "eval_loss" \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --use_lora