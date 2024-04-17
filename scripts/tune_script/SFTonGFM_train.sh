#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH='/home/cjz/SFTonGFM'

nohup python /home/cjz/SFTonGFM/graphgpt/train/train_mem.py \
    --model_name_or_path "/home/cjz/SFTonGFM/checkpoints/fine_tune-combined-prompt-2" \
    --version "v1" \
    --data_path "/home/cjz/SFTonGFM/reshape/train_items_tit_gen_5shots.json" \
    --graph_content "./arxiv_ti_ab.json" \
    --graph_data_path "/home/cjz/GraphGPT/graph_data/graph_data_all.pt" \
    --graph_tower "/home/cjz/GraphGPT/clip_gt_arxiv" \
    --pretrain_graph_mlp_adapter "/home/cjz/SFTonGFM/checkpoints/stage_2-combined-prompt/graph_projector/checkpoint-320000.bin" \
    --pretrain_graph_tower "/home/cjz/SFTonGFM/checkpoints/fine_tune-combined-prompt-2/tuned_graph_tower/checkpoint-2100.bin" \
    --pretrain_graph_prompt "/home/cjz/SFTonGFM/checkpoints/fine_tune-combined-prompt-2/combined_graph_prompt/checkpoint-2100.bin" \
    --tune_graph_mlp_adapter "False" \
    --tune_graph_tower "False" \
    --graph_select_layer "-2" \
    --use_graph_start_end "True" \
    --bf16 "True" \
    --output_dir "/home/cjz/SFTonGFM/checkpoints/tit_gen-combined-prompt_5shots" \
    --num_train_epochs "2" \
    --per_device_train_batch_size "1" \
    --per_device_eval_batch_size "1" \
    --gradient_accumulation_steps "1" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "2700" \
    --save_total_limit "1" \
    --learning_rate "2e-5" \
    --weight_decay "0." \
    --warmup_ratio "0.03" \
    --lr_scheduler_type "cosine" \
    --logging_steps "1" \
    --tf32 "True" \
    --model_max_length "2048" \
    --gradient_checkpointing "True" \
    --dataloader_num_workers "4" \
    --lazy_preprocess "True" \
    --report_to "wandb" \
    --use_graph_prompt "True" \
    --combined_graph_prompt "True" > /home/cjz/SFTonGFM/tit_gen_5shots_20240416.out 2>&1 &
