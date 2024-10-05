#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH='/home/cjz/SFTonGFM'

nohup python /home/cjz/SFTonGFM/graphgpt/train/train_mem.py \
    --model_name_or_path "/home/cjz/SFTonGFM/checkpoints/stage-2_task_prompt(arxiv+pub_node)" \
    --version "v1" \
    --data_path "/home/cjz/SFTonGFM/reshape/pub_link_std.json" \
    --graph_content "./arxiv_ti_ab.json" \
    --graph_data_path "/home/cjz/GraphGPT/graph_data/graph_data_all.pt" \
    --graph_tower "/home/cjz/GraphGPT/clip_gt_arxiv" \
    --pretrain_graph_tower "/home/cjz/SFTonGFM/checkpoints/stage-2_task_prompt(arxiv+pub_node)/tuned_graph_tower.bin" \
    --pretrain_graph_mlp_adapter "/home/cjz/SFTonGFM/checkpoints/stage-2_task_prompt(arxiv+pub_node)/graph_projector.bin" \
    --pretrain_graph_prompt "/home/cjz/SFTonGFM/checkpoints/stage-2_task_prompt(arxiv+pub_node)/combined_graph_prompt.bin" \
    --tune_graph_mlp_adapter "True" \
    --tune_graph_tower "True" \
    --graph_select_layer "-2" \
    --use_graph_start_end "True" \
    --bf16 "True" \
    --output_dir "/home/cjz/SFTonGFM/checkpoints/stage-2_task_prompt(complete)" \
    --num_train_epochs "2" \
    --per_device_train_batch_size "1" \
    --per_device_eval_batch_size "1" \
    --gradient_accumulation_steps "1" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "20000" \
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
    --combined_graph_prompt "False" \
    --task_text_path "/home/cjz/SFTonGFM/reshape/task_text.json" \
    --task_type "pub_link" \
    --task_related_prompt "True" > /home/cjz/SFTonGFM/SFT_pretrain_stage_2.3.out 2>&1 &
