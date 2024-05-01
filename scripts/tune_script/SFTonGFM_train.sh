#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH='/home/cjz/SFTonGFM'

python /home/cjz/SFTonGFM/graphgpt/train/train_mem.py \
    --model_name_or_path "/home/cjz/SFTonGFM/checkpoints/fine_tune-combined-prompt-50shots(sota)" \
    --version "v1" \
    --data_path "/home/cjz/OneForAll/data/single_graph/wikics/train_50shots_5ways.json" \
    --graph_content "./arxiv_ti_ab.json" \
    --graph_data_path "/home/cjz/OneForAll/data/single_graph/wikics/word2vec_graph_all.pt" \
    --graph_tower "/home/cjz/GraphGPT/clip_gt_arxiv" \
    --pretrain_graph_mlp_adapter "/home/cjz/SFTonGFM/checkpoints/stage_2-combined-prompt/graph_projector/checkpoint-320000.bin" \
    --pretrain_graph_tower "/home/cjz/SFTonGFM/checkpoints/fine_tune-combined-prompt-50shots(sota)/tuned_graph_tower/checkpoint-2100.bin" \
    --pretrain_graph_prompt "/home/cjz/SFTonGFM/checkpoints/fine_tune-combined-prompt-50shots(sota)/combined_graph_prompt/checkpoint-2100.bin" \
    --tune_graph_mlp_adapter "False" \
    --tune_graph_tower "False" \
    --graph_select_layer "-2" \
    --use_graph_start_end "True" \
    --bf16 "True" \
    --output_dir "/home/cjz/SFTonGFM/checkpoints/wikics-finetune-old/5_way_50_shots" \
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
    --combined_graph_prompt "True" \
    --task_related_prompt "False" \
    --task_text_path "/home/cjz/SFTonGFM/reshape/task_text.json" \
    --task_type "arxiv_clas" 
    # > /home/cjz/SFTonGFM/tit_gen_20240423.out 2>&1 &
