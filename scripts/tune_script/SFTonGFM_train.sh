#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH='/home/cjz/SFTonGFM'

model_path=/home/cjz/SFTonGFM/checkpoints/stage-2_task_prompt_complete
graph_data_path=/home/cjz/SFTonGFM/data/graph_data_all.pt
task_embedding_path=/home/cjz/SFTonGFM/data/task_embedding_1.pt
dataset=wikics
ways=5
task_type=${dataset}_${ways}ways
eva_data_path=/home/cjz/SFTonGFM/data/${dataset}/test_${ways}ways.json
eva_log_path=/home/cjz/SFTonGFM/output_eva_complete_${dataset}

arr=(50)

for i in "${arr[@]}"
do
    data_path="/home/cjz/SFTonGFM/data/${dataset}/train_${i}shots_${ways}ways.json"
    output_dir="/home/cjz/SFTonGFM/checkpoints/fine_tune-task-prompt/${dataset}_${ways}_ways_${i}_shots"

    python /home/cjz/SFTonGFM/graphgpt/train/train_mem.py \
        --model_name_or_path ${model_path} \
        --version "v1" \
        --data_path "$data_path" \
        --graph_content "./arxiv_ti_ab.json" \
        --graph_data_path ${graph_data_path} \
        --graph_tower "/home/cjz/GraphGPT/clip_gt_arxiv" \
        --pretrain_graph_mlp_adapter "${model_path}/graph_projector.bin" \
        --pretrain_graph_tower "${model_path}/tuned_graph_tower.bin" \
        --pretrain_graph_prompt "${model_path}/combined_graph_prompt.bin" \
        --tune_graph_mlp_adapter "False" \
        --tune_graph_tower "False" \
        --graph_select_layer "-2" \
        --use_graph_start_end "True" \
        --bf16 "True" \
        --output_dir "$output_dir" \
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
        --task_related_prompt "True" \
        --task_embedding_path ${task_embedding_path} \
        --task_type ${task_type} \
        --few_shot_adp "True"
        # > /home/cjz/SFTonGFM/tit_gen_20240423.out 2>&1 &
done