#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH='/home/cjz/SFTonGFM'

graph_data_path=/home/cjz/SFTonGFM/data/graph_data_all.pt
task_embedding_path=/home/cjz/SFTonGFM/data/task_embedding_1.pt
dataset=products
ways=5
task_type=${dataset}_${ways}ways
eva_data_path=/home/cjz/SFTonGFM/data/${dataset}/test_${ways}ways.json
eva_log_path=/home/cjz/SFTonGFM/output_eva_old_${dataset}_case

arr=(50)

for i in "${arr[@]}"
do
    eva_path="/home/cjz/SFTonGFM/checkpoints/${dataset}-finetune-old/${ways}_way_${i}_shots"
    python /home/cjz/SFTonGFM/graphgpt/eval/run_SFT_task.py \
        --model-name "$eva_path" \
        --prompting_file ${eva_data_path} \
        --graph_data_path ${graph_data_path} \
        --output_res_path ${eva_log_path} \
        --graph_tower_path "$eva_path" \
        --start_id "0" \
        --end_id "1000" \
        --num_gpus "3" \
        --tuned_graph_tower "False" \
        --gnn_type "gt" \
        --use_rag "False" \
        --tuned_graph_prompt "$eva_path" \
        --combined_graph_prompt "True" \
        --task_type ${task_type} 
        # --task_related_prompt "False" \
        # --task_embedding_path ${task_embedding_path}
done