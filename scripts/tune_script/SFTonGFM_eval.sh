#!/bin/bash

export PYTHONPATH=/home/cjz/SFTonGFM
export CUDA_VISIBLE_DEVICES=2

python /home/cjz/SFTonGFM/graphgpt/eval/run_graphgpt.py \
    --model-name "/home/cjz/SFTonGFM/checkpoints/wikics-finetune-old/5_way_50_shots" \
    --prompting_file "/home/cjz/OneForAll/data/single_graph/wikics/test_item_5ways.json" \
    --graph_data_path "/home/cjz/OneForAll/data/single_graph/wikics/word2vec_graph_all.pt" \
    --output_res_path "/home/cjz/SFTonGFM/output_eva_wikics" \
    --graph_tower_path "/home/cjz/SFTonGFM/checkpoints/wikics-finetune-old/5_way_50_shots" \
    --start_id "0" \
    --end_id "1000" \
    --num_gpus "3" \
    --tuned_graph_tower "False" \
    --gnn_type "gt" \
    --use_rag "False" \
    --tuned_graph_prompt "/home/cjz/SFTonGFM/checkpoints/wikics-finetune-old/5_way_50_shots" \
    --combined_graph_prompt "True"
