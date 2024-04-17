#!/bin/bash

export PYTHONPATH=/home/cjz/SFTonGFM
export CUDA_VISIBLE_DEVICES=1

python /home/cjz/SFTonGFM/graphgpt/eval/run_gpt_tit_gen.py \
    --model-name "/home/cjz/SFTonGFM/checkpoints/tit_gen-combined-prompt" \
    --prompting_file "/home/cjz/SFTonGFM/reshape/test_items_tit_gen.json" \
    --graph_data_path "/home/cjz/GraphGPT/graph_data/graph_data_all.pt" \
    --output_res_path "/home/cjz/SFTonGFM/output_eva_tit_gen" \
    --graph_tower_path "/home/cjz/SFTonGFM/checkpoints/tit_gen-combined-prompt" \
    --start_id "0" \
    --end_id "1000" \
    --num_gpus "3" \
    --tuned_graph_tower "False" \
    --gnn_type "gt" \
    --use_rag "False" \
    --tuned_graph_prompt "/home/cjz/SFTonGFM/checkpoints/tit_gen-combined-prompt" \
    --combined_graph_prompt "True"
