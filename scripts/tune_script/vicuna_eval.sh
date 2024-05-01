#!/bin/bash

export PYTHONPATH=/home/cjz/SFTonGFM
export CUDA_VISIBLE_DEVICES=2

python /home/cjz/SFTonGFM/graphgpt/eval/run_vicuna_tit_gen.py \
    --model-name "/home/cjz/vicuna-7b-v1.5-16k" \
    --prompting_file "/home/cjz/SFTonGFM/reshape/test_items_tit_gen.json" \
    --train_file "/home/cjz/SFTonGFM/reshape/train_items_tit_gen.json" \
    --graph_data_path "/home/cjz/GraphGPT/graph_data/graph_data_all.pt" \
    --output_res_path "/home/cjz/SFTonGFM/output_eva_cora" \
    --start_id "0" \
    --end_id "1000" \
    --num_gpus "3" \
    --shot_num "20" \