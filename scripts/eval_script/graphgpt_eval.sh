# to fill in the following path to extract projector for the second tuning stage!
#!/bin/bash
output_model=/home/cjz/GraphGPT/checkpoints/few-shot-prompt
datapath=/home/cjz/GraphGPT/reshape/test_items.json
graph_data_path=/home/cjz/GraphGPT/graph_data/graph_data_all.pt
res_path=/home/cjz/GraphGPT/output_eva_cora
graph_tower_path=/home/cjz/GraphGPT/checkpoints/few-shot-prompt

start_id=0
end_id=209
num_gpus=3
gnn_type=gt
tuned_graph_prompt=/home/cjz/GraphGPT/checkpoints/few-shot-prompt
export PYTHONPATH="/home/cjz/GraphGPT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=2


# nohup python3.8 /home/cjz/GraphGPT/graphgpt/eval/run_graphgpt.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus} > /home/cjz/graphgpt_eval.out 2>&1 &
python3.8 /home/cjz/GraphGPT/graphgpt/eval/run_graphgpt.py --model-name ${output_model}  --prompting_file ${datapath} --graph_data_path ${graph_data_path} --output_res_path ${res_path} --start_id ${start_id} --end_id ${end_id} --num_gpus ${num_gpus}\
    --graph_tower_path ${graph_tower_path} --tuned_graph_tower True --gnn_type ${gnn_type} --tuned_graph_prompt ${tuned_graph_prompt}