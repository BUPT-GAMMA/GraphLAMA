# to fill in the following path to run the second stage of our GraphGPT!
model_path=/home/cjz/vicuna-7b-v1.5-16k
instruct_ds=/home/cjz/GraphGPT/stage_2_instruct/arxiv_pub_node_st_cot_link_mix.json
graph_data_path=/home/cjz/GraphGPT/graph_data/graph_data_all.pt
pretra_gnn=/home/cjz/GraphGPT/clip_gt_arxiv
tuned_proj=/home/cjz/GraphGPT/checkpoints/stage_2/graph_projector/checkpoint-150000.bin
output_model=/home/cjz/SFTonGFM/checkpoints/stage_2-prompt
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH='/home/cjz/SFTonGFM'

wandb offline
# python -m torch.distributed.run --nnodes=1 --nproc_per_node=1 --master_port=20001 \
nohup python \
    /home/cjz/SFTonGFM/graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --pretrain_graph_mlp_adapter ${tuned_proj} \
    --tune_graph_mlp_adapter True \
    --tune_graph_tower True \
    --use_graph_prompt True \
    --graph_select_layer -2 \
    --use_graph_start_end True\
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb > /home/cjz/graphgpt_stage2_prompt.out 2>&1 &
