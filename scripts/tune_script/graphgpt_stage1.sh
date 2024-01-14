# to fill in the following path to run the first stage of our GraphGPT!
model_path=/home/cjz/vicuna-7b-v1.5-16k
instruct_ds=/home/cjz/GraphGPT/graph_matching/train_instruct_graphmatch.json
graph_data_path=/home/cjz/GraphGPT/graph_data/graph_data_all.pt
pretra_gnn=/home/cjz/GraphGPT/clip_gt_arxiv
output_model=/home/cjz/GraphGPT/checkpoints/stage_1
export CUDA_VISIBLE_DEVICES=1

wandb offline
# nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=20001 \
nohup python \
    /home/cjz/GraphGPT/graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb > /home/cjz/graphgpt.out 2>&1 &