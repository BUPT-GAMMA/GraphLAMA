model_path=/home/cjz/vicuna-7b-v1.5-16k
instruct_ds=/home/cjz/MoleculeSTM/data/PubChemSTM_data/molecule_match_train.json
graph_data_path=/home/cjz/MoleculeSTM/data/PubChemSTM_data/molecule_match_train_feature.json
pretra_gnn=/home/cjz/GraphGPT/clip_gt_arxiv
output_model=/home/cjz/SFTonGFM/checkpoints/molecule_pretrained_3
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH='/home/cjz/SFTonGFM'

wandb offline
# nohup python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=20001 \
nohup python \
    /home/cjz/SFTonGFM/graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --tune_graph_tower False\
    --graph_select_layer -2 \
    --use_graph_start_end \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --use_graph_prompt True \
    --combined_graph_prompt True\
    --report_to wandb > /home/cjz/SFTonGFM/outputs/SFT_molecule_pretrain_3.out 2>&1 &