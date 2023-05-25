projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

config=./config_v115.py
workdir=./

# python3 -m torch.distributed.launch --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node 8 descriptor_train.py --config $config \
python3 -m torch.distributed.launch --nproc_per_node ${gpu_count} descriptor_train.py --config $config \
--work_dir $workdir  \
--batch_size 120 \
--num_workers 16 \
--epochs 40 \
--warmup_ratio 0.05 \
--t 0.05 \
--lr 1e-4 \
--entropy_weight 30 \
--seed 95288 \
--fp16 \
--checkpointing

#--mixup
