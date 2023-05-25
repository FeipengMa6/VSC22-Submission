projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
config=./config_v106.py
workdir=./
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# mkdir -p $workdir
# cp -r $0 $workdir
# cp -r $config $workdir
# cp -r $projectdir $workdir

# export NCCL_DEBUG=INFO
# python3 -m torch.distributed.launch --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nnodes=$WORLD_SIZE --node_rank=$RANK --nproc_per_node 8 descriptor_train.py --config $config \
python3 -m torch.distributed.launch --nproc_per_node ${gpu_count} descriptor_train.py --config $config \
--work_dir $workdir  \
--batch_size 100 \
--num_workers 8 \
--epochs 40 \
--warmup_ratio 0.05 \
--t 0.05 \
--lr 1e-4 \
--entropy_weight 30 \
--seed 95281 \
--fp16 \
--checkpointing

#--mixup
