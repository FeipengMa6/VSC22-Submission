projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
config=./config_vid_score.py
workdir=./

# extract features first
python3 -m torch.distributed.launch  --nproc_per_node 2 video_score_train.py --config $config \
--save_file ../../data/feat_zip/feats.zip \
--zip_prefix ../../data/jpg_zips \
--input_file ../../data/meta/vids.txt \
--clip_dir ../../checkpoints/clip_vit-l-14 \
--bs 8 \
--max_video_frames 256 \



# train video score model
python3 -m torch.distributed.launch  --nproc_per_node 2 video_score_train.py --config $config \
--work_dir $workdir  \
--batch_size 64 \
--num_workers 8 \
--epochs 10 \
--warmup_ratio 0.1 \
--lr 5e-5 \
--fp16 \
--print_freq 10 \
--eval_freq 100 \
--bert_path ../../checkpoints/chinese-roberta-wwm-ext-base \
--val_ann_path ../../data/meta/train/train_ground_truth.csv \