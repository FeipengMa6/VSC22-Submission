projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# index=(0 1 2 3)
index=(0 1 2 3)
models=("swinv2_v115" "swinv2_v107" "swinv2_v106" "vit_v68")
img_sizes=(256 256 256 384)


jpg_zips_path=""../data/jpg_zips""

for i in ${index[@]}
do
# train
python3 -m torch.distributed.launch --nproc_per_node=${gpu_count} extract_ref_feats.py \
        --zip_prefix ${jpg_zips_path}  \
        --input_file "train/train_ref_vids.txt" \
        --save_file train_refs \
        --save_file_root ./outputs/${models[i]} \
        --batch_size 2 \
        --input_file_root "../data/meta/" \
        --dataset "vsc" \
        --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
        --transform "vit" \
        --img_size ${img_sizes[i]}

# test
python3 -m torch.distributed.launch --nproc_per_node=${gpu_count} extract_ref_feats.py \
        --zip_prefix ${jpg_zips_path}  \
        --input_file "test/test_ref_vids.txt" \
        --save_file test_refs \
        --save_file_root ./outputs/${models[i]} \
        --batch_size 2 \
        --input_file_root "../data/meta/" \
        --dataset "vsc" \
        --checkpoint_path ../checkpoints/${models[i]}.torchscript.pt \
        --transform "vit" \
        --img_size ${img_sizes[i]}
done

# concat and reduce dim, finally sn
python3 concat_pca_sn.py