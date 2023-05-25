root_dir="../../"
pretrained=f"{root_dir}/checkpoints/swinv2_base_patch4_window12to16_192to256_22kto1k_ft.pth"

img_width = 256
model_type = "swinv2"
preprocess = "vit"

model = dict(
    type='SimpleContrastRecognizer',
    backbone=dict(
        type='SwinTransformerV2',
        img_size=img_width,
        patch_size=4,
        window_size=16,
        num_heads=[ 4, 8, 16, 32 ],
        embed_dim=128,
        depths=[ 2, 2, 18, 2 ],
        pretrained_window_sizes=[ 12, 12, 12, 6 ],
        drop_path_rate=0.2,
        pretrained=pretrained,
        output_dim=512,
        p=3.,
        use_checkpoint=True
    )
)


data = dict(
    train=dict(
        type="LabelVideoLmdbDataSet", # VideoLmdbDataSet LabelVideoLmdbDataSet
        vids_path=[f"{root_dir}/data/meta/train/train_ref_vids.txt", f"{root_dir}/data/meta/train/train_query_id.csv"],
        meta_path=f"{root_dir}/data/lmdb/vsc/meta.npz",
        lmdb_path=f"{root_dir}/data/lmdb/vsc",
        lmdb_size=1e12,
        preprocess=preprocess,
        width=img_width,
        ann_path=f"{root_dir}/data/meta/train/train_matching_ground_truth.csv",
        arg_lmdb_path=f"{root_dir}/data/lmdb/train_vsc",
        probs=(0.6, 0.4),
        crop=0.8,
        mixup=0.2
    ),
    train1=dict(
        type="VideoLmdbDataSet", # VideoLmdbDataSet LabelVideoLmdbDataSet
        vids_path=[],
        meta_path=f"{root_dir}/data/lmdb/dics/meta.npz",
        lmdb_path=f"{root_dir}/data/lmdb/dics",
        lmdb_size=1e12,
        preprocess=preprocess,
        width=img_width,
    ),
)
