root_dir="../../"
pretrained=f"{root_dir}checkpoints/vit_base_patch32_384.npz"

model_type = "sscd"
img_width = 384
preprocess = "efficientnet" # imagenet, clip, efficientnet

model = dict(
    type='SimpleContrastRecognizer',
    backbone=dict(
        type="SSCDModel",
        name="vit_base_patch32_384",  # resnext101_32x4d  resnet50
        pool_param=3.,
        pool="gem",
        pretrained=pretrained,
        use_classify=False,
        dims=(768, 512),
        add_head=True
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
