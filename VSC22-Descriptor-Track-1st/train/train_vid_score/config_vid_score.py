root_dir="../../"
img_width=224
feat_dim = 1024
bert_dim = 768
gradient_checkpointing = False
max_frames = 256
output_dim = 256
bert_path = "../../checkpoints/chinese-roberta-wwm-ext-base" 

data = dict(
    train=dict(
        type="LabelFeatZipDataSet",
        vids_path=f"{root_dir}/data/meta/train/train_query_id.csv",  
        feat_zip_path=f"{root_dir}/data/feat_zip/feats.zip", #TODO
        max_frames=256,
        num_workers=8,
        ann_vids_path=f"{root_dir}/data/meta/train/train_positive_query.txt"
    ),
    val=dict(
        type="LabelFeatZipDataSet",
        vids_path=f"{root_dir}/data/meta/val/val_query_id.csv",
        feat_zip_path=f"{root_dir}/data/feat_zip/feats.zip", #TODO
        max_frames=256,
        num_workers=8,
        ann_vids_path=f"{root_dir}/data/meta/train/train_positive_query.txt"
        ) 
)
