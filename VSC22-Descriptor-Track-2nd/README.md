# Solution of VSC-2022-Descriptor-Track (Rank 1)
The code reproduce the results of the 1st solution of VSC-2022-Descriptor-Track

## Download dataset and pretrained models 
Please download the [VSC-2022](https://www.drivendata.org/competitions/101/meta-video-similarity-descriptor/page/570/) dataset into `data/videos/`.
Or make a softlink to your VSC-2022 dataset
```
ln -s path/to/vsc22/train data/videos/train
ln -s path/to/vsc22/test data/videos/test
```
 <!-- download the [DISC-2021](https://ai.facebook.com/datasets/disc21-downloads/) dataset into `data/images/`. -->

Download [checkpoints](https://drive.google.com/file/d/1GL0xhTTSHav_iG79yJ1jqgQcmuJFs_lF/view?usp=sharing) into `checkpoints/`.

## Shortcut
You can train and evaluate the model in a quick way.
```
bash train.sh
bash eval.sh
```
Or train and evaluate step by step.
## Train step by step

### 1. Preprocess
To speed up training process, we preprocess the raw data before training models.
```
cd preprocess
bash preprocess.sh
```

You will get the following dir tree of data before training.

```
data/
├── feat_zip
│   └── feats.zip # (will be generated after train_vid_score)
├── jpg_zips
├── lmdb
│   ├── dics
│   ├── train_vsc
│   └── vsc
├── meta
│   ├── test
│   ├── train
│   ├── val
│   └── vids.txt
├── images
└── videos
    ├── test
    │   ├── query
    │   └── reference
    └── train
        ├── query
        └── reference
```

### 2. Train models and create serializable models

We should train 6 models, 4 of them are used for ensemble, one for pca, the last one for predict the video score of each video.

Train video score model
```
cd train/train_vid_score
bash train_vid_score.sh
python3 torch2scripts.py
```

Train v68 model
```
cd train/train_v68
bash train_v68.sh
python3 torch2scripts.py
```

Train v106 model
```
cd train/train_v106
bash train_v106.sh
python3 torch2scripts.py
```

Train v107 model
```
cd train/train_v107
bash train_v107.sh
python3 torch2scripts.py
```

Train v115 model
```
cd train/train_v115
bash train_v115.sh
python3 torch2scripts.py
```

The PCA model is trained use train ref features in reference stage.

## Inference step by step

### 1. Extract reference vids features and train PCA model.
```
cd infer
bash infer_ref.sh
```
You will get `train_refs.npz`,`train_refs_sn.npz`,`test_refs.npz` and `test_refs_sn.npz` in `infer/outputs`. Then place `train_refs.npz` into `infer/data/`.
```
cd infer
mv outputs/train_refs.npz data/
```
### 2. Extract query vids features
```
cd infer
bash infer_query.sh
```
You will get `train_query_sn.npz` and `test_query_sn.npz` in `infer/outputs`.
### 3. Evaluation
```
cd infer
bash eval.sh
```

## Results
| User or team | Phase1 uAP |  Phase2 uAP |
| :----| :----:|:----: |
| **Ours** | **0.9176** | **0.8717** |
| FriendshipFirst | 0.9197 | 0.8514|
| cvl-descriptor | 0.8534 | 0.8362 |
| Zihao | 0.7841 | 0.7729|



## Resource consumption

|item |required  |
| :---| :----|
| GPU | 4*8 A100 |
| Memory | 40GB per A100|
| Train duration | ~20min/epoch | 
| Inference duration | ~1 vid/s | 

## Citation
If you find this code helpful, please cite our paper,
```

```
