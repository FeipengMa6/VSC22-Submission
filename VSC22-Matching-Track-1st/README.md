# Solution of VSC-2022-Matching-Track (Rank 1)
The code reproduce the results of the 1st solution of VSC-2022-Matching-Track

## Dependencies
We use conda to manage our runtime environment, please use the instructions below to create a virtual conda environment.
```
conda env create -f environment.yml
conda activate matching
```

## Prepare

The matching track heavily rely on decriptor track. To reproduce the result, you should first train the descriptor model, extract frame embeddings of train query/ref videos. The embedding required during train and inference are evenly extracted at one frame per second.   

Some resource and directory rely on Descriptor stages directly, and it shows as below:

```
${work_dir}/
├── checkpoints/ # shared with descriptor stage. You should make a soft-link to Descriptor `checkpoints` directory.
├── data/ # shared with descriptor stage. You should make a soft-link to Descriptor `data` directory.
├── infer
│   ├── outputs # shared with descriptor stage. You should make a soft-link to Descriptor `infer/outputs` directory.
├── train 
└── vsc
```

Before start the Matching task, be sure the following files have been generated:

- The extracted ref features(from the Descriptor track):
    - ./infer/outputs/swinv2_v115/train_refs.npz
    - ./infer/outputs/swinv2_v107/train_refs.npz
    - ./infer/outputs/swinv2_v106/train_refs.npz
    - ./infer/outputs/vit_v68/train_refs.npz

- The extracted query features(from the Descriptor track):
   - ./infer/outputs/swinv2_v115/train_query.npz
   - ./infer/outputs/swinv2_v107/train_query.npz
   - ./infer/outputs/swinv2_v106/train_query.npz
   - ./infer/outputs/vit_v68/train_query.npz

- One validation ref embedding that used to generate candidates recall file: `./infer/outputs/matching/candidates.csv`. The candidates file used from Descriptor stage is to generate samples for Matching stage. The raw recall is large and our methods do not relay on candidates recall score. We only use one model to get recall pairs. The file we use is:
    - ./infer/outputs/vit_v68/test_refs.npz

- generate candidates recall file(`./infer/outputs/matching/candidates.csv`): 

``` bash
python -m vsc.baseline.sscd_baseline \
    --query_features ./infer/outputs/vit_v68/train_query.npz \
    --ref_features ./infer/outputs/vit_v68/train_refs.npz \
    --score_norm_features ./infer/outputs/vit_v68/test_refs.npz \
    --ground_truth ./data/meta/train/train_matching_ground_truth.csv \
    --output_path ./infer/outputs/matching \
    --overwrite

```

- train model:

```bash
cd train
bash train.sh
```

- infer model. The output result file: `./infer/outputs/matching/test_matching.csv`
```
cd infer
bash infer.sh
```

## Results
| User or team | Phase1 uAP |  Phase2 uAP |
| :----| :----:|:----: |
| **Ours** | 0.9290 | 0.9153|
| CompetitionSecond | 0.8206 | 0.8020 |
| cvl-matching | 0.7727 | 0.7759 |



## Resource consumption

|item |required  |
| :---| :----|
| GPU | 1 A100 |
| Memory | 40GB per A100|
| Train duration | ~3 Hours | 
| Inference duration | ~1 vid/s | 

## Citation
If you find this code helpful, please cite our paper,
```

```