#!/bin/sh

workdir=$(pwd -P)
echo ${workdir}

# Preprocess
cd preprocess
source preprocess.sh
cd ${workdir}

# train models
cd train/train_vid_score
source train_vid_score.sh
python3 torch2scripts.py
cd ${workdir}

cd train/train_v68
source train_v68.sh
python3 torch2scripts.py
cd ${workdir}

cd train/train_v106
source train_v106.sh
python3 torch2scripts.py
cd ${workdir}

cd train/train_v107
source train_v107.sh
python3 torch2scripts.py
cd ${workdir}

cd train/train_v115
source train_v115.sh
python3 torch2scripts.py
cd ${workdir}

cd infer
source infer_ref.sh
mv outputs/train_refs.npz data/
source infer_query.sh
cd ${workdir}


