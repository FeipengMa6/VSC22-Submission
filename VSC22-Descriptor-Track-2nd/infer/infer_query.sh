projectdir=./
export PYTHONPATH=$PYTHONPATH:$projectdir

python3 extract_query_feats.py --split train
python3 extract_query_feats.py --split val
python3 extract_query_feats.py --split test
