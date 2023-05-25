python3 vid2jpg_zip.py

# train_vsc
python3 zip2lmdb.py \
    --vid_path "../data/meta/train/train_vids.txt" \
    --vid_pattern "../data/jpg_zips/%s/%s.zip" \
    --lmdb_path "../data/lmdb/train_vsc" \
    --output_path "../data/lmdb/train_vsc/meta"

# vsc
python3 zip2lmdb.py \
    --vid_path "../data/meta/vids.txt" \
    --vid_pattern "../data/jpg_zips/%s/%s.zip" \
    --lmdb_path "../data/lmdb/vsc" \
    --output_path "../data/lmdb/vsc/meta"