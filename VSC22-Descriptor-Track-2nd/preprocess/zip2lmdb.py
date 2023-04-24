import os

import lmdb
import zipfile
from tqdm import tqdm
import numpy as np
import argparse

def main(args):
    vid_path = args.vid_path#"./data/meta/train/train_vids.txt"
    vid_pattern = args.vid_pattern#"./data/jpg_zips/%s/%s.zip"
    lmdb_path = args.lmdb_path#"./data/lmdb/train_vsc"
    output_path = args.output_path#"./data/lmdb/train_vsc/meta"
    
    lmdb_mapsize = 1e12

    with open(vid_path, "r", encoding="utf-8") as f:
        feeds = [line.strip() for line in f if line.strip()]
        feeds.sort()
        print("### Total feeds %d" % len(feeds))
        print("### Samples: %s" % " ".join(feeds[:5]))

    os.makedirs(os.path.dirname(vid_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    global_idx = 0
    vids = []
    intervals = []
    with lmdb.open(lmdb_path, map_size=lmdb_mapsize) as env:
        with env.begin(write=True) as doc:
            for vid in tqdm(feeds):
                start = global_idx
                try:
                    zip_path = vid_pattern % (vid[-2:], vid)
                except Exception as e:
                    zip_path = vid_pattern % vid

                if not os.path.exists(zip_path):
                    continue

                try:
                    fn_num = 0
                    with zipfile.ZipFile(zip_path, 'r') as handler:
                        img_name_list = handler.namelist()
                        img_name_list = sorted(img_name_list)

                        for img_name in img_name_list:
                            content = handler.read(img_name)
                            doc.put(str(global_idx).encode(), content)
                            global_idx += 1
                            fn_num += 1

                    end = start + fn_num
                    vids.append(vid)
                    intervals.append([start, end])

                except Exception as e:
                    print(e)
                    continue

        number_entries = env.stat()['entries']
        print("Total entries %d" % number_entries)

    vids = np.array(vids)
    intervals = np.array(intervals)
    np.savez(output_path, vids=vids, intervals=intervals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_path",type=str,default="")
    parser.add_argument("--vid_pattern",type=str,default="")
    parser.add_argument("--lmdb_path",type=str,default="")
    parser.add_argument("--output_path",type=str,default="")
    args = parser.parse_args()
    main(args)
