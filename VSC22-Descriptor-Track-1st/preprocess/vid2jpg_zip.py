import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import Pool
import numpy as np
import os
import subprocess
import shutil
import zipfile

def convert_mp4_to_zip_ffmpeg(mp4_path: str,
                              zip_path: str,
                              fps: int = None,
                              uniform_num_frames: int = None,
                              scale: int = None,
                              start_time: int = 0,
                              quality: int = 2,
                              duration: int = None,
                              override: bool = True):
    if os.path.isfile(zip_path):
        if override:
            os.remove(zip_path)
        else:
            return 0
    
    output_dir = os.path.dirname(zip_path)
    tmp_dir = os.path.join(output_dir, '{}_tmp'.format(os.path.basename(zip_path)[:-4]))
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)


    cmd = f"ffmpeg -nostdin -y -i {mp4_path} -start_number 0 -q 0 -vf fps=1 {tmp_dir}/%07d.jpg"
    try:
        with open(os.devnull, "w") as null:
            subprocess.call(cmd, shell=True, timeout=60, stderr=null)
    except subprocess.TimeoutExpired:
        shutil.rmtree(tmp_dir)
        return -1
    else:
        imgpaths = [os.path.join(tmp_dir, fn) for fn in os.listdir(tmp_dir) if fn.endswith('.jpg')]
        #inds = np.linspace(0, len(imgpaths) - 1, uniform_num_frames, dtype=int).tolist()
        #imgpaths = [imgpaths[i] for i in inds]
        if len(imgpaths) != 0:
            with zipfile.ZipFile(zip_path, 'w') as wzip:
                for p in imgpaths:
                    name = p.split("/")[-1]
                    wzip.write(p, arcname=name)
        shutil.rmtree(tmp_dir)

    return 0

def do_convert(args):
    # mode, token, idx, fps = args
    mode, token, idx, fps = args
    video_path = "../data/videos/%s/%s/%s.mp4" % (mode, token, idx)
    zip_path = "../data/jpg_zips/%s/%s.zip" % (idx[-2:], idx)
    convert_mp4_to_zip_ffmpeg(video_path, zip_path, fps=fps)

def main():
    train_query_meta = "../data/meta/train/train_query_metadata.csv"
    train_ref_meta = "../data/meta/train/train_reference_metadata.csv"
    test_query_meta = "../data/meta/test/test_query_metadata.csv"
    test_ref_meta = "../data/meta/test/test_reference_metadata.csv"

    meta_query = pd.read_csv(train_query_meta)
    meta_ref = pd.read_csv(train_ref_meta)
    meta_query_ = pd.read_csv(test_query_meta)
    meta_ref_ = pd.read_csv(test_ref_meta)
    
    meta = pd.concat([meta_query, meta_ref], ignore_index=True)
    meta["mode"] = "train"

    meta_ = pd.concat([meta_query_, meta_ref_], ignore_index=True)
    meta_["mode"] = "test"
    meta = pd.concat([meta, meta_], ignore_index=True)

    video_list = meta["video_id"].tolist()
    modes = meta["mode"].tolist()

    args_list = []
    for idx, m in zip(video_list, modes):
        t = "query" if idx.startswith("Q") else "reference"

        args_list.append((m, t, idx, 1))

    print("#####", len(video_list))

    with Pool(16) as pool:
        pool.map(do_convert, tqdm(args_list))
    pool.join()

if __name__ == "__main__":
    main()



