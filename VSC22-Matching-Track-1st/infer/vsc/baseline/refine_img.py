import argparse
import io
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy as sp
from scipy.special import softmax


def main(args):
    data = np.load(args.input_file)
    vids = data["video_ids"]
    timestamps = data["timestamps"]
    features = data["features"]

    subimg_df = pd.read_csv(args.sub_img_path, sep="\t", header=None, names=["vid", "img_num"])
    special_vid2num = dict()
    for i in range(subimg_df.shape[0]):
        special_vid2num[subimg_df.loc[i, "vid"]] = int(subimg_df.loc[i, "img_num"])

    data = list(zip(vids, timestamps, features))
    df = pd.DataFrame(data, columns=["video_ids", "timestamps", "features"])
    agg = df.groupby(by="video_ids")
    zip_handler = zipfile.ZipFile(args.zip_feat_path, 'r')
    new_data = []
    for agg_name, agg_df in tqdm(agg):
        if agg_name in special_vid2num:

            agg_df.index = range(agg_df.shape[0])
            num_indexes = special_vid2num[agg_name]

            vecs = []
            for j in range(agg_df.shape[0]):
                sub_vecs = []
                for i in range(num_indexes):
                    pattern = f"{agg_name}_{j}_{i}"
                    frame_feature = np.load(io.BytesIO(zip_handler.read(pattern)), allow_pickle=True).astype(np.float32)
                    sub_vecs.append(frame_feature)
                sub_vecs = np.stack(sub_vecs, axis=0)
                vecs.append(sub_vecs)
            vecs = np.stack(vecs, axis=0)  # frames, 2 or 4, 512

            weight_pattern = args.np_prefix % agg_name
            weights = np.load(weight_pattern)  # (frames, 2 or 4, 1)

            avg_output = vecs.mean(axis=1)  # average

            # weights_prob = 1/(1 + np.exp(-weigths))
            # weights_prob = np.transpose(weights_prob, axes=[1, 0, 2])  # (frames, 2 or 4, 1)
            weights_prob = softmax(np.transpose(weights, axes=[1, 0, 2]), axis=1)
            # mean_weights = weights_prob.mean(axis=0)
            # if mean_weights.max() > 0.8:
            # weight_avg_output = vecs[:, np.argmax(mean_weights)]
            weight_avg_output = (weights_prob * vecs).sum(axis=1) / weights_prob.sum(axis=1)

            agg_df = pd.DataFrame(list(zip(agg_df["video_ids"], agg_df["timestamps"], weight_avg_output)),
                                  columns=["video_ids", "timestamps", "features"])

            new_data.append(agg_df)
        else:
            new_data.append(agg_df)

    zip_handler.close()

    new_df = pd.concat(new_data, ignore_index=True)
    vids = np.array(new_df["video_ids"]).astype('<U7')
    features = np.array(new_df["features"].tolist(), dtype=np.float32)
    timestamps = np.array(new_df["timestamps"].tolist(), dtype=np.int64)
    np.savez(args.output_file, video_ids=vids, features=features, timestamps=timestamps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAIN parameters')
    parser.add_argument('--zip_feat_path', type=str, default="/mnt/nanjing3cephfs/mmvision/tyewang/projects/runs/vsc2022/descriptor/v32/sub_img.zip")
    parser.add_argument('--np_prefix', type=str, default="/mnt/nanjing3cephfs/mmvision/feipengma/data/vsc22/vids_with_subimages/scores/%s.npy")
    parser.add_argument('--output_file', type=str, default="/mnt/nanjing3cephfs/mmvision/tyewang/projects/runs/vsc2022/descriptor/v32/train_query_new.npz")
    parser.add_argument('--input_file', type=str, default="/mnt/nanjing3cephfs/mmvision/tyewang/projects/runs/vsc2022/descriptor/v32/train_query.npz")
    parser.add_argument('--sub_img_path', type=str, default="/mnt/nanjing3cephfs/mmvision/feipengma/data/vsc22/vids_with_subimages/vids_subimages.txt")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    main(args)
