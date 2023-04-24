import pandas as pd
import numpy as np
import sys


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    id2score = dict()
    with open("/mnt/nanjing3cephfs/mmvision/data/video/matches/vsc22/query_pred_scores.csv", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            idx, score = line.strip().split(",")
            score = float(score)
            id2score[idx] = score

    small_score_set = set()
    for k, v in id2score.items():
        if v < 0.1:
            small_score_set.add(k)

    data = np.load(input_path)
    vids = data["video_ids"]
    timestamps = data["timestamps"]
    features = data["features"]

    data = list(zip(vids, timestamps, features))
    df = pd.DataFrame(data, columns=["video_ids", "timestamps", "features"])

    print(df.shape)
    print(df.dtypes)

    agg = df.groupby(by="video_ids")
    new_data = []
    for agg_name, agg_df in agg:
        if agg_name in small_score_set:
            agg_df.index = range(agg_df.shape[0])
            agg_df = agg_df.loc[:0, :]
            agg_df.loc[0, "features"] = np.random.uniform(-0.00001, 0.00001, size=features.shape[1])

        new_data.append(agg_df)

    new_df = pd.concat(new_data, ignore_index=True)

    vids = np.array(new_df["video_ids"]).astype('<U7')
    features = np.stack(new_df["features"].tolist(), axis=0).astype(np.float32)
    timestamps = np.array(new_df["timestamps"].tolist(), dtype=np.int64)

    print(vids.shape, features.shape, timestamps.shape)
    print("Max limit 346002")

    np.savez(output_path, video_ids=vids, features=features, timestamps=timestamps)


if __name__ == '__main__':
    main()




