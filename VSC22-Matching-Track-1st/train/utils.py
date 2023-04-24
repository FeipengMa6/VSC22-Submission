import pickle
import tqdm
import pandas as pd
import numpy as np
from functools import lru_cache


class VideoFeature:

    def __init__(self, file, normalize=False):
        data = np.load(file, allow_pickle=False)
        self.video_ids = data["video_ids"]
        self.feats = data["features"].astype(np.float32)
        self.timestamps = data["timestamps"]
        self.normalize = normalize
        if normalize:
            self.feats /= np.linalg.norm(self.feats, axis=1, keepdims=True)

    @lru_cache(maxsize=41000)
    def __getitem__(self, item):
        to_select = self.video_ids == item
        feats = self.feats[to_select]
        ts = self.timestamps[to_select]
        return feats, ts


def generate_candidates_classfiy_feature(
        # query_embed_file,
        # ref_embed_file,
        query,
        ref,
        candidate_file,
        to_save_meta_file,
        query_video_len_map,
        gt_file=None
):
    if isinstance(query, list):
        query = {vf.video_id:vf.feature for vf in query}
    if isinstance(ref, list):
        ref = {vf.video_id: vf.feature for vf in ref}
    cd_df = pd.read_csv(candidate_file)
    if gt_file is not None:
        gt_df = pd.read_csv(gt_file)
        gt_df = gt_df[['query_id', 'ref_id']]
        gt_df = gt_df[gt_df.duplicated() == False]
        gt_df['label'] = 1
        cd_df = pd.merge(cd_df, gt_df, how='left')
        cd_df.fillna(0, inplace=True)
    else:
        cd_df['label'] = 0
    # query = VideoFeature(query_embed_file, normalize=True)
    # ref = VideoFeature(ref_embed_file, normalize=True)
    features = []
    infos = []
    tasks = list(zip(cd_df.query_id.values, cd_df.ref_id.values, cd_df.label.values))
    for qid, rid, label in tqdm.tqdm(tasks):
        num_data = query_video_len_map[qid]
        qfeat = query[qid]
        rfeat = ref[rid]
        sim_mat = np.matmul(qfeat, rfeat.T)
        if num_data != len(qfeat):
            start = 0
            score_max = []
            feat_list = []
            while start < len(qfeat):
                maxs = sim_mat[start: start + num_data].max(1)
                maxs.sort()
                score_max.append(maxs[-10:].mean())
                feat_list.append(qfeat[start:start + num_data])
                start += num_data
            qfeat = feat_list[np.argmax(score_max)]
        features.append(np.matmul(qfeat, rfeat.T))
        infos.append([qid, rid, label])
        features.append([np.matmul(rfeat, qfeat.T)])
        infos.append([qid, rid, label])
    with open(to_save_meta_file, 'wb') as f:
        pickle.dump({'feature': features, 'infos': infos}, f)


def generate_matching_feature(
        query,
        ref,
        query_video_len_map,
        candidate_score_file,
        to_save_match_meta_file,
        select_threshold=0.005,
        gt_file=None
):
    if isinstance(query, list):
        query = {vf.video_id:vf.feature for vf in query}
    if isinstance(ref, list):
        ref = {vf.video_id: vf.feature for vf in ref}
    candidate_df = pd.read_csv(candidate_score_file)
    select_cand = candidate_df.groupby(['query_id', 'ref_id']).prob.max()
    select_cand = select_cand.reset_index(None)
    select_cand = select_cand[select_cand.prob > select_threshold]
    if gt_file is not None:
        gt_df = pd.read_csv(gt_file)
        select_cand = pd.merge(select_cand, gt_df, how='left')
    # query = VideoFeature(query_embed_file, normalize=True)
    # ref = VideoFeature(ref_embed_file, normalize=True)
    else:
        select_cand['query_start'] = np.nan
    res_list = []
    for (qid, rid), sdf in tqdm.tqdm(select_cand.groupby(['query_id', 'ref_id'])):
        num_data = query_video_len_map[qid]
        qfeat = query[qid]
        rfeat = ref[rid]
        sim_mat = np.matmul(qfeat, rfeat.T)
        if num_data != len(qfeat):
            start = 0
            score_max = []
            feat_list = []
            while start < len(qfeat):
                maxs = sim_mat[start: start + num_data].max(1)
                maxs.sort()
                score_max.append(maxs[-10:].mean())
                feat_list.append(qfeat[start:start+num_data])
                start += num_data
            qfeat = feat_list[np.argmax(score_max)]
        gt_list = []
        sdf.dropna(inplace=True)
        sdf.dropna(inplace=True)
        if len(sdf) == 0:
            res_list.append([qid, rid, qfeat, rfeat, gt_list])
            continue
        for qstart, qend, rstart, rend in zip(sdf.query_start.values, sdf.query_end.values, sdf.ref_start.values,
                                              sdf.ref_end.values):
            gt_list.append([qstart, qend, rstart, rend])
        res_list.append([qid, rid, qfeat, rfeat, gt_list])
    if to_save_match_meta_file:
        with open(to_save_match_meta_file, 'wb') as f:
            pickle.dump(res_list, f)
    else:
        return res_list



