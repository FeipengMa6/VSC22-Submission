import cv2
import tqdm
import dataclasses
import numpy as np
from sklearn.linear_model import RANSACRegressor

def calclualte_low_var_dim(score_norm_refs):
    sn_features = np.concatenate([ref.feature for ref in score_norm_refs], axis=0)
    low_var_dim = sn_features.var(axis=0).argmin()
    return low_var_dim

def transform_features(features, transform):
    return [
        dataclasses.replace(feature, feature=transform(feature.feature))
        for feature in features
    ]

def generate_candidates_classfiy_feature(
        query,
        ref,
        candidate_list,
        query_video_len_map,
):
    features = []
    infos = []
    for qid, rid, score in tqdm.tqdm(candidate_list):
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
        infos.append([qid, rid, score])
        features.append([np.matmul(rfeat, qfeat.T)])
        infos.append([qid, rid, score])
    return features, infos


def generate_matching_feature(
        query,
        ref,
        query_video_len_map,
        candidate_score_list
):
    res_list = []
    for qid, rid, score in tqdm.tqdm(candidate_score_list):
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
        res_list.append([qid, rid, qfeat, rfeat])
    return res_list


def generate_matching_result(
        res_list,
        threshold=0.05,
        std_ratio=2):
    match_res = []
    for qid, rid, sim_mat, sim_mat_org in tqdm.tqdm(res_list):
        qmat = sim_mat > threshold
        binary_mat = (sim_mat > threshold).astype(np.uint8) * 255
        conn_res = cv2.connectedComponentsWithStats(binary_mat, connectivity=8)
        num_label, conn_label, stats, centroids = conn_res
        label_cnt = {}
        for i in range(1, num_label):
            cnt = (conn_label == i).sum()
            if cnt > 10:
                label_cnt[i] = cnt
                x_, y_ = np.where(conn_label == i)
                qmat[x_, y_] = False
        if not label_cnt:
            conn_label = qmat.astype(np.int32)
            label_cnt[1] = conn_label.sum()
        match_res_ = []
        for i in label_cnt:
            x, y = np.where((conn_label == i) + qmat)
            if len(set(x)) > 3:
                ransac = RANSACRegressor(max_trials=200, random_state=2023, residual_threshold=2)
                prob = sim_mat[x, y]
                ransac.fit(x[:, np.newaxis], y[:, np.newaxis], sample_weight=np.square(prob))
                pred = ransac.predict(x[:, np.newaxis]).flatten()
                qualify = abs(y - pred) < 1
                coef = ransac.estimator_.coef_[0][0]
                if coef <= 0:
                    continue
                coef = max(1/coef, coef)
                if qualify.sum() > 5 and len(set(x[qualify])) > 3 and len(set(y[qualify])) > 3:
                    qs, qe = x[qualify][0], x[qualify][-1]
                    rs, re = y[qualify][0], y[qualify][-1]
                    top_sim = sim_mat[x[qualify], y[qualify]]
                    score = top_sim.max() - top_sim.std() * std_ratio - abs(coef - 1) / 10 
                    match_res_.append([qs, rs, qe, re, score])
        for qs, rs, qe, re, score in match_res_:
            match_res.append([qid, rid, qs, rs, qe, re, score])
    return match_res