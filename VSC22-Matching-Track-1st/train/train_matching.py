import numpy as np
import torch
import os
os.environ['TORCH_HOME'] = '../'
import pickle
import pandas as pd
import tqdm
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import cv2
from sklearn.metrics import classification_report
import sys
sys.path.append('../')
from dataset import MatchClassifyDataset, MatchingRefineDataset
from models import ClassifyModel, HRnet
from utils import VideoFeature, generate_candidates_classfiy_feature, generate_matching_feature
from vsc.storage import load_features
from sklearn.preprocessing import normalize
from vsc.baseline.score_normalization import transform_features
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from vsc.metrics import evaluate_matching_track

seed = 2023

def generate_matching_result(
        res_list,
        threshold=0.05,
        std_ratio=2):
    match_res = []
    match_map = {}
    for qid, rid, sim_mat, label, sim_mat_org in res_list:
        key = (qid, rid)
        if key in match_map:
            match_map[key][0] = match_map[key][0] + sim_mat
            match_map[key][1] = match_map[key][1] + label
            match_map[key][2] = match_map[key][2] + sim_mat_org
            match_map[key][3] += 1
        else:
            match_map[key] = [sim_mat, label, sim_mat_org, 1]
    for (qid, rid), (sim_mat, label, sim_mat_org, num_map) in tqdm.tqdm(match_map.items()):
        sim_mat = sim_mat / num_map
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
                    # top_sim = sim_mat[qs:qe+1, rs:re+1].max(0)
                    score = top_sim.max() - top_sim.std() * std_ratio -abs(coef - 1) / 10 # [:len(set(x[qualify])) + 2].std() * 2 #- abs(1 - abs(coef))
                    match_res_.append([qs, rs, qe, re, score])
        for qs, rs, qe, re, score in match_res_:
            match_res.append([qid, rid, qs, rs, qe, re, score])
    return match_res


def eval_match_res(
        match_res,
        gt_file,
        select_cand,
        to_save='temp.csv'
):
    match_df = pd.DataFrame(match_res)
    match_df = match_df[match_df.duplicated() == False]
    match_df.columns = ['query_id', 'ref_id', 'qs', 'rs', 'qe', 're', 'sc']
    match_df.columns = ['query_id', 'ref_id', 'query_start', 'ref_start', 'query_end', 'ref_end', 'score']
    max_match = match_df.groupby(['query_id', 'ref_id', 'query_start', 'ref_start', 'query_end', 'ref_end']).score.max()
    match_df = max_match.reset_index(None)
    match_df.to_csv(to_save, index=False)
    match_metrics = evaluate_matching_track(gt_file, to_save)
    print(f"Matching track metric: {match_metrics.segment_ap.ap:.4f}")
    match_new = pd.merge(match_df, select_cand)
    match_new['score'] = [min([x, y]) for x, y in zip(match_new.prob.values, match_new.score.values)]
    del match_new['prob']
    match_new.to_csv(to_save, index=False)
    match_metrics = evaluate_matching_track(gt_file, to_save)
    print(f"Matching track metric: {match_metrics.segment_ap.ap:.4f}")


def classify_train_loop(model, cuda_id, optimizer, loss_func, num_epoch, trainloader, testloader, to_save_file):
    for epoch in range(num_epoch):
        model = model.train()
        acc_list = []
        loss_list = []
        for idx, (feature, label, qids, rids) in tqdm.tqdm(enumerate(trainloader)):
            feature = feature.cuda(cuda_id)
            label = label.cuda(cuda_id)
            pred = model(feature)
            loss = loss_func(pred, label.flatten())
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss = loss.item()
            label = label.cpu().numpy()
            pred = pred.detach().cpu().numpy()
            acc = (label == pred.argmax(1)).mean()
            acc_list.append(acc)
            loss_list.append(loss)
        print(f'epoch: {epoch}, iter: {idx}, current loss: {np.mean(loss_list):.5f}, acc: {np.mean(acc_list):.4f}')
        model = model.eval()
        pred_list = []
        label_list = []
        qid_list = []
        rid_list = []
        for feature, label, qids, rids in tqdm.tqdm(testloader):
            with torch.no_grad():
                pred_list.append(model(feature.cuda(cuda_id)).cpu())
                label_list.append(torch.LongTensor(label))
                qid_list.extend(list(qids))
                rid_list.extend(list(rids))
        pred = torch.cat(pred_list, dim=0)
        label = torch.cat(label_list, dim=0)
        label_arr = label.numpy().flatten()
        pred_prob = pred.softmax(axis=1)[:, 1]
        print(classification_report(label_arr, pred_prob > 0.5))
    model = model.eval()
    torch.save(model.cpu(), to_save_file)
    return model


def classify_test_loop(model, cuda_id, testloader):
    model = model.cuda(cuda_id).eval()
    pred_list = []
    label_list = []
    qid_list = []
    rid_list = []
    for feature, label, qids, rids in tqdm.tqdm(testloader):
        with torch.no_grad():
            pred_list.append(model(feature.cuda(cuda_id)).cpu())
            label_list.append(torch.LongTensor(label))
            qid_list.extend(list(qids))
            rid_list.extend(list(rids))
    pred = torch.cat(pred_list, dim=0)
    label = torch.cat(label_list, dim=0)
    pred_prob = pred.softmax(axis=1)[:, 1]
    res_df = pd.DataFrame(qid_list)
    res_df.columns = ['query_id']
    res_df['ref_id'] = rid_list
    res_df['prob'] = pred_prob
    return res_df

def classify_eval_loop(model_list, cuda_id, testloader):
    model_list = [x.cuda(cuda_id).eval() for x in model_list]
    pred_list = []
    qid_list = []
    rid_list = []
    for feature, label, qids, rids in tqdm.tqdm(testloader):
        with torch.no_grad():
            pred_list_ = [model(feature.cuda(cuda_id)).cpu() for model in model_list]
            pred_list_ = [x.softmax(axis=1)[:, 1] for x in pred_list_]
            pred_list.append(sum(pred_list_) / len(pred_list_))
            qid_list.extend(list(qids))
            rid_list.extend(list(rids))
    pred_prob = torch.cat(pred_list, dim=0)
    res_df = pd.DataFrame(qid_list)
    res_df.columns = ['query_id']
    res_df['ref_id'] = rid_list
    res_df['prob'] = pred_prob
    return res_df


def matching_train_loop(model, cuda_id, optimizer, loss_func, num_epoch, trainloader, testloader, to_save_file):
    for epoch in range(num_epoch):
        model = model.train()
        for idx, (feature, label, qid, rid, h, w) in tqdm.tqdm(enumerate(trainloader)):
            feature = feature.cuda(cuda_id)
            label = label.cuda(cuda_id)
            pred = model(feature)
            loss = loss_func(pred, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            loss = loss.item()
            if idx % 5 == 0:
                print(f'train epoch: {epoch}, iter: {idx}, current loss: {loss:.5f}')
        model = model.eval()
        pred_list = []
        label_list = []
        qid_list = []
        rid_list = []
        h_list = []
        w_list = []
        loss_list = []
        for feature, label, qid, rid, h, w in tqdm.tqdm(testloader):
            with torch.no_grad():
                pred = model(feature.cuda(cuda_id)).cpu()
                pred_list.append(pred)
                label_list.append(label)
                loss = loss_func(pred, label)
                qid_list.extend(list(qid))
                rid_list.extend(list(rid))
                h_list.extend(list(h))
                w_list.extend(list(w))
                loss_list.append(loss)
        print(f"eval epoch {epoch}, iter: {idx}, current loss: {np.mean(loss_list):.5f}")
    model = model.eval()
    torch.save(model.cpu(), to_save_file)
    return model


def matching_test_loop(model, cuda_id, testloader):
    model = model.cuda(cuda_id).eval()
    res_list = []
    for feature, label, qid, rid, h, w in tqdm.tqdm(testloader):
        with torch.no_grad():
            feature_ = feature.cuda(cuda_id)
            pred = model(feature_).cpu()
            pred = pred.softmax(dim=1)
            pred = pred.numpy()
            pred_t = model(feature_.transpose(3, 2)).cpu()
            pred_t = pred_t.softmax(dim=1)
            pred_t = pred_t.transpose(3, 2).numpy()
            pred += pred_t
            pred /= 2
            h = h.numpy()
            w = w.numpy()
        for i in range(len(pred)):
            h_, w_ = h[i], w[i]
            p = pred[i][1][:h_, :w_]
            fea = feature[i][0][:h_, :w_]
            max_p = p.max()
            max_l = label[i].max()
            res_list.append([qid[i], rid[i], p, label[i][:h_, :w_].numpy(), fea.numpy()])
    return res_list


def matching_eval_loop(model_list, cuda_id, testloader):
    model_list = [model.cuda(cuda_id).eval() for model in model_list]
    res_list = []
    for feature, label, qid, rid, h, w in tqdm.tqdm(testloader):
        with torch.no_grad():
            feature_ = feature.cuda(cuda_id)
            pred_list = []
            for model in model_list:
                pred = model(feature_).cpu()
                pred = pred.softmax(dim=1)
                pred = pred.numpy()
                pred_t = model(feature_.transpose(3, 2)).cpu()
                pred_t = pred_t.softmax(dim=1)
                pred_t = pred_t.transpose(3, 2).numpy()
                pred += pred_t
                pred /= 2
                pred_list.append(pred)
            pred = sum(pred_list) / 2
            h = h.numpy()
            w = w.numpy()
        for i in range(len(pred)):
            h_, w_ = h[i], w[i]
            p = pred[i][1][:h_, :w_]
            fea = feature[i][0][:h_, :w_]
            res_list.append([qid[i], rid[i], p, label[i][:h_, :w_].numpy(), fea.numpy()])
    return res_list


def train_classify_model(
        meta,
        model_str,
        source_dir,
        version,
        candidate_score_file,
        num_epoch=10,
        cuda_id=0,
        extra_meta=[],
        retrain=False
):
    qid_list = list(set([x[0] for x in meta['infos']]))
    qid_list.sort()
    np.random.seed(2023)
    np.random.shuffle(qid_list)
    train_ids = set(qid_list[:len(qid_list) // 2])
    train_feature, train_info = [], []
    test_feature, test_info = [], []
    for i in range(len(meta['feature'])):
        if meta['infos'][i][0] in train_ids:
            test_feature.append(meta['feature'][i])
            test_info.append(meta['infos'][i])
        else:
            train_feature.append(meta['feature'][i])
            train_info.append(meta['infos'][i])
    extra_train_feature, extra_train_info = [], []
    extra_test_feature, extra_test_info = [], []
    for emeta in extra_meta:
        for i in range(len(emeta['feature'])):
            if emeta['infos'][i][0] in train_ids:
                extra_test_feature.append(emeta['feature'][i])
                extra_test_info.append(emeta['infos'][i])
            else:
                extra_train_feature.append(emeta['feature'][i])
                extra_train_info.append(emeta['infos'][i])
    train_dataset = MatchClassifyDataset(train_feature + extra_train_feature, train_info + extra_train_info, enhance=True)
    test_dataset = MatchClassifyDataset(test_feature, test_info)
    trainloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, prefetch_factor=4, num_workers=12)
    testloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, prefetch_factor=4, num_workers=12)
    to_save_file_1 = os.path.join(source_dir, f'classify_h1_{version}.pt')
    if retrain:
        model_h1 = ClassifyModel(model_str)
        model_h1 = model_h1.cuda(cuda_id)
        optimizer = optim.Adam(model_h1.parameters(), lr=0.0001, weight_decay=5e-5)
        loss_func = nn.CrossEntropyLoss(reduce=None)
        model_h1 = classify_train_loop(model_h1, cuda_id, optimizer, loss_func, num_epoch, trainloader, testloader, to_save_file_1)
    else:
        model_h1 = torch.load(to_save_file_1)
    pred_res1 = classify_test_loop(model_h1, cuda_id, testloader)
    train_dataset = MatchClassifyDataset(train_feature, train_info)
    test_dataset = MatchClassifyDataset(test_feature + extra_test_feature, test_info + extra_test_info, enhance=True)
    trainloader = DataLoader(train_dataset, batch_size=2048, shuffle=False, prefetch_factor=4, num_workers=12)
    testloader = DataLoader(test_dataset, batch_size=2048, shuffle=True, prefetch_factor=4, num_workers=12)
    to_save_file_2 = os.path.join(source_dir, f'classify_h2_{version}.pt')
    if retrain:
        model_h2 = ClassifyModel(model_str)
        model_h2 = model_h2.cuda(cuda_id)
        optimizer = optim.Adam(model_h2.parameters(), lr=0.0001, weight_decay=5e-5)
        loss_func = nn.CrossEntropyLoss(reduce=None)
        model_h2 = classify_train_loop(model_h2, cuda_id, optimizer, loss_func, num_epoch, testloader, trainloader, to_save_file_2)
    else:
        model_h2 = torch.load(to_save_file_2)
    pred_res2 = classify_test_loop(model_h2, cuda_id, trainloader)
    pred = pd.concat([pred_res1, pred_res2])
    pred.to_csv(candidate_score_file, index=False)
    return to_save_file_1, model_h1, to_save_file_2, model_h2


def train_matching_model(
        match_meta,
        source_dir,
        version,
        to_save_match_res_file,
        resoluton=(128, 128),
        num_epoch=20,
        cuda_id=0,
        retrain=False,
        extra_meta=[]
):
    qid = list(set([x[0] for x in match_meta]))
    qid.sort()
    np.random.seed(2023)
    np.random.shuffle(qid)
    print(qid[:5])
    train_id = set(qid[: len(qid) // 2])
    train_meta = []
    test_meta = []
    for x in match_meta:
        if x[0] in train_id:
            train_meta.append(x)
        else:
            test_meta.append(x)
    extra_train_meta = []
    extra_test_meta = []
    for x in extra_meta:
        if x[0] in train_id:
            extra_train_meta.append(x)
        else:
            extra_test_meta.append(x)
    class Loss(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.softmax = nn.LogSoftmax(dim=1)
        def forward(self, pred, label):
            label = label.unsqueeze(1)
            label = torch.cat([1 - label, label], dim=1)
            log_pred = self.softmax(pred)
            loss = -(log_pred * label).sum()
            # loss = (loss * loss).sum()
            return loss
    loss_func = Loss()
    train_dataset = MatchingRefineDataset(train_meta + extra_train_meta, resolution=resoluton, random_t=True)
    test_dataset = MatchingRefineDataset(test_meta, resolution=resoluton)
    trainloader = DataLoader(train_dataset, batch_size=56, shuffle=True, prefetch_factor=4, num_workers=12)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=True, prefetch_factor=4, num_workers=12)
    to_save_file_1 = os.path.join(source_dir, f'match_h1_{version}.pt')
    if not os.path.exists(to_save_file_1) or retrain:
        model_h1 = HRnet()
        model_h1 = model_h1.cuda(cuda_id)
        optimizer = optim.Adam(model_h1.parameters(), lr=0.0001, weight_decay=5e-5)
        model_h1 = matching_train_loop(model_h1, cuda_id, optimizer, loss_func, num_epoch, trainloader, testloader,
                                       to_save_file_1)
    else:
        model_h1 = torch.load(to_save_file_1)
    match_res1 = matching_test_loop(model_h1, cuda_id, testloader)
    train_dataset = MatchingRefineDataset(test_meta + extra_test_meta, resolution=resoluton, random_t=True)
    test_dataset = MatchingRefineDataset(train_meta, resolution=resoluton)
    trainloader = DataLoader(train_dataset, batch_size=56, shuffle=True, prefetch_factor=4, num_workers=12)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=True, prefetch_factor=4, num_workers=12)
    to_save_file_2 = os.path.join(source_dir, f'match_h2_{version}.pt')
    if not os.path.exists(to_save_file_2) or retrain:
        model_h2 = HRnet()
        model_h2 = model_h2.cuda(cuda_id)
        optimizer = optim.Adam(model_h2.parameters(), lr=0.0001, weight_decay=5e-5)
        model_h2 = matching_train_loop(model_h2, cuda_id, optimizer, loss_func, num_epoch, trainloader, testloader,
                                       to_save_file_2)
    else:
        model_h2 = torch.load(to_save_file_2)
    match_res2 = matching_test_loop(model_h2, cuda_id, testloader)
    match_res = match_res1 + match_res2
    with open(to_save_match_res_file, 'wb') as f:
        pickle.dump(match_res, f)
    return to_save_file_1, model_h1, to_save_file_2, model_h2






if __name__ == "__main__":
    cuda_id = 0
    version = 'v1.0'
    query_emb_file = '../infer/outputs/vit_v68/train_query.npz'
    extra_query_emb_files = [
        '../infer/outputs/swinv2_v115/train_query.npz',
        '../infer/outputs/swinv2_v107/train_query.npz',
        '../infer/outputs/swinv2_v106/train_query.npz',
    ]
    ref_emb_file = "../infer/outputs/vit_v68/train_refs.npz"
    extra_ref_emb_files = [
        "../infer/outputs/swinv2_v115/train_refs.npz", 
        "../infer/outputs/swinv2_v107/train_refs.npz", 
        "../infer/outputs/swinv2_v106/train_refs.npz", 
    ]
    source_dir = '../infer/outputs/matching'
    if not os.path.exists(source_dir):
        os.mkdir(source_dir)
    candidate_file = "../infer/outputs/matching/candidates.csv"
    candidate_score_file = "../infer/outputs/matching/candidates_score.csv"
    to_save_meta_file =f"../infer/outputs/matching/match_cann_featrure_{version}.meta"
    to_save_match_meta_file = f"../infer/outputs/matching/match_meta_{version}.meta"
    extra_match_meta_file = '../infer/outputs/matching/extra_feature.meta'
    to_save_match_res_file =  f"../infer/outputs/matching/match_res_{version}.pkl"

    gt_file = "../data/meta/train/train_matching_ground_truth.csv"
    video_len_file = "../data/meta/train/query_video_len.csv"
    match_result_file = '../infer/outputs/matchingmatch_res.csv'

    gt_df = pd.read_csv(gt_file)

    video_len_df = pd.read_csv(video_len_file)
    video_len_map = dict(zip(video_len_df.query_id.values, video_len_df.frame_len.values))

    query = load_features(query_emb_file)
    ref = load_features(ref_emb_file)


    query, ref = [transform_features(x, normalize) for x in [query, ref]]

    generate_candidates_classfiy_feature(query, ref, candidate_file, to_save_meta_file, video_len_map, gt_file=gt_file)

    with open(to_save_meta_file, 'rb') as f:
        meta = pickle.load(f)

    extra_meta_files = []
    for i, (train_ref_file, train_query_file) in enumerate(zip(extra_ref_emb_files, extra_query_emb_files)):
        print(train_ref_file, train_query_file)
        extra_ref = load_features(train_ref_file)
        extra_query = load_features(train_query_file)
        extra_query, extra_ref = [transform_features(x, normalize) for x in [extra_query, extra_ref]]
        to_save_extra_meta_file = to_save_meta_file + f'.extra_{i}'
        generate_candidates_classfiy_feature(extra_query, extra_ref, candidate_file, to_save_extra_meta_file, video_len_map, gt_file=gt_file)
        extra_meta_files.append(to_save_extra_meta_file)

    extra_meta = []
    for extra_file in extra_meta_files:
        with open(extra_file, 'rb') as f:
            extra_meta.append(pickle.load(f))
        
    cls_model_file1, cls_model1, cls_model_file2, cls_model2 = train_classify_model(
        meta, "mobilenetv3_small_100", source_dir, version, candidate_score_file, num_epoch=10, cuda_id=cuda_id, extra_meta=extra_meta,
        retrain=True)

    candidate_df = pd.read_csv(candidate_score_file)
    select_cand = candidate_df.groupby(['query_id', 'ref_id']).prob.max()
    select_cand = select_cand.reset_index(None)
    select_cand.sort_values('prob', inplace=True, ascending=False)
    threshold = select_cand.prob.iloc[10000]
    temp_df = pd.merge(select_cand[select_cand.prob > threshold], gt_df)
    temp_df['score'] = temp_df['prob']
    del temp_df['prob']
    temp_df.to_csv('temp.csv', index=False)
    match_metrics = evaluate_matching_track(gt_file, 'temp.csv')
    print(f"Upper Matching track metric: {match_metrics.segment_ap.ap:.4f}")


    generate_matching_feature(
        query, ref, video_len_map, candidate_score_file, to_save_match_meta_file, select_threshold=threshold, gt_file=gt_file)

    extra_feature = []
    for i, (train_ref_file, train_query_file) in enumerate(zip(extra_ref_emb_files, extra_query_emb_files)):
        print(train_ref_file, train_query_file, len(extra_feature))
        try:
            query_temp = load_features(train_query_file)
            ref_temp = load_features(train_ref_file)
            query_temp, ref_temp = [transform_features(x, normalize) for x in [query_temp, ref_temp]]
            extra_feature += generate_matching_feature(query_temp, ref_temp, video_len_map, candidate_score_file, '', select_threshold=threshold, gt_file=gt_file)
        except Exception as e:
            print(e)

    with open(to_save_match_meta_file, 'rb') as f:
        match_meta = pickle.load(f)
    extra_match_meta = extra_feature
    to_save_match_res_file = os.path.join(source_dir, f"match_res_{version}.pkl")

    match_model_file1, match_model1, match_model_file2, match_model2 = train_matching_model(
        match_meta, source_dir, version, to_save_match_res_file, resoluton=(128, 128), num_epoch=20,
        cuda_id=cuda_id, retrain=True, extra_meta=extra_match_meta
    )
    with open(to_save_match_res_file, 'rb') as f:
        res_list = pickle.load(f)

    match_high = generate_matching_result(res_list, threshold=0.35, std_ratio=0.5)
    match_middle = generate_matching_result(res_list, threshold=0.1, std_ratio=1.25)
    match_low = generate_matching_result(res_list, threshold=0.001, std_ratio=2)
    eval_match_res(match_high, gt_file, select_cand)
    eval_match_res(match_middle, gt_file, select_cand)
    eval_match_res(match_low, gt_file, select_cand)
    eval_match_res(match_low  + match_middle+ match_high, gt_file, select_cand, match_result_file)
    eval_match_res(match_low + match_high, gt_file, select_cand, match_result_file)
