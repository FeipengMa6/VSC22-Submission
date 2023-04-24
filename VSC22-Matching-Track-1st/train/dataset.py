import numpy as np
import torch
import cv2


class MatchClassifyDataset:
    def __init__(self, features, infos, resolution=(160, 160), enhance=False):
        self.features = features
        self.infos = infos
        self.zero = np.zeros(resolution, dtype=np.float32)
        self.resolution = resolution
        self.enhance = enhance
    def __len__(self):
        return len(self.features)
    def __getitem__(self, item):
        feature = self.features[item]
        if isinstance(feature, list):
            feature = feature[0]
        label = int(self.infos[item][2])
        h, w = feature.shape
        if h > self.resolution[0]:
            feature = feature[:self.resolution[0]]
            h = self.resolution[0]
        if w > self.resolution[1]:
            feature = feature[:, :self.resolution[1]]
            w = self.resolution[1]
        if self.enhance:
            val = np.random.uniform(0, 1)
            if val > 0.7:
                radian = np.random.choice([3, 5, 7])
                sigma = np.random.uniform(0.1, 0.7)
                feature = cv2.GaussianBlur(feature, (radian, radian), sigma)
            elif val > 0.4:
                random_noise = np.random.uniform(-0.1, 0.1, (h, w))
                feature[:h, :w] += random_noise
            elif val < 0.1:
                feature = abs(feature)
                feature = np.sqrt(feature)
        zero = self.zero.copy()
        zero[:h, :w] += feature
        return np.stack([zero]*3), torch.LongTensor([label]), self.infos[item][0], self.infos[item][1]


class MatchingRefineDataset:
    def __init__(self, meta, resolution=(160, 160), random_t=False):
        self.meta = meta
        self.resolution = resolution
        self.zero = np.zeros(resolution, dtype=np.float32)
        self.random_t = random_t
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, item):
        qid, rid, qfeat, rfeat, gt_list = self.meta[item]
        label = self.zero.copy()
        # label = np.zeros((60, 60))
        for qs, qe, rs, re in gt_list:
            v_vec = np.array([rs - re, qe - qs], dtype=np.float32)
            # v_vec = np.array([qe - re, rs - qs])
            v_vec /= np.linalg.norm(v_vec)
            for i in range(round(qs), int(qe) + 1):
                for j in range(round(rs), int(re) + 1):
                    dist = (i - qs) * v_vec[0] + (j-rs) * v_vec[1]
                    if abs(dist) < 1:
                        label[i, j] = np.sqrt(1 - abs(dist))
            # if label.max() > 0:
            #     label /= label.max()
        sim_mat = np.matmul(qfeat, rfeat.T)
        feat = self.zero.copy()
        h, w = sim_mat.shape
        if h > self.resolution[0]:
            h = self.resolution[0]
        if w > self.resolution[1]:
            w = self.resolution[1]
        feat[:h, :w] += sim_mat[:h, :w]
        if self.random_t:
            if np.random.uniform(0, 1) > 0.5:
                feat = feat.T
                label = label.T
                h, w = w, h
            val = np.random.uniform(0, 1)
            if val > 0.7:
                radian = np.random.choice([3, 5, 7])
                sigma = np.random.uniform(0.1, 0.7)
                feat = cv2.GaussianBlur(feat, (radian, radian), sigma)
            elif val > 0.4:
                random_noise = np.random.uniform(-0.125, 0.125, (h, w))
                feat[:h, :w] += random_noise
            elif val < 0.1:
                feat = abs(feat)
                feat = np.sqrt(feat)
        return np.stack([feat, feat, feat]), label, qid, rid, h, w


