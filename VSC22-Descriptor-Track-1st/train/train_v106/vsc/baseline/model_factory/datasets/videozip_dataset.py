import io
import os
import random
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from .transforms_utils import build_transforms
import itertools


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


from ..utils import DATASETS


@DATASETS.register_module()
class VideoZipDataSet(torch.utils.data.Dataset):

    def __init__(
            self, vids_path, zip_prefix, width, max_frames, preprocess, original_fps=1
    ):
        self.zip_prefix = zip_prefix
        self.width = width
        self.max_frames = max_frames
        self.original_fps = original_fps
        self.transform = build_transforms(preprocess, width, width)

        with open(vids_path, "r", encoding="utf-8") as f:
            self.vids = [x.strip() for x in f]
        print(f"### Num vids {len(self.vids)}")
        print(f"### Samples {' '.join(self.vids[:5])}")

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]
        zip_path = self.zip_prefix % (vid[-2:], vid)
        frames_tensor = torch.zeros(self.max_frames, 3, self.width, self.width)
        frames_mask = torch.zeros(self.max_frames).long()
        timestamps = torch.zeros(self.max_frames, 2).long()

        try:
            with ZipFile(zip_path, 'r') as handler:
                img_name_list = handler.namelist()
                img_name_list = sorted(img_name_list)

                for i, img_name in enumerate(img_name_list):
                    i_img_content = handler.read(img_name)
                    i_img = Image.open(io.BytesIO(i_img_content))
                    i_img_tensor = self.transform(i_img)
                    frames_tensor[i, ...] = i_img_tensor
                    frames_mask[i] = 1
                    timestamps[i] = torch.tensor([i / self.original_fps, (i + 1) / self.original_fps])
        except FileNotFoundError as e:
            print(e)

        record = {
            "name": vid,
            "timestamp": np.array(timestamps),
            "input": frames_tensor,
            "input_mask": frames_mask,
        }

        return record


@DATASETS.register_module()
class ImagePatternDataSet(torch.utils.data.Dataset):

    def __init__(
            self, vids_path, img_prefix, width, preprocess
    ):
        self.img_prefix = img_prefix
        self.width = width
        self.transform = build_transforms(preprocess, width, width)

        self.vids = []
        with open(vids_path, "r", encoding="utf-8") as f:
            for line in f:
                vid, num = line.strip().split("\t")
                num = int(num)
                for n in range(num):
                    dir_name = self.img_prefix % (vid, n)
                    img_name_list = [os.path.join(dir_name, x) for x in sorted(os.listdir(dir_name))]
                    meta = [(vid, k, n, x) for k, x in enumerate(img_name_list)]
                    self.vids.extend(meta)
        print("### Tot vids %d" % len(self.vids))
        print("### Examples:")
        print(f"### {self.vids[:2]}")

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid, frame_index, img_idx, path = self.vids[index]

        i_img = Image.open(path)
        i_img_tensor = self.transform(i_img)

        record = {
            "name": vid, "frame_index": frame_index, "img_index": img_idx, "input": i_img_tensor
        }

        return record


@DATASETS.register_module()
class PairWiseFeatZipDataSet(torch.utils.data.Dataset):

    def __init__(
            self, query_vids_path, ref_vids_path, ann_path, feat_zip_path, max_frames=256,
            positive_ratio=0.2, num_workers=8
    ):
        self.max_frames = max_frames

        with open(query_vids_path, "r", encoding="utf-8") as f:
            self.q_vids = [x.strip() for x in f]
        print("#### query vid num %d" % len(self.q_vids))
        print("#### query vid samples %s" % " ".join(self.q_vids[:5]))

        with open(ref_vids_path, "r", encoding="utf-8") as f:
            self.r_vids = [x.strip() for x in f]
        print("#### ref vid num %d" % len(self.r_vids))
        print("#### ref vid samples %s" % " ".join(self.r_vids[:5]))

        self.ann = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                q_vid, r_vid = line.strip().split(",")
                self.ann.append((q_vid, r_vid))
        print("#### ann num %d" % len(self.ann))
        self.ann_set = set(self.ann)
        self.pairs = list(itertools.product(self.q_vids, self.r_vids))

        self.handles = []
        for i in range(num_workers):
            self.handles.append(ZipFile(feat_zip_path, 'r'))

        self.positive_ratio = positive_ratio

    def __len__(self):
        return len(self.pairs)
        # return len(self.q_vids)

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()

        info = self.sample(index, worker_info.id)

        return info

    def _read_feats(self, vid, worker_id):

        video_feature = np.load(io.BytesIO(self.handles[worker_id].read(vid)), allow_pickle=True).astype(np.float32)
        video_feature = video_feature[:self.max_frames]
        video_feature = torch.tensor(video_feature)

        return video_feature

    def sample(self, index, worker_id):
        labels = 0
        q_vid, r_vid = self.pairs[index]
        if (q_vid, r_vid) in self.ann_set:
            labels = 1
        prob = np.random.random()
        if prob < self.positive_ratio:
            q_vid, r_vid = random.choice(self.ann)
            labels = 1

        # q_vid = self.q_vids[index]
        # if q_vid in self.ann:
        #     prob = np.random.random()
        #     if prob < self.positive_ratio:
        #         sampled_vid = random.choice(self.ann[q_vid])
        #         labels = 1
        #     else:
        #         sampled_vid = random.choice(self.r_vids)
        #         labels = -1
        # else:
        #     sampled_vid = random.choice(self.r_vids)
        #     labels = -1

        feat_a = self._read_feats(q_vid, worker_id)
        feat_b = self._read_feats(r_vid, worker_id)

        res = {
            "frames_a": feat_a, "frames_b": feat_b, "vid_a": q_vid, "vid_b": r_vid, "labels": labels
        }

        return res


@DATASETS.register_module()
class FeatZipDataSet(torch.utils.data.Dataset):

    def __init__(
            self, vids_path, feat_zip_path, max_frames=256, num_workers=8
    ):
        self.max_frames = max_frames

        with open(vids_path, "r", encoding="utf-8") as f:
            self.vids = [x.strip() for x in f]
        print("#### query vid num %d" % len(self.vids))
        print("#### query vid samples %s" % " ".join(self.vids[:5]))

        self.handles = []
        for i in range(num_workers):
            self.handles.append(ZipFile(feat_zip_path, 'r'))

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        worker_info = torch.utils.data.get_worker_info()

        info = self.sample(index, worker_info.id)

        return info

    def _read_feats(self, vid, worker_id):

        video_feature = np.load(io.BytesIO(self.handles[worker_id].read(vid)), allow_pickle=True).astype(np.float32)
        video_feature = video_feature[:self.max_frames]
        video_feature = torch.tensor(video_feature)

        return video_feature

    def sample(self, index, worker_id):
        vid = self.vids[index]
        video_feature = self._read_feats(vid, worker_id)

        res = {
            "frames": video_feature, "vid": vid
        }

        return res


@DATASETS.register_module()
class LabelFeatZipDataSet(FeatZipDataSet):

    def __init__(
            self, vids_path, feat_zip_path, max_frames=256, num_workers=8, ann_vids_path=""
    ):
        super(LabelFeatZipDataSet, self).__init__(vids_path, feat_zip_path, max_frames, num_workers)

        self.ann_set = set()
        with open(ann_vids_path, "r", encoding="utf-8") as f:
            for line in f:
                self.ann_set.add(line.strip())

    def sample(self, index, worker_id):
        vid = self.vids[index]
        video_feature = self._read_feats(vid, worker_id)
        if vid in self.ann_set:
            labels = 1
        else:
            labels = 0

        res = {
            "frames": video_feature, "vid": vid, "labels": labels
        }

        return res


@DATASETS.register_module()
class MatchFeatZipDataSet(FeatZipDataSet):

    def __init__(
            self, vids_path, feat_zip_path, max_frames=256, num_workers=8, ref_vids_path="", ann_path="",
            positive_ratio=0.2
    ):
        super(MatchFeatZipDataSet, self).__init__(vids_path, feat_zip_path, max_frames, num_workers)
        # self.q_vids = self.vids
        with open(ref_vids_path, "r", encoding="utf-8") as f:
            self.r_vids = [x.strip() for x in f]

        print("#### ref vid num %d" % len(self.r_vids))
        print("#### ref vid samples %s" % " ".join(self.r_vids[:5]))

        ann_df = pd.read_csv(ann_path)
        self.ann_pair = dict()
        self.q2pair = dict()

        for i in range(ann_df.shape[0]):
            pair = (ann_df.loc[i, "query_id"], ann_df.loc[i, "ref_id"])
            self.ann_pair.setdefault(pair, list()).append([
                ann_df.loc[i, "query_start"], ann_df.loc[i, "ref_start"],
                ann_df.loc[i, "query_end"], ann_df.loc[i, "ref_end"]
            ])
            self.q2pair.setdefault(ann_df.loc[i, "query_id"], list()).append(pair)
        self.q_vids = list(self.q2pair.keys())

        self.positive_ratio = positive_ratio

    def __len__(self):
        # return len(self.q_vids)
        return len(self.q2pair)

    def sample(self, index, worker_id):
        q_vid = self.q_vids[index]
        if q_vid in self.q2pair:
            if np.random.random() < self.positive_ratio:
                pair = random.choice(self.q2pair[q_vid])
                r_vid = pair[1]
                gt = self.ann_pair[pair]
            else:
                r_vid = random.choice(self.r_vids)
                if (q_vid, r_vid) in self.q2pair:
                    gt = self.ann_pair[(q_vid, r_vid)]
                else:
                    gt = []
        else:
            r_vid = random.choice(self.r_vids)
            gt = []

        q_feat = self._read_feats(q_vid, worker_id)
        r_feat = self._read_feats(r_vid, worker_id)

        if len(gt) > 0:
            s_label = 1.
        else:
            s_label = 0.

        m_label = self._format_label(gt)

        res = {
            "feats_q": q_feat, "feats_r": r_feat, "s_labels": s_label, "m_labels": m_label
        }

        return res

    def _format_label(self, annotations):

        classes = np.asarray([1] * len(annotations), dtype=np.int64)
        area = np.asarray([(x[2] - x[0]) * (x[3] - x[1]) for x in annotations], dtype=np.float32)

        boxes = np.asarray(annotations).reshape((-1, 4))

        # norm
        bboxes_corners = self._corners_to_center_format_numpy(boxes) / self.max_frames

        target = dict(
            class_labels=torch.from_numpy(classes).long(), area=torch.from_numpy(area).float(),
            boxes=torch.from_numpy(bboxes_corners).float()
        )

        return target

    @staticmethod
    def _center_to_corners_format_numpy(bboxes_center: np.ndarray) -> np.ndarray:
        center_x, center_y, width, height = bboxes_center.T
        bboxes_corners = np.stack(
            # top left x, top left y, bottom right x, bottom right y
            [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
            axis=-1,
        )
        return bboxes_corners

    @staticmethod
    def _corners_to_center_format_numpy(bboxes_corners: np.ndarray) -> np.ndarray:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = bboxes_corners.T
        bboxes_center = np.stack(
            [
                (top_left_x + bottom_right_x) / 2,  # center x
                (top_left_y + bottom_right_y) / 2,  # center y
                (bottom_right_x - top_left_x),  # width
                (bottom_right_y - top_left_y),  # height
            ],
            axis=-1,
        )
        return bboxes_center
