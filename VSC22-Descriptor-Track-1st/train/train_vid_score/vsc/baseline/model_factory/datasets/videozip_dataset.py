import io
import random
from zipfile import ZipFile

import numpy as np
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
