import io
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import lmdb
from .transforms_utils import build_transforms
from vsc.baseline.model_factory.transforms import OverlayEmoji, OverlayText, CropAndPad, \
    RandomOverlayImages, RandomStackImages, RandomOverlayCorners, RandomCompose
from augly.image.transforms import OverlayImage

import albumentations as A

from ..utils import DATASETS


@DATASETS.register_module()
class VideoLmdbDataSet(torch.utils.data.Dataset):

    def __init__(
            self, vids_path, meta_path, preprocess, lmdb_path, lmdb_size=1e12, width=224,
            arg_lmdb_path="",
            probs=(0.8, 0.2), crop=0.5, mixup=0.1,
    ):
        self.lmdb_path = lmdb_path
        self.arg_lmdb_path = arg_lmdb_path
        self.lmdb_size = lmdb_size
        self.transform = build_transforms(preprocess, width, width)
        self.width = width

        if len(vids_path) > 0:
            if isinstance(vids_path, str):
                vids_path = [vids_path]
            self.vids = []
            for path in vids_path:
                with open(path, "r", encoding="utf-8") as f:
                    self.vids.extend([x.strip() for x in f])
            self.vids_set = set(self.vids)
        else:
            self.vids = []
        print(f"### Num vids {len(self.vids)}")
        print(f"### Samples {' '.join(self.vids[:5])}")
        self.vid2interval, self.id2vid, self.id_list, self.id2vidstr = self._load_meta(meta_path)

        self.hard_pipelines = np.array([
            A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.RandomRotate90(p=1)
                ], p=0.2),
                A.RandomResizedCrop(self.width, self.width, scale=(crop, 1), p=1),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(p=0.5),
                A.RandomScale(p=0.1),
                A.Perspective(p=0.1),
                A.ImageCompression(20, 100, p=0.1),
                A.RandomSnow(p=0.1),
                OverlayText(p=0.1),
                OverlayEmoji(p=0.1),
                RandomCompose([
                    A.OneOf([
                        CropAndPad(p=1),
                        A.CropAndPad(percent=(-0.4, 0.4), p=1)
                    ], p=0.1),
                    A.OneOf([
                        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1),
                        A.RandomBrightness((-0.2, 0.1), p=1),
                        A.ToGray(p=1),
                        A.HueSaturationValue(p=1),
                        A.RandomBrightnessContrast(p=1)
                    ], p=0.8),
                    # RandomStackImages(self.arg_lmdb_path, self.lmdb_size, self.width, p=0.1),
                    RandomOverlayImages(self.arg_lmdb_path, self.lmdb_size, self.width, p=mixup),
                    RandomOverlayCorners(p=0.1),
                    A.Rotate(45, border_mode=0, p=0.1)
                ], shuffle=True, p=1),
            ]),
            A.Compose([
                A.RandomResizedCrop(self.width, self.width, scale=(crop, 1), p=1),
                RandomOverlayImages(self.arg_lmdb_path, self.lmdb_size, self.width, p=mixup),
                RandomOverlayCorners(p=0.1),
                OverlayText(p=0.1),
                OverlayEmoji(p=0.1),
                RandomCompose([
                    A.OneOf([
                        CropAndPad(p=1),
                        A.CropAndPad(percent=(-0.4, 0.4), p=1)
                    ], p=0.2),
                    A.OneOf([
                        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1),
                        A.RandomBrightness((-0.2, 0.1), p=1),
                        A.ToGray(p=1),
                        A.HueSaturationValue(p=1),
                        A.RandomBrightnessContrast(p=1)
                    ], p=0.8),
                    # RandomStackImages(self.arg_lmdb_path, self.lmdb_size, self.width, p=0.5),
                    A.Rotate(45, border_mode=0, p=0.1)
                ], shuffle=True, p=1)
            ])
        ], dtype="object")
        self.hard_pipeline_probabilities = probs

        self.easy_pipeline = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.RandomResizedCrop(self.width, self.width, scale=(crop, 1), p=1),
            A.OneOf([
                A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1),
                A.RandomBrightness((-0.2, 0.1), p=1),
                A.ToGray(p=1),
                A.HueSaturationValue(p=1)
            ], p=0.5),
            A.Rotate(45, border_mode=0, p=0.1),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(p=0.5),
            A.RandomScale(p=0.1),
            A.Perspective(p=0.1),
            A.OneOf([
                CropAndPad(p=1),
                A.CropAndPad(percent=(-0.4, 0.4), p=1)
            ], p=0.2),
        ])  # simple argument

        self.native_pipeline = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.RandomResizedCrop(self.width, self.width, scale=(crop, 1), p=1),
            A.GaussNoise(p=0.1),
            A.ImageCompression(50, 100, p=0.1),
        ])

    def _open_lmdb(self):
        self.lmdb_env = lmdb.open(
            self.lmdb_path,
            map_size=int(self.lmdb_size),
            readonly=True,
            readahead=False,
            max_readers=8192,
            max_spare_txns=8192,
            lock=False
        )

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        res = self._sample_pair(index)

        return res

    def _load_meta(self, path):
        array = np.load(path)

        id2vid = dict()
        vid2interval = dict()
        id2vidstr = dict()
        id_list = []
        if len(self.vids) != 0:
            target_vid_set = set(self.vids)
        else:
            target_vid_set = set(array["vids"])

        vid_ = 0
        for i, (vid, inter) in enumerate(zip(array["vids"], array["intervals"])):
            if vid in target_vid_set:
                interval = list(range(inter[0], inter[1], 1))
                id_list.extend(interval)

                for x in interval:
                    id2vid[x] = vid_
                    id2vidstr[x] = vid
                vid2interval[vid] = interval
                vid_ += 1

        return vid2interval, id2vid, id_list, id2vidstr

    def _sample_pair(self, idx):
        if not hasattr(self, 'lmdb_env') or self.lmdb_env is None:
            self._open_lmdb()

        id_a = self.id_list[idx]
        vid_a = self.id2vid[id_a]

        """
        正样本采样策略
        1. 增强本身产生正样本（baseline）
        2. 采样同一video内的帧为正样本
        3. 来源于标注数据随机采样正样本
        """
        id_b = id_a
        vid_b = vid_a

        with self.lmdb_env.begin() as doc:
            img_a = Image.open(io.BytesIO(doc.get(str(id_a).encode())))
            if id_b != id_a:
                img_b = Image.open(io.BytesIO(doc.get(str(id_b).encode())))
            else:
                img_b = img_a

        img_a = self.transform_q(img_a)
        img_b = self.transform_k(img_b)

        res = dict(
            id_a=id_a, vid_a=vid_a, img_a=img_a,
            id_b=id_b, vid_b=vid_b, img_b=img_b
        )
        return res

    def transform_q(self, img):
        sample_pipeline = np.random.choice(self.hard_pipelines, p=self.hard_pipeline_probabilities)
        img = Image.fromarray(sample_pipeline(image=np.array(img))["image"])

        res = self.transform(img)

        return res

    def transform_k(self, img):
        img = Image.fromarray(self.easy_pipeline(image=np.array(img))["image"])

        res = self.transform(img)

        return res

    def transform_n(self, img):
        img = Image.fromarray(self.native_pipeline(image=np.array(img))["image"])

        res = self.transform(img)

        return res


@DATASETS.register_module()
class LabelVideoLmdbDataSet(VideoLmdbDataSet):

    def __init__(
            self, vids_path, meta_path, preprocess, lmdb_path, lmdb_size=1e12, width=224,
            arg_lmdb_path="",
            ann_path="", probs=(0.8, 0.2), crop=0.5, mixup=0.1
    ):
        super().__init__(vids_path, meta_path, preprocess, lmdb_path, lmdb_size, width, arg_lmdb_path, probs, crop, mixup)
        ann_df = pd.read_csv(ann_path)
        self.ann = dict()
        for i, (q_id, r_id) in enumerate(zip(ann_df["query_id"], ann_df["ref_id"])):
            # Example
            # query_id	ref_id	query_start	query_end	ref_start	ref_end
            # Q100009	R122648	14.194032	20.314716	1.098193	8.760109
            if q_id in self.vids_set:
                self.ann.setdefault(q_id, list()).append(ann_df.loc[i, :].values.tolist())
                self.ann.setdefault(r_id, list()).append(ann_df.loc[i, :].values.tolist())
        print(f"#### NUM ANN DICT {len(self.ann)}.")

    def _sample_pair(self, idx):
        if not hasattr(self, 'lmdb_env') or self.lmdb_env is None:
            self._open_lmdb()

        id_a = self.id_list[idx]
        vid_str_a = self.id2vidstr[id_a]

        if vid_str_a in self.ann:  # 监督的pair
            id_a, id_b = self.sample_ann_imgs(*random.choice(self.ann[vid_str_a]))
        else:
            id_b = id_a

        vid_a = self.id2vid[id_a]
        vid_b = self.id2vid[id_b]

        # vid_str_b = self.id2vidstr[id_b]

        with self.lmdb_env.begin() as doc:
            img_a = Image.open(io.BytesIO(doc.get(str(id_a).encode())))
            if id_b != id_a:
                img_b = Image.open(io.BytesIO(doc.get(str(id_b).encode())))
            else:
                img_b = img_a

        if vid_str_a in self.ann:  # 监督的pair
            img_a = self.transform_n(img_a)
            img_b = self.transform_n(img_b)
            weight = 1.
        elif vid_str_a.startswith("Q"):
            img_a = self.transform_n(img_a)
            img_b = self.transform_n(img_b)
            weight = 1.
        else:
            if np.random.random() < 0.5:
                img_a = self.transform_k(img_a)
                img_b = self.transform_k(img_b)
            else:
                img_a = self.transform_q(img_a)
                img_b = self.transform_k(img_b)
            weight = 0.

        res = dict(
            id_a=id_a, vid_a=vid_a, img_a=img_a,
            id_b=id_b, vid_b=vid_b, img_b=img_b,
            weight=weight
        )
        return res

    def sample_ann_imgs(self, q_vid, r_vid, q_start, q_end, r_start, r_end):

        random_num = np.random.randint(10)

        q_range = self.vid2interval[q_vid]
        r_range = self.vid2interval[r_vid]

        q_intervals = np.linspace(q_start, q_end, 11).round().astype(np.int32)  #
        r_intervals = np.linspace(r_start, r_end, 11).round().astype(np.int32)

        q_interval = [q_intervals[random_num], q_intervals[random_num + 1]]
        r_interval = [r_intervals[random_num], r_intervals[random_num + 1]]

        if q_interval[0] < q_interval[1]:
            q_index = np.random.randint(q_interval[0], q_interval[1] + 1)
        else:
            q_index = q_interval[0]

        if r_interval[0] < r_interval[1]:
            r_index = np.random.randint(r_interval[0], r_interval[1] + 1)
        else:
            r_index = r_interval[0]

        id_q = q_range[min(q_index, len(q_range) - 1)]
        id_r = r_range[min(r_index, len(r_range) - 1)]

        return id_q, id_r


@DATASETS.register_module()
class ImageLmdbDataSet(VideoLmdbDataSet):

    def _sample_pair(self, idx):
        if not hasattr(self, 'lmdb_env') or self.lmdb_env is None:
            self._open_lmdb()

        id_a = self.id_list[idx]

        """
        正样本采样策略
        1. 增强本身产生正样本（baseline）
        2. 采样同一video内的帧为正样本
        3. 来源于标注数据随机采样正样本
        """
        id_b = self.id_list[np.random.randint(0, len(self))]

        with self.lmdb_env.begin() as doc:
            img_a = Image.open(io.BytesIO(doc.get(str(id_a).encode())))
            img_b = Image.open(io.BytesIO(doc.get(str(id_b).encode())))

        img_a = Image.fromarray(self.easy_pipeline(image=np.array(img_a))["image"])
        img_b = Image.fromarray(self.easy_pipeline(image=np.array(img_b))["image"])

        # if np.random.random() < 0.5:
        #     img_b, img_a = img_a, img_b
        #
        # opacity = np.random.uniform(0.2, 0.7)
        # overlay_size = np.random.uniform(0.5, 1.)
        # img = OverlayImage(img_a, opacity=opacity, overlay_size=overlay_size,
        #                    x_pos=random.uniform(0.0, 1.0 - overlay_size),
        #                    y_pos=random.uniform(0.0, 1.0 - overlay_size),
        #                    p=1.0)(img_b)

        if np.random.random() < 0.3:
            opacity = np.random.uniform(0.3, 0.7)
            overlay_size = np.random.uniform(0.3, 0.7)
            img = OverlayImage(img_a, opacity=opacity, overlay_size=overlay_size,
                               x_pos=random.uniform(0.0, 1.0 - overlay_size),
                               y_pos=random.uniform(0.0, 1.0 - overlay_size),
                               p=1.0)(img_b)
        else:
            opacity = np.random.uniform(0.3, 0.7)
            img = OverlayImage(img_a.resize((img_b.width, img_b.height)), opacity=opacity, overlay_size=1,
                               x_pos=0., y_pos=0, p=1.0)(img_b)

        img = self.transform(img)
        img_a = self.transform(img_a)
        img_b = self.transform(img_b)

        if np.random.random() < 0.1:
            labels = 1.
        else:
            labels = 0.
            img = img_a

        res = dict(
            img=img, img_a=img_a, img_b=img_b, id_a=id_a, id_b=id_b, labels=labels
        )
        return res


