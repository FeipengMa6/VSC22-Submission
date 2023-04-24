import io

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import lmdb
from .transforms_utils import build_transforms
from vsc.baseline.model_factory.transforms import OverlayEmoji, OverlayText, CropAndPad, \
    RandomOverlayImages, RandomStackImages, RandomOverlayCorners, RandomCompose

import albumentations as A

from ..utils import DATASETS


@DATASETS.register_module()
class VideoLmdbDataSet(torch.utils.data.Dataset):

    def __init__(
            self, vids_path, meta_path, preprocess, lmdb_path, lmdb_size=1e12, width=224
    ):
        self.lmdb_path = lmdb_path
        self.lmdb_size = lmdb_size
        self.transform = build_transforms(preprocess, width, width)
        self.width = width

        if len(vids_path) > 0:
            with open(vids_path, "r", encoding="utf-8") as f:
                self.vids = [x.strip() for x in f]
        else:
            self.vids = []
        print(f"### Num vids {len(self.vids)}")
        print(f"### Samples {' '.join(self.vids[:5])}")
        self.vid2interval, self.id2vid, self.id_list = self._load_meta(meta_path)

        self.hard_pipelines = np.array([
            A.Compose([
                A.OneOf([
                    A.HorizontalFlip(p=1),
                    A.VerticalFlip(p=1),
                    A.RandomRotate90(p=1)
                ], p=0.2),
                A.RandomResizedCrop(self.width, self.width, scale=(0.5, 1), p=1),
                A.GaussNoise(p=0.1),
                A.GaussianBlur(p=0.5),
                A.RandomScale(p=0.1),
                A.Perspective(p=0.1),
                A.ImageCompression(20, 100, p=0.1),
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
                        A.HueSaturationValue(p=1)
                    ], p=0.8),
                    RandomStackImages(self.lmdb_path, self.lmdb_size, self.width, p=0.1),
                    RandomOverlayImages(self.lmdb_path, self.lmdb_size, self.width, p=0.1),
                    RandomOverlayCorners(p=0.1),
                    A.Rotate(45, border_mode=0, p=0.1)
                ], shuffle=True),
            ]),
            A.Compose([
                A.RandomResizedCrop(self.width, self.width, scale=(0.5, 1), p=1),
                A.OneOf([
                    A.GaussNoise(p=1),
                    A.GaussianBlur(p=1),
                    A.RandomScale(p=1),
                    A.Perspective(p=1),
                    A.ImageCompression(20, 100, p=1),
                    OverlayText(p=1),
                    OverlayEmoji(p=1),
                    CropAndPad(p=1),
                    A.CropAndPad(percent=(-0.4, 0.4), p=1),
                    A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1),
                    A.RandomBrightness((-0.2, 0.1), p=1),
                    A.ToGray(p=1),
                    A.HueSaturationValue(p=1),
                    RandomStackImages(self.lmdb_path, self.lmdb_size, self.width, p=1),
                    RandomOverlayImages(self.lmdb_path, self.lmdb_size, self.width, p=1),
                    RandomOverlayCorners(p=1),
                    A.Rotate(45, border_mode=0, p=1)
                ], p=1)
            ]),
            # A.Compose([
            #     A.RandomResizedCrop(self.width, self.width, scale=(0.5, 1), p=1),
            #     RandomCompose([
            #         RandomStackImages(self.lmdb_path, self.lmdb_size, self.width, p=1),
            #         A.OneOf([
            #             CropAndPad(p=1),
            #             A.CropAndPad(percent=(-0.4, 0.4), p=1)
            #         ], p=0.5),
            #     ])
            # ])
        ], dtype="object")
        self.hard_pipeline_probabilities = [0.9, 0.1]

        self.easy_pipeline = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.RandomResizedCrop(self.width, self.width, scale=(0.5, 1), p=1),
            A.OneOf([
                A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1),
                A.RandomBrightness((-0.2, 0.1), p=1),
                A.ToGray(p=1),
                A.HueSaturationValue(p=1)
            ], p=0.5),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(p=0.5),
            A.RandomScale(p=0.1),
            A.Perspective(p=0.1),
            A.OneOf([
                CropAndPad(p=1),
                A.CropAndPad(percent=(-0.4, 0.4), p=1)
            ], p=0.1),
        ])  # simple argument

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
                vid2interval[vid] = interval
                vid_ += 1

        return vid2interval, id2vid, id_list

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




