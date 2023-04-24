import torch
from PIL import Image
import numpy as np
from vsc.baseline.video_reader.ffmpeg_video_reader import FFMpegVideoReader
from vsc.baseline.inference import  VideoReaderType
import glob
import os
from functools import lru_cache


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

class VideoDataset(torch.utils.data.Dataset):
    """Decodes video frames at a fixed FPS via ffmpeg."""

    def __init__(
        self,
        path: str,
        fps: float,
        vids: list,
        preprocess=None,
        transform1={},
        transform2={},
        extensions=("mp4",),
        video_reader=VideoReaderType.FFMPEG,
        ffmpeg_path="ffmpeg",
        
    ):
        self.path = path
        self.fps = fps
        self.video_reader = video_reader
        self.ffmpeg_path = ffmpeg_path
        self.preprocess = preprocess
        self.transfrom1 = transform1
        self.transfrom2 = transform2
        if len(extensions) == 1:
            filenames = glob.glob(os.path.join(path, f"*.{extensions[0]}"))
        else:
            filenames = glob.glob(os.path.join(path, "*.*"))
            filenames = (fn for fn in filenames if fn.rsplit(".", 1)[-1] in extensions)
        self.videos = sorted(filenames)
        if not self.videos:
            raise Exception("No videos found!")
        self.vids = vids
        self.vid_video_map = {
            os.path.basename(x).split('.')[0]: x for x in self.videos
        }
        inter_vids = set(self.vids).intersection(set(self.vid_video_map))
        diff_vids = set(self.vids).difference(set(self.vid_video_map))
        # if diff_vids:
        #     print(f'failed to find files for {diff_vids}')
        self.vids = [x for x in self.vids if x in inter_vids]
    def __len__(self):
        return len(self.vids)
    
    def __getitem__(self, idx):
        vid = self.vids[idx]
        video_file = self.vid_video_map[vid]
        timestamps,frames = self.read_frames(video_file) 
        # for i in range(len(frames)):
        #     timestamps[i] = torch.tensor([i / self.fps, (i + 1) / self.fps])
        record = {
            "name": vid,
            "timestamp": timestamps,
            "frames": frames,
        }
        
        for key, transform in self.transfrom1.items():
            temp_feature_list = []
            for frame in frames:
                temp_feature_list.append(transform(frame))
            record[key] = torch.stack(temp_feature_list)
        
        status, frames_preprocess = self.preprocess(frames)
        if status:
            frames_preprocess += frames
        for key, transform in self.transfrom2.items():
            temp_feature_list = []
            for frame in frames_preprocess:
                temp_feature_list.append(transform(frame))
            record[key] = torch.stack(temp_feature_list)
        return record
        # return vid,timestamps,frames
    def read_frames(self, video):
        if self.video_reader == VideoReaderType.FFMPEG:
            reader = FFMpegVideoReader(
                video_path=video, required_fps=self.fps, ffmpeg_path=self.ffmpeg_path
            )
        else:
            raise ValueError(f"VideoReaderType: {self.video_reader} not supported")
        timestamps = []
        frames = []
        for start_timestamp, end_timestamp, frame in reader.frames():
            timestamps.append([start_timestamp, end_timestamp])
            frames.append(frame)
        return timestamps,frames


class MatchClassifyDataset:
    def __init__(self, features, infos, resolution=(160, 160)):
        self.features = features
        self.infos = infos
        self.zero = np.zeros(resolution, dtype=np.float32)
        self.resolution = resolution
    def __len__(self):
        return len(self.features)
    def __getitem__(self, item):
        feature = self.features[item]
        if isinstance(feature, list):
            feature = feature[0]
        h, w = feature.shape
        if h > self.resolution[0]:
            feature = feature[:self.resolution[0]]
            h = self.resolution[0]
        if w > self.resolution[1]:
            feature = feature[:, :self.resolution[1]]
            w = self.resolution[1]
        zero = self.zero.copy()
        zero[:h, :w] += feature
        return np.stack([zero]*3), self.infos[item][0], self.infos[item][1]


class MatchRefineDataset:
    def __init__(self, meta, resolution=(160, 160)):
        self.meta = meta
        self.resolution = resolution
        self.zero = np.zeros(resolution, dtype=np.float32)
    def __len__(self):
        return len(self.meta)
    def __getitem__(self, item):
        qid, rid, qfeat, rfeat = self.meta[item]
        sim_mat = np.matmul(qfeat, rfeat.T)
        feat = self.zero.copy()
        h, w = sim_mat.shape
        if h > self.resolution[0]:
            h = self.resolution[0]
        if w > self.resolution[1]:
            w = self.resolution[1]
        feat[:h, :w] += sim_mat[:h, :w]
        return np.stack([feat, feat, feat]), qid, rid, h, w
