import torch
from PIL import Image
from vsc.baseline.video_reader.ffmpeg_video_reader import FFMpegVideoReader
from vsc.baseline.inference import  VideoReaderType
import glob
import os
from torch.utils.data import Dataset
import torch
from zipfile import ZipFile
from PIL import Image
import io
import os
from torch.nn.utils.rnn import pad_sequence

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
        #diff_vids = set(self.vids).difference(set(self.vid_video_map))
        #if diff_vids: no log output
            #print(f'failed to find files for {diff_vids}')
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
    
class D_vsc(Dataset):
    def __init__(self,video_ids,zip_prefix,img_size=320,transform=None,max_video_frames=None):
        super(D_vsc,self).__init__()
        self.video_ids = video_ids
        self.zip_prefix = zip_prefix
        self.transform = transform
        self.img_size = img_size
        self.max_video_frames = max_video_frames

        # filter path
        print("len of data before filter: ",len(self.video_ids))
        tmp = []
        for video_id in self.video_ids:
            zip_path = "%s/%s/%s.zip" % (self.zip_prefix, video_id[-2:], video_id)
            if(os.path.exists(zip_path)):
                tmp.append(video_id)
        self.video_ids = tmp
        print("len of data after filter: ",len(self.video_ids))

    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, index):
        video_id = self.video_ids[index]
        zip_path = "%s/%s/%s.zip" % (self.zip_prefix, video_id[-2:], video_id)
        with ZipFile(zip_path, 'r') as handler:
            img_name_list = handler.namelist() # 压缩图像名称
            img_name_list = sorted(img_name_list) # 按照帧顺序排序
            # 预先定义好tensor，加速读写
            # img_tensor = torch.zeros(self.max_video_frames, 3, self.img_size, self.img_size)
            img_tensor = []
            # video_mask = torch.zeros(self.max_video_frames).long()

            for i, img_name in enumerate(img_name_list):
                i_img_content = handler.read(img_name)
                i_img = Image.open(io.BytesIO(i_img_content))
                i_img_tensor = self.transform(i_img)
                img_tensor.append(i_img_tensor)

                # video_mask[i] = 1
            img_tensor = torch.stack(img_tensor,dim=0)

        # return img_tensor, video_mask, video_id
        return img_tensor, video_id
    @staticmethod
    def collate_fn(batch):
        img_tensor, video_id = zip(*batch)
        img_tensor = pad_sequence(img_tensor,batch_first=True,padding_value=0.0)
        B,S,_,_,_ = img_tensor.shape
        video_mask = torch.sum(img_tensor.reshape(B,S,-1),dim=-1) != 0 
        video_mask = video_mask.long()
        return img_tensor, video_mask, video_id