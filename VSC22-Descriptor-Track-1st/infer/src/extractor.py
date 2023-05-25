from zipfile import ZipFile
# from einops import rearrange
from tqdm import tqdm
import numpy as np
from vsc.baseline.utils.comm import all_gather
from vsc.index import VideoFeature
import torch

def extract_vsc_feat(model,dataloader,device):
    save_feat = []
    save_vids = []
    save_timestamps = []
    for batch in tqdm(dataloader):
        video_frames,video_mask,video_id = batch
        video_frames = video_frames.to(device)
        video_mask = video_mask.to(device)
        with torch.no_grad():
            frame_num = video_mask.sum(dim=1).long()
            timestamp = [range(i) for i in frame_num]
            video_id = [[i]*j for i,j in zip(video_id,frame_num)]
            video_ids = []
            for vid in video_id:
                video_ids.extend(vid)
            flat_frames = video_frames[video_mask.bool()]
            flat_features = model(flat_frames)
            out_feature = flat_features
        assert out_feature.shape[0] == len(video_ids)
        
        
        save_feat.append(out_feature.cpu().detach().numpy())
        save_vids.extend(video_ids)
        save_timestamps.extend(timestamp)
        
    save_feat = np.concatenate(save_feat)
    save_timestamps = np.concatenate(save_timestamps)
    
    return save_vids,save_feat,save_timestamps

def extract_isc_feat(model,dataloader,device):
    save_feat = []
    save_vids = []
    save_timestamps = []
    for batch in tqdm(dataloader):
        image_ids,images = batch
        images = images.to(device)
        # video_mask = video_mask.to(device)
        with torch.no_grad():
            img_feats = model(images)
        timestamp = [np.array([1])]*len(image_ids)
        assert img_feats.shape[0] == len(image_ids)
        save_feat.append(img_feats.cpu().detach().numpy())
        save_vids.extend(image_ids)
        save_timestamps.extend(timestamp)
    save_feat = np.concatenate(save_feat)
    save_timestamps = np.concatenate(save_timestamps)
    
    return save_vids,save_feat,save_timestamps

# def extract_ref_feat(model,dataloader,device):
    