from pathlib import Path
import pandas as pd
import numpy as np
from vsc.storage import store_features
import torch
from torch.utils.data import DataLoader
from src.dataset import VideoDataset
from src.image_preprocess import image_process
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from vsc.index import VideoFeature
from vsc.baseline.score_normalization import query_score_normalize
from vsc.storage import load_features, store_features
from vsc.metrics import Dataset
from src.utils import calclualte_low_var_dim
from sklearn.preprocessing import normalize
import math
import pickle
import os
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

torch.jit.fuser('off')
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--split",type=str,default='train')
args = parser.parse_args()

MODEL_ROOT_DIRECTORY = Path("../checkpoints")
if(args.split in ["train","val"]):
    DATA_DIRECTORY = Path(f"../data/videos/train")
else:
    DATA_DIRECTORY = Path(f"../data/videos/{args.split}")

if(args.split in ["train","val"]):
    NORM_DATA_FILE = Path("./outputs/test_refs.npz") 
else:
    NORM_DATA_FILE = Path("./data/train_refs.npz") 
QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY / "query"
OUTPUT_FILE = f"./outputs/{args.split}_query_sn.npz"
QUERY_SUBSET_FILE = Path(f"../data/meta/{args.split}/{args.split}_query_metadata.csv")

SCORE_THRESHOLD = 0.001 
LEN_THRESHOLD = 48
FRAME_THRESHOLD = 0.975 
NK = 1
BETA = 1.2


SSCD_MODELS = [MODEL_ROOT_DIRECTORY / "swinv2_v115.torchscript.pt",MODEL_ROOT_DIRECTORY / "swinv2_v107.torchscript.pt",
               MODEL_ROOT_DIRECTORY / "swinv2_v106.torchscript.pt",MODEL_ROOT_DIRECTORY / "vit_v68.torchscript.pt"]

PCA_MODEL = MODEL_ROOT_DIRECTORY / "pca_model.pkl" # TODO

CLIP_MODEL = MODEL_ROOT_DIRECTORY / "clip.torchscript.pt"
VIDEO_SCORE_MODEL = MODEL_ROOT_DIRECTORY / "vsm.torchscript.pt"


class Main:

    def __init__(self):
        query_subset = pd.read_csv(QUERY_SUBSET_FILE)
        query_subset_video_ids = query_subset.video_id.values.astype('U')
        
        self.device = torch.device("cuda")
        self.clip_model = torch.jit.load(CLIP_MODEL)
        self.clip_model.eval()
        self.clip_model.cuda()

        self.video_score_model = torch.jit.load(VIDEO_SCORE_MODEL)
        self.video_score_model.eval()
        self.video_score_model.cuda()

        with open(PCA_MODEL, 'rb') as f:
            self.pca_model = pickle.load(f)

        self.sscd_feature_keys = ['sscd_feature_1','sscd_feature_2','sscd_feature_3','sscd_feature_4']

        self.sscd_models = dict()
        for i,k in enumerate(self.sscd_feature_keys):
            self.sscd_models[k] = torch.jit.load(SSCD_MODELS[i]).eval().cuda() # TODO right?

        self.video_score_transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)),
            ])
        self.sscd_transform_1 = transforms.Compose(
            [
                transforms.Resize([256, 256], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.sscd_transform_2 = transforms.Compose(
            [
                transforms.Resize([256, 256], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.sscd_transform_3 = transforms.Compose(
            [
                transforms.Resize([256, 256], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.sscd_transform_4 = transforms.Compose(
            [
                transforms.Resize([384, 384], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.dataset = VideoDataset(
            QRY_VIDEOS_DIRECTORY, fps=1, vids=query_subset_video_ids, 
            preprocess = image_process, 
            transform1 = {'video_score_feature': self.video_score_transform},
            transform2 = {'sscd_feature_1': self.sscd_transform_1,
                          'sscd_feature_2':self.sscd_transform_2,
                          'sscd_feature_3': self.sscd_transform_3,
                          'sscd_feature_4': self.sscd_transform_4,
                          }
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=1, collate_fn=lambda x:x, shuffle=False, prefetch_factor=6, num_workers=6
        )

        self.rnd_idx = 0
        self.video_scores = {}

    def single_infer(self,model,feature):
        process_times = math.ceil(feature.shape[0] / LEN_THRESHOLD)
        feature_list = []
        with torch.no_grad():
            for i in range(process_times):
                flat_feature = model(feature[i*LEN_THRESHOLD:(i+1)*LEN_THRESHOLD,...])
                if flat_feature.dim() == 3:
                    flat_feature = flat_feature[:, 0]
                feature_list.append(flat_feature.cpu().numpy())
            flat_feature = np.concatenate(feature_list,axis=0)
        return flat_feature

    def process(self, video_feature):
        timestamp = video_feature['timestamp']
        frames_tensor = video_feature['video_score_feature'].cuda()
        num_frames = len(frames_tensor)
        with torch.no_grad(), torch.cuda.amp.autocast():
            flat_feature = self.clip_model(frames_tensor[:256,...])[:, 0] # TODO check check
        if(num_frames <= 256): # TODO check check
            flat_feature = F.pad(flat_feature, pad=(0, 0, 0, 256 - num_frames))
        with torch.no_grad():
            logit = self.video_score_model(flat_feature.unsqueeze(0))
            score = logit.sigmoid()
            score = score.detach().cpu().numpy()[0]
            self.video_scores[video_feature['name']] = score

        sub_features = []
        for k in self.sscd_feature_keys:
            sscd_feature = video_feature[k].cuda() # TODO
            sub_feature = normalize(self.single_infer(self.sscd_models[k],sscd_feature))
            sub_features.append(sub_feature)

        features = np.concatenate(sub_features, axis=1)
        num_split_frames = len(features)
        split_ratio = num_split_frames // num_frames
        timestamp = timestamp * split_ratio
        assert len(timestamp) == len(features)
        sub_feats = []
        for sub_feature in sub_features:
            sub_feats.append(
                VideoFeature(
                video_id=video_feature['name'],
                timestamps=np.array(timestamp),
                feature=sub_feature,
            ))
        if score >= SCORE_THRESHOLD:
            # TODO check check
            feat = features / np.linalg.norm(features, axis=1, keepdims=True)
            sim_mat = np.matmul(feat, feat.T) - np.eye(len(feat))
            sim_mean = sim_mat.mean(0)
            to_remove_idx = []
            for i in sim_mean.argsort()[::-1]:
                if i in to_remove_idx:
                    continue
                for j in np.where(sim_mat[i] > FRAME_THRESHOLD)[0]:
                    to_remove_idx.append(j)
            to_keep_idx = [i for i in range(len(sim_mat)) if i not in to_remove_idx]
            features = features[to_keep_idx]

            # TODO do pca here
            features = self.pca_model.transform(features)
            timestamps = np.array(timestamp)[to_keep_idx]
            feat = VideoFeature(
                video_id=video_feature['name'],
                timestamps=timestamps,
                feature=features,
            )
        else:
            self.rnd_idx += 1
            np.random.seed(self.rnd_idx)
            random_feature = np.random.uniform(-1e-5,1e-5,size=512).astype(np.float32)
            timestamps = np.array([0,1])
            feat = VideoFeature(
                video_id=video_feature['name'],
                timestamps=timestamps[None,...],
                feature=random_feature[None,...],
            )
        return feat, sub_feats
    
    def run(self):
        feature_list = []
        sub_feature_list = []
        for feature in tqdm(self.dataloader):
            for fea in feature:
                feat, sub_feats = self.process(fea)
                feature_list.append(feat)
                sub_feature_list.append(sub_feats)
        torch.cuda.empty_cache()
        # save_sub_features
        for i, k in enumerate(SSCD_MODELS):
            sub_feat = [x[i] for x in sub_feature_list]
            model_key = os.path.split(k)[-1].split('.')[0]
            sub_dir = f"./outputs/{model_key}"
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            store_features(f"./outputs/{model_key}/{args.split}_query.npz", sub_feat)
        # score normalization
        score_norm_refs = load_features(NORM_DATA_FILE,Dataset.REFS)
        low_var_dim = calclualte_low_var_dim(score_norm_refs) # should be consisent with ref features
        feature_list = query_score_normalize(feature_list,score_norm_refs,self.video_scores,SCORE_THRESHOLD,low_var_dim,nk=NK,beta=BETA)

        store_features(OUTPUT_FILE , feature_list)




if __name__ == '__main__':
    main = Main()
    main.run()
