from pathlib import Path
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from src.dataset import VideoDataset, MatchClassifyDataset, MatchRefineDataset
from src.image_preprocess import image_process
from src.utils import calclualte_low_var_dim, generate_candidates_classfiy_feature, generate_matching_feature, generate_matching_result, transform_features
import tqdm
from PIL import Image
from torchvision import transforms
from vsc.index import VideoFeature
import faiss
from vsc.storage import load_features, store_features
from vsc.baseline.score_normalization import query_score_normalize
from vsc.metrics import Dataset
from sklearn.preprocessing import normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import gc

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
    NORM_DATA_FILE = Path("../infer/outputs/test_refs.npz")
    REF_DATA_FILE = Path("../infer/outputs/train_refs.npz")
    REF_NORM_DATA_FILE = Path("../infer/outputs/train_refs_sn.npz")
else:
    DATA_DIRECTORY = Path(f"../data/videos/{args.split}")
    NORM_DATA_FILE = Path("../infer/outputs/train_refs.npz")
    REF_DATA_FILE = Path("../infer/outputs/test_refs.npz")
    REF_NORM_DATA_FILE = Path("../infer/outputs/test_refs_sn.npz")

QRY_VIDEOS_DIRECTORY = DATA_DIRECTORY / "query"
OUTPUT_FILE = f"../infer/outputs/matching/{args.split}_matching.csv"
QUERY_SUBSET_FILE = Path(f"../data/meta/{args.split}/{args.split}_query_metadata.csv")

SSCD_MODELS = [MODEL_ROOT_DIRECTORY / "swinv2_v115.torchscript.pt",MODEL_ROOT_DIRECTORY / "swinv2_v107.torchscript.pt",
               MODEL_ROOT_DIRECTORY / "swinv2_v106.torchscript.pt",MODEL_ROOT_DIRECTORY / "vit_v68.torchscript.pt"]
CLS_MODEL_LIST = [MODEL_ROOT_DIRECTORY / "submit_cls_model1.pt", MODEL_ROOT_DIRECTORY / "submit_cls_model2.pt"]
REFINE_MODEL_LIST = [MODEL_ROOT_DIRECTORY / "submit_match_model1.pt", MODEL_ROOT_DIRECTORY / "submit_match_model2.pt" ]
PCA_MODEL = MODEL_ROOT_DIRECTORY / "pca_model.pkl" # TODO

SEARCH_THRESHOLD = -0.1
MATCH_CLS_THRESHOLD = 0.0005
MATCH_REFINE_THRESHOLD_LOW = 0.001
MATCH_REFINE_THRESHOLD_MID = 0.1
MATCH_REFINE_THRESHOLD_HIGH = 0.35

EMBEDDING_BATCH = 48



class Main:

    def __init__(self):
        query_subset = pd.read_csv(QUERY_SUBSET_FILE).head(100)
        query_subset_video_ids = query_subset.video_id.values.astype('U')
        self.device = torch.device("cuda")
        with open(PCA_MODEL, 'rb') as f:
            self.pca_model = pickle.load(f)
        self.sscd_feature_keys = ["swin_feature", "swin_feature", "swin_feature", "vit_transform"]
        self.sscd_models = []
        for model_str in SSCD_MODELS:
            self.sscd_models.append(torch.jit.load(model_str).eval().cuda())
        assert len(self.sscd_models) == len(self.sscd_feature_keys)
        self.swin_transform = transforms.Compose(
            [
                transforms.Resize([256, 256], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.vit_transform = transforms.Compose(
            [
                transforms.Resize([384, 384], interpolation=BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.dataset = VideoDataset(
            QRY_VIDEOS_DIRECTORY, fps=1, vids=query_subset_video_ids, 
            preprocess = image_process, 
            transform1 = {},
            transform2 = {
                'swin_feature': self.swin_transform, 
                'vit_transform': self.vit_transform
                }
        )
        self.dataloader = DataLoader(
            self.dataset, batch_size=4, collate_fn=lambda x:x, shuffle=False, prefetch_factor=6, num_workers=6
        )
        self.cls_model_list = []
        for model_file in CLS_MODEL_LIST:
            cls_model = torch.jit.load(model_file).eval()
            cls_model.cuda()
            self.cls_model_list.append(cls_model)
        self.refine_model_list = []
        for model_file in REFINE_MODEL_LIST:
            refine_model = torch.jit.load(model_file).eval()
            refine_model.cuda()
            self.refine_model_list.append(refine_model)
        self.vid_feature_len_map = {}

    def single_infer(self, sscd_model, sscd_feature):
        start = 0
        flat_feature_list = []
        with torch.no_grad():
            while start < len(sscd_feature):
                flat_feature = sscd_model(sscd_feature[start:start+EMBEDDING_BATCH,...])
                if flat_feature.dim() == 3:
                    flat_feature = flat_feature[:, 0]
                flat_feature_list.append(flat_feature.cpu().numpy())
                start += EMBEDDING_BATCH
        flat_feature = np.concatenate(flat_feature_list, axis=0)
        return flat_feature


    def process_feature(self, video_feature):
        num_frames = len(video_feature['frames'])
        timestamp = video_feature['timestamp']
        embed_list = []
        for feature_key, sscd_model in zip(self.sscd_feature_keys, self.sscd_models):
            sscd_feature = video_feature[feature_key].cuda()
            embed_list.append(self.single_infer(sscd_model, sscd_feature))
        embed_list = [normalize(x) for x in embed_list]
        embeds = np.concatenate(embed_list, axis=1)
        flat_feature = self.pca_model.transform(embeds)
        num_split_frames = len(sscd_feature)
        split_ratio = num_split_frames // num_frames
        timestamp = timestamp * split_ratio
        assert len(timestamp) == len(flat_feature)
        feat = VideoFeature(
            video_id=video_feature['name'],
            timestamps=np.array(timestamp),
            feature=flat_feature
        )
        self.vid_feature_len_map[video_feature['name']] = num_frames
        return feat

    def match_classify(self, match_cls_feature, match_cls_info):
        cls_dataset = MatchClassifyDataset(match_cls_feature, match_cls_info, (160, 160))
        cls_dataloader = DataLoader(cls_dataset, batch_size=2048, shuffle=False)
        pred_list = []
        qid_list = []
        rid_list = []
        for feature, qids, rids in tqdm.tqdm(cls_dataloader):
            with torch.no_grad():
                pred_list_ = [model(feature.cuda()).cpu() for model in self.cls_model_list]
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
    
    def match_refine(self, match_meta):
        test_dataset = MatchRefineDataset(match_meta, resolution=(224, 224))
        testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        res_list = []
        for feature, qid, rid, h, w in tqdm.tqdm(testloader):
            with torch.no_grad():
                feature_ = feature.cuda()
                pred_list = []
                for model in self.refine_model_list:
                    pred = model(feature_).cpu()
                    pred = pred.softmax(dim=1)
                    pred = pred.numpy()
                    pred_t = model(feature_.transpose(3, 2)).cpu()
                    pred_t = pred_t.softmax(dim=1)
                    pred_t = pred_t.transpose(3, 2).numpy()
                    pred += pred_t
                    pred /= 2
                    pred_list.append(pred)
                pred = sum(pred_list) / len(self.refine_model_list)
                h = h.numpy()
                w = w.numpy()
            for i in range(len(pred)):
                h_, w_ = h[i], w[i]
                p = pred[i][1][:h_, :w_]
                fea = feature[i][0][:h_, :w_]
                res_list.append([qid[i], rid[i], p, fea.numpy()])
        return res_list

    def run(self):
        query_list = []
        for feature in tqdm.tqdm(self.dataloader):
            for fea in feature:
                query_list.append(self.process_feature(fea))
        torch.cuda.empty_cache()
        score_norm_refs = load_features(NORM_DATA_FILE, Dataset.REFS)
        low_var_dim = calclualte_low_var_dim(score_norm_refs) # should be consisent with ref features
        sn_query_list = query_score_normalize(query_list, score_norm_refs, low_var_dim, beta=1.5, nk=10)
        del score_norm_refs
        gc.collect()
        sn_refs = load_features(REF_NORM_DATA_FILE, Dataset.REFS)
        index_cpu = faiss.index_factory(512, "Flat", faiss.METRIC_INNER_PRODUCT)
        ref_id_list = []
        for ref_vf in sn_refs:
            ref_id_list.extend([ref_vf.video_id for _ in range(ref_vf.feature.shape[0])])
            index_cpu.add(ref_vf.feature)
        del sn_refs
        gc.collect()
        if faiss.get_num_gpus() > 0:
            index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
        search_res_map = {}
        k = min(index_gpu.ntotal, 1024)
        for vf in tqdm.tqdm(sn_query_list):
            vf_id = vf.video_id
            vf_feature = vf.feature
            D, I = index_gpu.search(vf_feature, k)
            mask = D[:, k - 1] > SEARCH_THRESHOLD
            if mask.sum() > 0:
                lim_remain, D_remain, I_remain = index_cpu.range_search(vf_feature[mask], SEARCH_THRESHOLD)
            D_res, I_res = [], []
            nr = 0
            for i in range(len(vf_feature)):
                if not mask[i]:
                    nv = (D[i, :] > SEARCH_THRESHOLD).sum()
                    D_res.extend(list(D[i, :nv]))
                    I_res.extend(list(I[i, :nv]))
                else:
                    l0, l1 = lim_remain[nr], lim_remain[nr + 1]
                    D_res.extend(list(D_remain[l0:l1]))
                    I_res.extend(list(I_remain[l0:l1]))
                    nr += 1
            for dis, idx in zip(D_res, I_res):
                recall_id = ref_id_list[idx]
                recall_pair = (vf_id, recall_id)
                if recall_pair in search_res_map:
                    search_res_map[recall_pair] = max(search_res_map[recall_pair], dis)
                else:
                    search_res_map[recall_pair] = dis
        search_res_list = [(qid, rid, dis) for (qid, rid), dis in search_res_map.items()]
        search_res_list.sort(key=lambda x: -x[2])
        search_res_df = pd.DataFrame(search_res_list)
        search_res_df.columns = ['query_id', 'ref_id', 'score']
        search_res_df.to_csv('match_candidates_score.csv', index=False)
        del index_cpu
        del index_gpu
        torch.cuda.empty_cache()
        gc.collect()
        refs = load_features(REF_DATA_FILE, Dataset.REFS)
        query_list, refs = [
            transform_features(x, normalize) for x in [query_list, refs]
        ]
        query_map = {vf.video_id:vf.feature for vf in query_list}
        ref_map = {vf.video_id:vf.feature for vf in refs}
        match_cls_feature, match_cls_info = generate_candidates_classfiy_feature(
            query_map, ref_map, search_res_list, self.vid_feature_len_map
        )
        match_cls_df = self.match_classify(match_cls_feature, match_cls_info)
        select_cand = match_cls_df.groupby(['query_id', 'ref_id']).prob.max()
        select_cand = select_cand.reset_index(None)
        select_cand = select_cand[select_cand.prob > MATCH_CLS_THRESHOLD]
        candidate_score_list = list(zip(select_cand.query_id.values, select_cand.ref_id.values, select_cand.prob.values))
        match_meta = generate_matching_feature(
            query_map, ref_map,  self.vid_feature_len_map, candidate_score_list)
        torch.cuda.empty_cache()
        match_refine_res = self.match_refine(match_meta)
        match_res_high = generate_matching_result(match_refine_res, threshold=MATCH_REFINE_THRESHOLD_HIGH, std_ratio=0.5)
        match_res_mid = generate_matching_result(match_refine_res, threshold=MATCH_REFINE_THRESHOLD_MID, std_ratio=1.25)
        match_res_low = generate_matching_result(match_refine_res, threshold=MATCH_REFINE_THRESHOLD_LOW, std_ratio=2)
        match_df = pd.DataFrame(match_res_high + match_res_mid + match_res_low)
        match_df.columns = ['query_id', 'ref_id', 'query_start', 'ref_start', 'query_end', 'ref_end', 'score']
        max_match = match_df.groupby(['query_id', 'ref_id', 'query_start', 'ref_start', 'query_end', 'ref_end']).score.max()
        match_df = max_match.reset_index(None)
        match_df['query_start'] = match_df.query_start.astype(np.float64)
        match_df['ref_start'] = match_df.ref_start.astype(np.float64)
        match_df['query_end'] = match_df.query_end.astype(np.float64)
        match_df['ref_end'] = match_df.ref_end.astype(np.float64)
        match_df[['query_id', 'ref_id', 'query_start','query_end', 'ref_start', 'ref_end', 'score']].to_csv(OUTPUT_FILE, index=False) 

        
if __name__ == '__main__':
    main = Main()
    main.run()