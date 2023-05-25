import os
import numpy as np
from sklearn.decomposition import PCA
from vsc.storage import load_features,store_features
from vsc.index import VideoFeature
from vsc.baseline.score_normalization import query_score_normalize,ref_score_normalize,score_normalize
from sklearn.preprocessing import normalize
import pickle
import joblib
import torch

def VList2Dict(V_List):
    V_Dict = {}
    for vf in V_List:
        V_Dict[vf.video_id] = vf
    return V_Dict

root_dir = "./outputs"
vers = ['swinv2_v115','swinv2_v107','swinv2_v106','vit_v68']
feat_type = ["train_refs","test_refs"] 
feature_dict = {}
for v in vers:
    feature_dict[f"{v}_train_refs"] = os.path.join(root_dir, f'{v}/train_refs.npz')
    feature_dict[f"{v}_test_refs"] = os.path.join(root_dir, f'{v}/test_refs.npz')


pca_model_path = "../checkpoints/pca_model.pkl"
# pca_model = joblib.load(pca_model_path)


def generate_pca_model(data_list):
    dlist = [np.concatenate([x.feature for x in data]) for data in data_list]
    dlist = np.concatenate(dlist, axis=1)
    pca_model = PCA(n_components=512, random_state=2023)
    pca_model.fit(dlist)
    return pca_model



if __name__=='__main__':
    to_load_features  = [feature_dict[f"{v}_train_refs"] for v in vers]
    features_list = [VList2Dict(load_features(path)) for path in to_load_features]
    train_features = []
    for vid in features_list[0].keys():
        vid_feats = [normalize(x[vid].feature) for x in features_list]
        vid_feats = np.concatenate(vid_feats,axis=1) 
        train_features.append(vid_feats)
    train_features = np.concatenate(train_features)
    pca_model = PCA(n_components=512, random_state=2023)
    pca_model.fit(train_features)
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca_model, f)
    for t in feat_type: # 遍历 type
        features_list = []
        merge_features_list = []
        for v in vers: # 遍历 v
            feat_path = feature_dict[f"{v}_{t}"]
            features_list.append(VList2Dict(load_features(feat_path))) # 将 list of VideoFeature 转为 dict
        for vid in features_list[0].keys():
            vid_feats = [normalize(x[vid].feature) for x in features_list] # norm
            vid_feats = np.concatenate(vid_feats,axis=1) # concat
            vid_feats = pca_model.transform(vid_feats)
            merge_features_list.append(VideoFeature(video_id=vid,feature=vid_feats,timestamps=features_list[0][vid].timestamps))
        store_features(f"{root_dir}/{t}.npz",merge_features_list)
    

    nk = 1 # TODO 
    beta = 1.2 # TODO
    OUTPUT_FILE = f"{root_dir}/test_refs_sn.npz"
    INPUT_FILE = f"{root_dir}/test_refs.npz"
    NORM_FILE = f"{root_dir}/train_refs.npz"
    score_norm_refs = load_features(NORM_FILE)
    refs = load_features(INPUT_FILE)
    sn_refs = ref_score_normalize(refs,score_norm_refs,nk=nk,beta=beta)
    store_features(OUTPUT_FILE,sn_refs)

    OUTPUT_FILE = f"{root_dir}/train_refs_sn.npz"
    INPUT_FILE = f"{root_dir}/train_refs.npz"
    NORM_FILE = f"{root_dir}/test_refs.npz"
    score_norm_refs = load_features(NORM_FILE)
    refs = load_features(INPUT_FILE)
    sn_refs = ref_score_normalize(refs,score_norm_refs,nk=nk,beta=beta)
    store_features(OUTPUT_FILE,sn_refs)