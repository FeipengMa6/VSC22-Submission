from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from ..utils import DATASETS


@DATASETS.register_module()
class EmbedDataSet(torch.utils.data.Dataset):

    def __init__(
            self, query_embeds_paths, ref_embeds_paths,
    ):
        print("loading embeddings")
        queries = []
        for path in tqdm(query_embeds_paths):
            data = np.load(path)
            features = data["features"] / np.linalg.norm(data["features"], axis=1, keepdims=True)
            queries.append(features.astype(np.float16))
        query_embeds = np.concatenate(queries, axis=1)

        refs = []
        for path in tqdm(ref_embeds_paths):
            data = np.load(path)
            features = data["features"] / np.linalg.norm(data["features"], axis=1, keepdims=True)
            refs.append(features.astype(np.float16))

        ref_embeds = np.concatenate(refs, axis=1)

        self.data = np.concatenate([query_embeds, ref_embeds], axis=0)
        print("Data Shape", self.data.shape)

        del queries, query_embeds, refs, ref_embeds

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        res = {
            "input_embed": deepcopy(self.data[index, :].astype(np.float32))
        }

        return res

