import numpy as np

from models import ClassifyModel, HRnet
import os
import torch
import pickle
from dataset import MatchClassifyDataset, MatchingRefineDataset
import numpy as np
version = 'v1.0'

source_dir = '../infer/outputs/matching'
target_dir = '../checkpoints'
match_meta_file = os.path.join(source_dir, f"match_meta_{version}.meta")
cls_meta_file = os.path.join(source_dir, f"match_cann_featrure_{version}.meta")


cls_file1 = os.path.join(source_dir, f"classify_h1_{version}.pt")
cls_file2 = os.path.join(source_dir, f"classify_h2_{version}.pt")
match_file1 = os.path.join(source_dir, f"match_h1_{version}.pt")
match_file2 = os.path.join(source_dir, f"match_h2_{version}.pt")


with open(match_meta_file, 'rb') as f:
    match_meta = pickle.load(f)

with open(cls_meta_file, 'rb') as f:
    cls_meta = pickle.load(f)

cls_dataset = MatchClassifyDataset(cls_meta['feature'], cls_meta['infos'], enhance=False)
match_dataset = MatchingRefineDataset(match_meta, resolution=(128, 128), random_t=False)

input1 = cls_dataset[0][0]
input2 = match_dataset[0][0]
input1 = torch.tensor(input1[np.newaxis, :])
input2 = torch.tensor(input2[np.newaxis, :])

cls_model1 = torch.load(cls_file1)
cls_model2 = torch.load(cls_file2)

match_model1 = torch.load(match_file1)
match_model2 = torch.load(match_file2)

to_save_cls_file1 = os.path.join(target_dir, 'submit_cls_model1.pt')
to_save_cls_file2 = os.path.join(target_dir, 'submit_cls_model2.pt')
to_save_match_file1 = os.path.join(target_dir, 'submit_match_model1.pt')
to_save_match_file2 = os.path.join(target_dir, 'submit_match_model2.pt')



cls_model1 = cls_model1.eval()
with torch.no_grad():
    out = cls_model1(input1)
    jit_model = torch.jit.trace(cls_model1, input1)

jit_model.save(to_save_cls_file1)
model = torch.jit.load(to_save_cls_file1)
with torch.no_grad():
    out1 = model(input1)

print(out, out1)



cls_model2 = cls_model2.eval()
with torch.no_grad():
    out = cls_model2(input1)
    jit_model = torch.jit.trace(cls_model2, input1)

jit_model.save(to_save_cls_file2)
model = torch.jit.load(to_save_cls_file2)
with torch.no_grad():
    out1 = model(input1)

print(out, out1)



match_model1 = match_model1.eval()
with torch.no_grad():
    out = match_model1(input2)
    jit_model = torch.jit.trace(match_model1, input2)

jit_model.save(to_save_match_file1)
model = torch.jit.load(to_save_match_file1)
with torch.no_grad():
    out1 = model(input2)

print((out1 - out).abs().max())


match_model2 = match_model2.eval()
with torch.no_grad():
    out = match_model2(input2)
    jit_model = torch.jit.trace(match_model2, input2)

jit_model.save(to_save_match_file2)
model = torch.jit.load(to_save_match_file2)
with torch.no_grad():
    out1 = model(input2)

print((out1 - out).abs().max())
