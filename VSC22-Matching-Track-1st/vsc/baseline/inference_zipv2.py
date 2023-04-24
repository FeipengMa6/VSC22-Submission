import argparse
import os
import time
import pickle
import glob
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv import Config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vsc.baseline.model_factory.utils import build_model, build_dataset
from vsc.baseline.utils.comm import all_gather
from vsc.index import VideoFeature
from vsc.storage import store_features

os.environ['TORCH_HOME'] = "/mnt/nanjing3cephfs/mmvision/edinliu"
def main(args):
    s_time = time.time()
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)
    print(world_size, args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=args.local_rank)
    device = torch.device("cuda", args.local_rank)
    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank
    model = build_model(cfg.model)
    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location="cpu")["state_dict"]
        state_dict_ = dict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                state_dict_[k[len("module."):]] = v
            else:
                state_dict_[k] = v
        model.load_state_dict(state_dict_)
        print(f"load ckpt from {args.ckpt}")
    # input = torch.randn(1,3,384,384)
    # jit_model = torch.jit.trace(model.backbone, input)
    # jit_model.save('my_temp.pt')
    model.cuda(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.eval()
    print(f"to_eval_dataset: {args.test}")
    cfg.data[args.test]['type'] = "VideoZipDataSetV2"
    infer_dataset = build_dataset(cfg.data[args.test])
    infer_dataloader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        sampler=DistributedSampler(infer_dataset, shuffle=False),
        collate_fn=lambda x: x
    )

    video_features = []
    vids_set = set()
    total = 0
    filter = 0
    # with torch.no_grad(), torch.cuda.amp.autocast():
    filter_threshold = cfg.data[args.test]['filter_threshold']
    print(f'filter threshold: {filter_threshold}')
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(infer_dataloader):
            for item in batch:
                frames = item["input"]
                if not frames:
                    continue
                timestamps = item["timestamp"]
                frames = torch.stack(frames)
                flat_feature = model(return_loss=False, is_inference=True, x=frames.cuda(device))
                if flat_feature.dim() == 3:
                    flat_feature = flat_feature[:, 0]  # bz2, h
                features = flat_feature.cpu().detach().numpy()  # bz, max_frames, h
                # if cfg.data[args.test].with_rotate:
                #     frames = torch.stack(item['input_90'])
                #     flat_feature_90 = model(return_loss=False, is_inference=True, x=frames.cuda(device))
                #     if flat_feature_90.dim() == 3:
                #         flat_feature_90 = flat_feature_90[:, 0]  # bz2, h
                #     frames = torch.stack(item['input_180'])
                    # flat_feature_180 = model(return_loss=False, is_inference=True, x=frames.cuda(device))
                    # if flat_feature_180.dim() == 3:
                    #     flat_feature_180 = flat_feature_180[:, 0]  # bz2, h
                    # frames = torch.stack(item['input_270'])
                    # flat_feature_270 = model(return_loss=False, is_inference=True, x=frames.cuda(device))
                    # if flat_feature_270.dim() == 3:
                    #     flat_feature_270 = flat_feature_270[:, 0]  # bz2, h
                    # features_90 = flat_feature_90.cpu().detach().numpy()
                    # features_180 = flat_feature_180.cpu().detach().numpy()
                    # features_270 = flat_feature_270.cpu().detach().numpy()
                    # feature_list = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in [features, features_90]]
                    # features = sum(feature_list) / 2
                timestamps = timestamps  # bz, max_frames, 2
                vid = item["name"]
                if vid in vids_set:
                    continue
                vids_set.add(vid)
                feat = features / np.linalg.norm(features, axis=1, keepdims=True)
                sim_mat = np.matmul(feat, feat.T) - np.eye(len(feat))
                sim_mean = sim_mat.mean(0)
                to_remove_idx = []
                for i in sim_mean.argsort()[::-1]:
                    if i in to_remove_idx:
                        continue
                    for j in np.where(sim_mat[i] > filter_threshold)[0]:
                        to_remove_idx.append(j)
                to_keep_idx = [i for i in range(len(sim_mat)) if i not in to_remove_idx]
                total += len(to_keep_idx)
                filter += len(to_remove_idx)
                features = features[to_keep_idx]
                timestamps = timestamps[to_keep_idx]
                feat = VideoFeature(
                    video_id=vid,
                    timestamps=timestamps,
                    feature=features,
                )
                video_features.append(feat)
    tmp_file = f'{args.output_filename}_{args.local_rank}'
    with open(tmp_file, 'wb') as f:
        pickle.dump(video_features, f)
    files = glob.glob(args.output_filename + '_*')
    feature_list = []
    if len(files) == world_size:
        for file in files:
            with open(file, 'rb') as f:
                feature_list.extend(pickle.load(f))
            os.remove(file)
        feature_list.sort(key=lambda v: v.video_id)
        store_features(args.output_filename, feature_list)
    print(f"cost time is {time.time() - s_time}, {total}, {filter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAIN parameters')
    parser.add_argument('--config', default="")
    parser.add_argument('--output_filename',
                        default="/mnt/nanjingcephfs/mm-base-vision/tyewang/downloads/metaai/test")
    parser.add_argument('--test', default='')
    parser.add_argument('--ckpt', default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    main(args)
