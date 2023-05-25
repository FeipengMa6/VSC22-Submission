import argparse
import os
import time

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


def main(args):
    s_time = time.time()
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size)
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

    model.cuda(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.eval()

    infer_dataset = build_dataset(cfg.data.test)
    infer_dataloader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=int(args.batch_size),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        sampler=DistributedSampler(infer_dataset, shuffle=False)
    )

    video_features = []
    vids_set = set()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(infer_dataloader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            frames = batch["input"]
            video_mask = batch["input_mask"]
            timestamps = batch["timestamp"]

            max_frames = frames.size(1)
            frame_num = video_mask.sum(dim=1).long()
            flat_frames = frames[video_mask.bool()]  # bz2, c, w, h
            flat_feature = model(return_loss=False, is_inference=True, x=flat_frames)

            if flat_feature.dim() == 3:
                flat_feature = flat_feature[:, 0]  # bz2, h

            tot = 0
            stack_feature = []
            for n in frame_num:
                n = int(n)
                real_feat = flat_feature[tot: tot + n]
                feat = F.pad(real_feat, pad=(0, 0, 0, max_frames - real_feat.size(0)))
                tot += n
                stack_feature.append(feat)
            out_feature = torch.stack(stack_feature, dim=0)
            out_feature = out_feature * video_mask[..., None]
            out_feature = out_feature.reshape(-1, max_frames, out_feature.size(-1))

            vids = []  # bz
            for x in all_gather(batch["name"]):
                vids.extend(x)

            features = torch.cat([x.cuda(args.local_rank) for x in all_gather(out_feature)], dim=0)  # bz, max_frames, h
            features_mask = torch.cat([x.cuda(args.local_rank) for x in all_gather(video_mask)], dim=0).bool()
            timestamps = torch.cat([x.cuda(args.local_rank) for x in all_gather(timestamps)], dim=0)

            assert features.size(0) == features_mask.size(0) == timestamps.size(0) == len(vids)

            features = features.cpu().detach().numpy()  # bz, max_frames, h
            features_mask = features_mask.cpu().detach().numpy()  # bz, max_frames
            timestamps = timestamps.cpu().detach().numpy()  # bz, max_frames, 2

            for i in range(len(vids)):
                vid = vids[i]
                if vid in vids_set:
                    continue
                vids_set.add(vid)

                mask = features_mask[i]
                if mask.sum() == 0:  # 没有帧
                    continue

                feat = VideoFeature(
                    video_id=vid,
                    timestamps=timestamps[i][mask],
                    feature=features[i][mask],
                )
                video_features.append(feat)

    if args.local_rank == 0:
        video_features.sort(key=lambda v: v.video_id)
        store_features(args.output_filename, video_features)

    print("cost time is {}".format(time.time() - s_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAIN parameters')
    parser.add_argument('--config', default="")
    parser.add_argument('--output_filename',
                        default="")
    parser.add_argument('--ckpt', default="")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    main(args)
