import argparse
import os
import time

import torch
import torch.distributed as dist
from mmcv import Config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vsc.baseline.model_factory.utils import build_model, build_dataset
from vsc.baseline.utils.comm import all_gather
import numpy as np
import io
import zipfile


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

    if args.local_rank == 0:
        output_handler = zipfile.ZipFile(args.output_filename, 'w', compression=zipfile.ZIP_STORED)

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

            dec_outputs = []
            dec_scores = []
            for i in range(0, flat_frames.size(0), 128):
                outputs = model(return_loss=False, is_inference=True, x=flat_frames[i: i + 128])
                logits_ = outputs["logits"]
                dec_outputs_ = outputs["dec_outputs"]  # bz, 6, w, h
                dec_scores_ = logits_.sigmoid()  # bz
                dec_outputs.append(dec_outputs_)
                dec_scores.append(dec_scores_)
            dec_outputs = torch.cat(dec_outputs)
            dec_scores = torch.cat(dec_scores)

            tot = 0
            stack_features = []
            stack_scores = []
            for n in frame_num:
                n = int(n)
                real_feats = dec_outputs[tot: tot + n]
                real_scores = dec_scores[tot: tot + n]
                tot += n
                stack_features.append(real_feats.detach().cpu().numpy())
                stack_scores.append(real_scores.detach().cpu().numpy())

            store_vids = []
            store_scores = []
            store_feats = []

            for x in all_gather(batch["name"]):
                store_vids.extend(x)

            for x in all_gather(stack_features):
                store_feats.extend(x)

            for x in all_gather(stack_scores):
                store_scores.extend(x)

            assert len(store_vids) == len(store_feats) == len(store_scores)

            if args.local_rank == 0:
                for sv, sf, ss in zip(store_vids, store_feats, store_scores):
                    ioproxy = io.BytesIO()
                    np.save(ioproxy, sf)
                    npy_str = ioproxy.getvalue()
                    output_handler.writestr(f"{sv}_f", npy_str)
                    ioproxy = io.BytesIO()
                    np.save(ioproxy, ss)
                    npy_str = ioproxy.getvalue()
                    output_handler.writestr(f"{sv}_s", npy_str)

    if args.local_rank == 0:
        output_handler.close()
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
