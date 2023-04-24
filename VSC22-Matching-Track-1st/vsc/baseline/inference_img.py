import argparse
import io
import os
import time
import zipfile

import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vsc.baseline.model_factory.utils import build_model, build_dataset
from vsc.baseline.utils.comm import all_gather


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

    with torch.no_grad():
        for batch in tqdm(infer_dataloader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            frames = batch["input"]
            frame_id = batch["frame_index"]
            img_id = batch["img_index"]

            features = model(return_loss=False, is_inference=True, x=frames)  # bz2, h

            if features.dim() == 3:
                features = features[:, 0]  # bz2, h

            video_id = []  # bz
            for x in all_gather(batch["name"]):
                video_id.extend(x)

            frame_id = torch.cat([x.cuda(args.local_rank) for x in all_gather(frame_id)], dim=0).cpu().numpy()
            img_id = torch.cat([x.cuda(args.local_rank) for x in all_gather(img_id)], dim=0).cpu().numpy()
            features = torch.cat([x.cuda(args.local_rank) for x in all_gather(features)], dim=0).cpu().numpy()

            assert frame_id.shape[0] == img_id.shape[0] == features.shape[0] == len(video_id)

            if args.local_rank == 0:
                for i in range(features.shape[0]):
                    pattern = f"{video_id[i]}_{frame_id[i]}_{img_id[i]}"
                    ioproxy = io.BytesIO()
                    np.save(ioproxy, features[i])
                    npy_str = ioproxy.getvalue()
                    output_handler.writestr(pattern, npy_str)

    if args.local_rank == 0:
        output_handler.close()
        print("cost time is {}".format(time.time() - s_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAIN parameters')
    parser.add_argument('--config', default="")
    parser.add_argument('--output_filename', default="/mnt/nanjingcephfs/mm-base-vision/tyewang/downloads/metaai/test")
    parser.add_argument('--ckpt', default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch', type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args = parser.parse_args()

    main(args)
