import argparse
import os
import time

import pandas as pd
import torch
import torch.distributed as dist
from mmcv import Config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from comm import all_gather
from video.model import MS
from vsc.baseline.model_factory.utils import build_dataset


def main(args):
    s_time = time.time()
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size)
    device = torch.device("cuda", args.local_rank)

    cfg = Config.fromfile(args.config)
    cfg.local_rank = args.local_rank

    model = MS(args)

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

    test_dataset = build_dataset(cfg.data.test)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, sampler=test_sampler)

    vid2score = dict()
    for k, batch in tqdm(enumerate(test_loader)):
        for _k, _v in batch.items():
            if isinstance(_v, torch.Tensor):
                batch[_k] = _v.cuda(args.local_rank)

        vids = []
        for x in all_gather(batch["vid"]):
            vids.extend(x)

        with torch.no_grad():
            logits = model(batch["frames"].cuda(args.local_rank))
            scores = logits.sigmoid()
            scores = torch.cat([x.cuda(args.local_rank) for x in all_gather(scores)], dim=0)
            scores = scores.detach().cpu().numpy()

        for idx, s in zip(vids, scores):
            vid2score[idx] = s

    if args.local_rank == 0:
        data = []
        for k, v in vid2score.items():
            data.append([k, v])
        df = pd.DataFrame(data, columns=["id", "score"])
        df.to_csv(args.save_file, index=False)
        print("cost time is {}".format(time.time() - s_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAIN parameters')
    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--feat_dim', type=int, default=1024)
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--output_dim', type=int, default=256)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--bert_path', type=str, default="")
    parser.add_argument('--max_frames', type=int, default=256)
    parser.add_argument('--save_file', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
