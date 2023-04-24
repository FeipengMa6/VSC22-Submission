import logging
import sys, os
import argparse
import random

import torch
from vsc.baseline.model_factory.utils import build_model, build_dataset
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mmcv import Config
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--t', type=float, default=0.05)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--output_dim', type=int, default=512)
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm', type=float, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=-1)
    parser.add_argument('--instance_mask', action='store_true', default=False)
    parser.add_argument('--entropy_loss', action='store_true', default=False)
    parser.add_argument('--entropy_weight', type=float, default=30)
    parser.add_argument('--ici_weight', type=float, default=1.)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--bl_weight', type=float, default=0.)
    args = parser.parse_args()
    return args


args = parse_args()

work_dir = args.work_dir
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
print_freq = args.print_freq
resume = args.resume if args.resume != '' else None
warmup_ratio = args.warmup_ratio


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


#setup_seed(1234)


def all_gather(local_rank, world_size, **tensors):
    tensors = list(tensors.values())
    _dims = [t.shape[-1] for t in tensors]
    tensors = torch.cat(tensors, dim=-1)
    tensors_all = [torch.zeros_like(tensors) for _ in range(world_size)]
    dist.all_gather(tensors_all, tensors)
    tensors_all[local_rank] = tensors
    tensors_all = torch.cat(tensors_all, dim=0)

    results = list()
    dimStart = 0
    assert sum(_dims) == tensors_all.shape[-1]
    for d in _dims:
        results.append(tensors_all[..., dimStart: dimStart + d])
        dimStart += d

    return tuple(results)


class MemoryBank(object):

    def __init__(self, k=20000):
        self.queue = None
        self.K = k

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = self.concat_all_gather(keys)

        if self.queue is None:
            self.queue = keys
        else:
            self.queue = torch.cat((keys, self.queue))[:self.K]

    @staticmethod
    @torch.no_grad()
    def concat_all_gather(tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output


class BarlowTwins(nn.Module):
    def __init__(self, lambd=5e-3):
        super().__init__()

        self.lambd = lambd

    def forward(self, e1, e2):

        z1 = (e1 - e1.mean(dim=0)) / e1.std(dim=0).clamp(1e-5)
        z2 = (e2 - e2.mean(dim=0)) / e2.std(dim=0).clamp(1e-5)
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        c.div_(z1.size(0) * dist.get_world_size())
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss_ = on_diag + self.lambd * off_diag

        return loss_

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

world_size = int(os.environ['WORLD_SIZE'])
args.rank = int(os.environ['RANK'])
torch.cuda.set_device(args.local_rank)
# torch.cuda.set_device(args.rank)

# dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size)
dist.init_process_group(backend='nccl')

# student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
# teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
# # we need DDP wrapper to have synchro batch norms working...
# teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
# teacher_without_ddp = teacher.module

# if args.local_rank == 0:
if args.rank == 0:
    os.system("mkdir -p %s" % work_dir)
    os.system("mkdir -p %s/checkpoints" % work_dir)
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(work_dir + '/log.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


cfg = Config.fromfile(args.config)
cfg.local_rank = args.local_rank

model = build_model(cfg.model)
model.cuda(args.local_rank)

train_dataset = build_dataset(cfg.data.train)
# train_dataset = ConcatDataset([build_dataset(cfg.data.train), build_dataset(cfg.data.train1)])
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=train_sampler)


# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

opt = AdamW(model.parameters(), lr=lr)
batch_size = batch_size * world_size
stepsize = (len(train_dataset) // batch_size + 1)
total_steps = (len(train_dataset) // batch_size + 1) * epochs
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_ratio * total_steps,
                                            num_training_steps=total_steps)

# model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
model = DDP(model, find_unused_parameters=True)
model._set_static_graph()
# teacher and student start with the same weights
# teacher_model.load_state_dict(model.module.state_dict())
# # there is no backpropagation through the teacher, so no need for gradients
# for p in teacher_model.parameters():
#     p.requires_grad = False

scaler = torch.cuda.amp.GradScaler()

start_epoch = 0
ckpt = None
if resume:
    ckpt = torch.load(resume, map_location='cpu')
elif os.path.exists(work_dir + '/last.txt'):
    f = open(work_dir + '/last.txt')
    e = int(f.readline())
    f.close()
    ckpt = torch.load(work_dir + '/checkpoints/epoch_%d.pth' % e, map_location='cpu')
if ckpt is not None:
    model.load_state_dict(ckpt['state_dict'])
    opt.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch'] + 1
    del ckpt

# MEMORY = MemoryBank()
barlow_loss_model = BarlowTwins().cuda(args.local_rank)


# @torchsnooper.snoop()
def contrast_loss_fn(emb_a, emb_b, temperature, mask, m=None):
    bz = emb_a.size(0)
    emb = torch.cat([emb_a, emb_b], dim=0)  # 2xbz
    sims = emb @ emb.t()
    diag = torch.eye(sims.size(0)).to(sims.device)

    small_value = torch.tensor(-10000.).to(sims.device).to(sims.dtype)
    sims = torch.where(diag.eq(0), sims, small_value)
    gt = torch.cat([torch.arange(bz) + bz, torch.arange(bz)], dim=0).to(sims.device)
    mask = torch.cat([mask, mask], dim=0).bool()
    loss_ = F.cross_entropy(sims / temperature, gt, reduction="none")[mask.bool()].mean()

    return loss_


def entropy_loss_fn(sims, mask):
    device = sims.device
    diag = torch.eye(sims.size(0)).to(device)
    # local_mask = mask[:, None] * mask[None, :] * (1 - diag)  # 加上这行 matching会提高，descriptor降低
    local_mask = (1 - diag)
    small_value = torch.tensor(-10000.).to(device).to(sims.dtype)
    max_non_match_sim = torch.where(local_mask.bool(), sims, small_value)[mask.bool()].max(dim=1)[0]
    closest_distance = (1 / 2 - max_non_match_sim / 2).clamp(min=1e-6).sqrt()
    entropy_loss_ = -closest_distance.log().mean() * args.entropy_weight
    return entropy_loss_


# @torchsnooper.snoop()
def train_step(batch_data):
    vid_a, vid_b = batch_data["vid_a"], batch_data["vid_b"]
    bz = batch_data["img_a"].size(0)
    device = batch_data["img_a"].device

    cat_x = torch.cat([batch_data["img_a"], batch_data["img_b"]], dim=0)

    embeds = model(x=cat_x)
    embeds_norm = embeds / embeds.norm(dim=1, keepdim=True)
    emb_a, emb_b = embeds[:bz], embeds[bz:2 * bz]
    emb_a_norm, emb_b_norm = embeds_norm[:bz], embeds_norm[bz:2*bz]

    ga_emb_a_norm, ga_emb_b_norm, ga_vid_a, ga_vid_b = all_gather(
        args.rank, world_size, emb_a=emb_a_norm, emb_b=emb_b_norm, vid_a=vid_a[..., None], vid_b=vid_b[..., None]
    )
    sims_norm = ga_emb_a_norm @ ga_emb_b_norm.t()

    rank_mask = torch.zeros(bz * dist.get_world_size()).to(device)
    rank_mask[args.rank*bz:(args.rank + 1)*bz] = 1

    entropy_loss_ = entropy_loss_fn(sims_norm, rank_mask)

    ici_loss_ = contrast_loss_fn(ga_emb_a_norm, ga_emb_b_norm, args.t, rank_mask) * args.ici_weight

    bl_loss_ = barlow_loss_model(emb_a_norm, emb_b_norm) * args.bl_weight

    return ici_loss_, entropy_loss_, bl_loss_


global_step = 0
for _e in range(start_epoch, epochs):
    model.train()
    train_sampler.set_epoch(_e)
    for _b, batch in enumerate(train_loader):

        for _k, _v in batch.items():
            if isinstance(_v, torch.Tensor):
                batch[_k] = _v.cuda(args.local_rank)

        opt.zero_grad()
        if args.fp16:
            with torch.cuda.amp.autocast():
                ici_loss, entropy_loss, bl_loss = train_step(batch)
                loss = ici_loss + entropy_loss + bl_loss
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(opt)
                scaler.update()
        else:
            ici_loss, entropy_loss, bl_loss = train_step(batch)
            loss = ici_loss + entropy_loss + bl_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()
        scheduler.step()

        global_step += 1
        if args.rank == 0 and _b % print_freq == 0:
            logger.info('Epoch %d Batch %d Loss %.3f, ICI Loss %.3f, Entropy loss %.3f, BL loss %.3f' % (
                _e, _b, loss.item(), ici_loss.item(), entropy_loss.item(), bl_loss.item())
            )

    # if args.local_rank == 0:
    if args.rank == 0:
        ckpt = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': _e}
        torch.save(ckpt, work_dir + '/checkpoints/epoch_%d.pth' % _e)
