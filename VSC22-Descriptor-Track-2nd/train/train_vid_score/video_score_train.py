import logging
import sys, os
import argparse
import random

import pandas as pd
import torch
from vsc.baseline.model_factory.utils import build_dataset
import torch.nn.functional as F
from video.model import MS
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mmcv import Config
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--feat_dim', type=int, default=1024)
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--output_dim', type=int, default=256)
    parser.add_argument('--bert_path', type=str, default="")
    parser.add_argument('--val_ann_path', type=str, default="")
    parser.add_argument('--max_frames', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--t', type=float, default=0.05)
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--positive_ratio', type=float, default=0.1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--work_dir', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--clip_grad_norm', type=float, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False)
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


setup_seed(1234)


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


world_size = int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size)

if args.local_rank == 0:
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

train_dataset = build_dataset(cfg.data.train)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=train_sampler)

val_dataset = build_dataset(cfg.data.val)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, drop_last=False, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=val_sampler)

model = MS(args)
model.cuda(args.local_rank)


opt = AdamW(model.parameters(), lr=lr)
batch_size = batch_size * world_size
stepsize = (len(train_dataset) // batch_size + 1)
total_steps = (len(train_dataset) // batch_size + 1) * epochs
scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_ratio * total_steps,
                                            num_training_steps=total_steps)

model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
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


def train_step(batch_data):
    labels = batch_data["labels"]
    frames = batch_data["frames"]
    logits = model(frames)
    scores = logits.sigmoid()

    loss_ = F.binary_cross_entropy_with_logits(logits, labels.float())

    scores, labels = all_gather(args.local_rank, world_size, scores=scores[:, None], labels=labels[:, None])
    scores = scores.squeeze(1)
    labels = labels.squeeze(1)

    acc = (scores.round() == labels).float().mean().item()
    pn = (labels == 1).sum() / labels.size(0)

    ap_ = average_precision_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())

    return loss_, ap_, acc, pn


global_step = 0
for e in range(start_epoch, epochs):
    model.train()
    train_sampler.set_epoch(e)
    for b, batch in enumerate(train_loader):

        for _k, _v in batch.items():
            if isinstance(_v, torch.Tensor):
                batch[_k] = _v.cuda(args.local_rank)

        opt.zero_grad()
        if args.fp16:
            with torch.cuda.amp.autocast():
                loss, ap, acc, pn = train_step(batch)
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(opt)
                scaler.update()
        else:
            loss, ap, acc, pn = train_step(batch)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()
        scheduler.step()
        global_step += 1

        if args.local_rank == 0 and b % print_freq == 0:
            logger.info('Epoch %d Batch %d Loss %.3f, AP %.3f, ACC %.3f, PN %.3f' % (e, b, loss.item(), ap, acc, pn))

        if b > 0 and global_step % args.eval_freq == 0:
            # eval
            model.eval()
            tot_val_labels = []
            tot_val_scores = []
            with torch.no_grad(), torch.cuda.amp.autocast():
                for val_batch in val_loader:
                    for _k, _v in val_batch.items():
                        if isinstance(_v, torch.Tensor):
                            val_batch[_k] = _v.cuda(args.local_rank)
                    val_labels = val_batch["labels"]
                    val_logits = model(val_batch["frames"].cuda(args.local_rank))
                    val_scores = val_logits.sigmoid()

                    val_scores, val_labels = all_gather(
                        args.local_rank, world_size, val_scores=val_scores[:, None], val_labels=val_labels[:, None]
                    )
                    val_scores = val_scores.squeeze(1)
                    val_labels = val_labels.squeeze(1)

                    tot_val_scores.append(val_scores.detach().cpu().numpy())
                    tot_val_labels.append(val_labels.detach().cpu().numpy())

            tot_val_labels = np.concatenate(tot_val_labels, axis=0)
            tot_val_scores = np.concatenate(tot_val_scores, axis=0)

            val_acc = (tot_val_scores.round() == tot_val_labels).mean()
            val_ap = average_precision_score(tot_val_labels, tot_val_scores)

            if args.local_rank == 0:

                print("*** Epoch %d Batch %d VAL AP %.3f, VAL ACC %.3f" % (e, b, val_ap, val_acc))
                logger.info("*** Epoch %d Batch %d VAL AP %.3f, VAL ACC %.3f" % (e, b, val_ap, val_acc))
                ckpt = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                        'epoch': e}
                torch.save(ckpt, work_dir + '/checkpoints/epoch_%d_step_%d.pth' % (e, global_step))
            model.train()
