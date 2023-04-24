import logging
import sys, os
import argparse
import random

import pandas as pd
import torch
from vsc.baseline.model_factory.utils import build_dataset
import torch.nn.functional as F
from video.model import MD
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mmcv import Config
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torchsnooper
import torch.nn as nn
from video.comm import all_gather as comm_gather
from sklearn.metrics import average_precision_score
from tqdm import tqdm


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
train_dataset.positive_ratio = args.positive_ratio
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=train_sampler)

val_dataset = build_dataset(cfg.data.val)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, drop_last=False, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, sampler=val_sampler)

model = MD(args)
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
    # vid_a, vid_b = batch_data["vid_a"], batch_data["vid_b"]
    labels = batch_data["labels"]
    cat_x = torch.cat([batch_data["frames_a"], batch_data["frames_b"]], dim=0)
    bz = batch_data["frames_a"].size(0)
    embeds = model(cat_x)
    embeds = embeds / embeds.norm(dim=1, keepdim=True)

    emb_a, emb_b = embeds[:bz], embeds[bz:]

    # loss_ = F.cosine_embedding_loss(emb_a, emb_b, labels, margin=args.margin)
    loss_ = F.binary_cross_entropy_with_logits(F.cosine_similarity(emb_a, emb_b, dim=1) / 0.12, labels.float())

    emb_a, emb_b, labels = all_gather(args.local_rank, world_size, emb_a=emb_a, emb_b=emb_b, labels=labels[:, None])
    labels = labels.squeeze(1)

    sims = emb_a @ emb_b.t()
    gt = torch.arange(sims.size(0)).to(sims.device)
    vcv_loss_ = F.cross_entropy(sims / 0.07, gt, reduction="none")[labels.gt(0)].mean()
    loss_ = vcv_loss_ + loss_

    scores_ = F.cosine_similarity(emb_a, emb_b, dim=1).detach().cpu().numpy()
    labels_ = labels.int().detach().cpu().numpy()
    pn_ = (labels_ == 1).sum() / labels_.shape[0]
    ap_ = average_precision_score(labels_, scores_)

    return loss_, ap_, pn_


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
                loss, ap, pn = train_step(batch)
                scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(opt)
                scaler.update()
        else:
            loss, ap, pn = train_step(batch)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            opt.step()
        scheduler.step()
        global_step += 1

        if args.local_rank == 0 and b % print_freq == 0:
            logger.info('Epoch %d Batch %d Loss %.3f, AP %.3f, PN %.3f' % (e, b, loss.item(), ap, pn))

        if b > 0 and b % args.eval_freq == 0:
            # eval
            model.eval()
            vid2embedding = dict()

            with torch.no_grad(), torch.cuda.amp.autocast():
                for val_batch in val_loader:
                    vids_ = []
                    for vid_list in comm_gather(val_batch["vid"]):
                        vids_.extend(vid_list)

                    val_embeds = model(val_batch["frames"].cuda(args.local_rank))
                    val_embeds = val_embeds / val_embeds.norm(dim=1, keepdim=True)
                    tensors_temp = [torch.zeros_like(val_embeds) for _ in range(world_size)]
                    dist.all_gather(tensors_temp, val_embeds)

                    val_embeds = torch.cat(tensors_temp, dim=0).detach().cpu().numpy()

                    for idx, emb in zip(vids_, val_embeds):
                        vid2embedding[idx] = emb

            # do eval
            q_embeds = []
            r_embeds = []
            q_vids = []
            r_vids = []
            for idx, emb in vid2embedding.items():
                if idx.startswith("Q"):
                    q_embeds.append(emb)
                    q_vids.append(idx)
                else:
                    r_embeds.append(emb)
                    r_vids.append(idx)

            with torch.no_grad():
                q_embeds = torch.tensor(np.array([x.tolist() for x in q_embeds]))
                r_embeds = torch.tensor(np.array([x.tolist() for x in r_embeds]))
                scores = q_embeds @ r_embeds.t()  # q, (r1 + r2)
                scores = scores.detach().cpu().numpy()

            max_k = 1200
            norm_k = 1
            r1_scores = []
            r2_scores = []
            r1_vids = []
            for j, idx in enumerate(r_vids):
                if idx.startswith("R1"):
                    r1_scores.append(scores[:, j])
                    r1_vids.append(idx)
                else:
                    r2_scores.append(scores[:, j])

            r1_scores = np.stack(r1_scores, axis=1)  # q, r1
            r2_scores = np.stack(r2_scores, axis=1)  # q, r1

            if args.local_rank == 0:
                print("#####R1 Shape", r1_scores.shape)
                print("#####R2 Shape", r2_scores.shape)

            bias = np.sort(r2_scores, axis=1)[:, -norm_k:].mean(axis=1, keepdims=True)  # q, 1
            # norm_scores = r1_scores - bias
            norm_scores = r1_scores
            top_k_scores = np.sort(norm_scores, axis=1)[:, -max_k:]  # q, k
            top_k_index = np.argsort(norm_scores, axis=1)[:, -max_k:]  # q, k

            val_ann = set()
            with open(args.val_ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    id1, id2 = line.strip().split(",")
                    val_ann.add((id1, id2))

            pair_query = []
            pair_ref = []
            pair_scores = []
            pair_labels = []
            for r in range(top_k_scores.shape[0]):
                for c in range(top_k_scores.shape[1]):
                    k = top_k_index[r, c]
                    pair_scores.append(top_k_scores[r, c])
                    pair_query.append(q_vids[r])
                    pair_ref.append(r1_vids[k])
                    if (q_vids[r], r1_vids[k]) in val_ann:
                        pair_labels.append(1)
                    else:
                        pair_labels.append(0)

            pair_scores = np.array(pair_scores)
            pair_labels = np.array(pair_labels)

            result_df = pd.DataFrame()
            result_df["scores"] = pair_scores
            result_df["query"] = pair_query
            result_df["ref"] = pair_ref
            result_df["labels"] = pair_labels

            val_ap = average_precision_score(pair_labels, pair_scores)

            if args.local_rank == 0:
                print("*** Epoch %d Batch %d VAL AP %.3f" % (e, b, val_ap))
                logger.info("*** Epoch %d Batch %d VAL AP %.3f" % (e, b, val_ap))
                result_df.to_csv(f"{args.work_dir}/result_step{global_step}.csv", index=False)

                ckpt = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
                        'epoch': e}
                torch.save(ckpt, work_dir + '/checkpoints/epoch_%d_step_%d.pth' % (e, global_step))
            model.train()
