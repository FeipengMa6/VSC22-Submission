import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os

from src.extractor import extract_vsc_feat,extract_isc_feat
from src.transform import sscd_transform,eff_transform,vit_transform
from src.dataset import D_vsc
import argparse
from vsc.storage import load_features,store_features

TRANSFORMS = {"imagenet":sscd_transform,"effnet":eff_transform,'vit':vit_transform}

def main(args):
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',rank=args.local_rank,world_size=world_size)
    device = torch.device("cuda",args.local_rank)
    checkpoint_path = args.checkpoint_path # './checkpoints/sscd_v60.torchscript.pt'
    model = torch.jit.load(checkpoint_path)
    model.to(device)
    model = DDP(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    model.eval()
    img_size = args.img_size
    with open(args.input_file, "r", encoding="utf-8") as f:
        vids = [x.strip() for x in f]
    dataset = D_vsc(vids,args.zip_prefix,img_size=img_size,transform=TRANSFORMS[args.transform](args.img_size,args.img_size),max_video_frames=256)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,num_workers=4,drop_last=False, pin_memory=False, # ,prefetch_factor=4,
                            sampler=DistributedSampler(dataset,shuffle=False),collate_fn=dataset.collate_fn)
    vids,features,timestamps = extract_vsc_feat(model,dataloader,device)
    np.savez(args.save_file+f'_{args.local_rank}.npz',video_ids=vids,features=features,timestamps=timestamps)
    dist.barrier()
    if(args.local_rank == 0):
        # merge sub files and remove
        feats = []
        vids = []
        timestamps = []
        nums = world_size
        for i in range(nums):
            file = args.save_file+f'_{i}'+'.npz'
            partial_data = np.load(file)
            vids.extend(list(partial_data['video_ids']))
            timestamps.extend(list(partial_data['timestamps']))
            feats.append(partial_data['features'])
            del partial_data
            print("To remove: ",file)
            os.remove(file) # remove sub files
        feats = np.concatenate(feats)
        timestamps = np.array(timestamps)
        np.savez(args.save_file+'.npz',video_ids=vids,features=feats,timestamps=timestamps)
        features = load_features(args.save_file+'.npz')
        features = sorted(features,key=lambda x:x.video_id)
        store_features(args.save_file+'.npz',features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAIN parameters')
    parser.add_argument('--save_file', default="test_refs")
    parser.add_argument('--zip_prefix', default="")
    parser.add_argument('--input_file', default="test/test_reference.txt")
    parser.add_argument('--input_file_root', default="")
    parser.add_argument('--checkpoint_path', default="")
    parser.add_argument('--save_file_root', default="./outputs")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size',type=int,default=320)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--dataset',type=str,default='vsc')
    parser.add_argument('--transform',type=str,default='imagenet')
    
    args = parser.parse_args()
    input_file_root = args.input_file_root
    save_file_root = args.save_file_root
    args.input_file = os.path.join(input_file_root,args.input_file)
    args.save_file = os.path.join(save_file_root,args.save_file)
    os.makedirs(args.save_file_root,exist_ok=True)
    main(args)