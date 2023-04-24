import argparse
import io
import os
import time
import zipfile
from zipfile import ZipFile

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

import clip
from comm import all_gather

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class D(torch.utils.data.Dataset):
    def __init__(self, vids, zip_prefix, preprocess=None, args=None):
        self.transform = preprocess
        self.zip_prefix = zip_prefix
        self.vids = vids
        self.args = args

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, index):
        vid = self.vids[index]
        zip_path = "%s/%s/%s.zip" % (self.zip_prefix, vid[-2:], vid)
        img_tensor = torch.zeros(self.args.max_video_frames, 3, 224, 224)
        video_mask = torch.zeros(self.args.max_video_frames).long()

        try:
            with ZipFile(zip_path, 'r') as handler:
                img_name_list = handler.namelist()
                img_name_list = sorted(img_name_list)

                for i, img_name in enumerate(img_name_list):
                    i_img_content = handler.read(img_name)
                    i_img = Image.open(io.BytesIO(i_img_content))
                    i_img_tensor = self.transform(i_img)
                    img_tensor[i, ...] = i_img_tensor
                    video_mask[i] = 1
        except FileNotFoundError as e:
            print(e)

        return img_tensor, video_mask, vid


def main(args):
    s_time = time.time()
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=world_size)
    device = torch.device("cuda", args.local_rank)

    model = clip.from_pretrained(args.clip_dir)
    model.cuda(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    model.eval()

    output_path = args.save_file

    with open(args.input_file, "r", encoding="utf-8") as f:
        vids = [x.strip() for x in f]

    infer_dataset = D(vids, args.zip_prefix, preprocess=_transform(224), args=args)
    infer_dataloader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=int(args.bs),
        num_workers=8, drop_last=False,
        prefetch_factor=4, sampler=DistributedSampler(infer_dataset, shuffle=False)
    )

    if args.local_rank == 0:
        output_handler = zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_STORED)

    vid_set = set()
    for k, batch in tqdm(enumerate(infer_dataloader)):
        img = batch[0].to(device)  # .reshape(-1, 3, 224, 224)
        video_mask = batch[1].to(device)
        vids = []
        for x in all_gather(batch[2]):
            vids.extend(x)

        with torch.no_grad(), torch.cuda.amp.autocast():
            frame_num = video_mask.sum(dim=1).long()
            flat_frames = img[video_mask.bool()]  # bz2, c, w, h
            flat_feature = model(flat_frames)
            flat_feature = flat_feature[:, 0]  # bz2, h

            tot = 0
            stack_feature = []
            for n in frame_num:
                n = int(n)
                real_feat = flat_feature[tot: tot + n]
                feat = F.pad(real_feat, pad=(0, 0, 0, args.max_video_frames - real_feat.size(0)))
                tot += n
                stack_feature.append(feat)
            out_feature = torch.stack(stack_feature, dim=0)
            out_feature = out_feature * video_mask[..., None]
            out_feature = out_feature.reshape(-1, args.max_video_frames, out_feature.size(-1))

        features = torch.cat([x.cuda(args.local_rank) for x in all_gather(out_feature)], dim=0)
        features = features.cpu().detach().numpy().astype(np.float16)

        assert features.shape[0] == len(vids)

        if args.local_rank == 0:
            for i in range(features.shape[0]):
                vid = vids[i]
                if vid in vid_set:
                    continue
                vid_set.add(vid)
                ioproxy = io.BytesIO()
                np.save(ioproxy, features[i])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(vid, npy_str)

    if args.local_rank == 0:
        output_handler.close()
        print("cost time is {}".format(time.time() - s_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAIN parameters')
    parser.add_argument('--save_file', default="")
    parser.add_argument('--zip_prefix', default="")
    parser.add_argument('--input_file', default="")
    parser.add_argument('--clip_dir', type=str, default="")
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--max_video_frames', type=int, default=256)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    main(args)
