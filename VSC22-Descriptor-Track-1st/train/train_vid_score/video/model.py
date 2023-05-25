import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class MD(nn.Module):

    def __init__(self, args):

        super(MD, self).__init__()

        self.frame_proj = nn.Sequential(nn.Linear(args.feat_dim, args.bert_dim), nn.LayerNorm(args.bert_dim))

        config = AutoConfig.from_pretrained(args.bert_path)
        config.gradient_checkpointing = args.gradient_checkpointing
        self.bert = AutoModel.from_pretrained(args.bert_path, config=config)
        self.max_frames = args.max_frames

        self.output_proj = nn.Linear(args.bert_dim * 2, args.output_dim)

    def forward(self, feats):

        vision_feats = self.frame_proj(feats)  # b, max_frames, h
        masks = feats.abs().sum(dim=2).gt(0)

        # add special tokens
        bz, device = vision_feats.size(0), vision_feats.device

        text = torch.tensor([101, 102], dtype=torch.long)[None].to(device)  # 1, 2
        emb = self.bert.get_input_embeddings()
        text_emb = emb(text).expand((bz, -1, -1))  # bz, 2, hidden
        cls_emb, sep_emb = text_emb[:, 0], text_emb[:, 1]

        inputs_embeds = torch.cat([cls_emb[:, None], vision_feats, sep_emb[:, None]], dim=1)
        masks = torch.cat([torch.ones((bz, 2)).to(device), masks], dim=1)

        states = self.bert(inputs_embeds=inputs_embeds, attention_mask=masks)[0]
        avg_pool = self._nonzero_avg_pool(states, masks)
        cls_pool = states[:, 0]
        cat_pool = torch.cat([cls_pool, avg_pool], dim=1)
        embeds = self.output_proj(cat_pool)

        return embeds

    @staticmethod
    def _random_mask_frame(frame, prob=0.15):

        # frame bs len 768
        mask_prob = torch.empty(frame.shape[0], frame.shape[1]).uniform_(0, 1).to(device=frame.device)
        mask = (mask_prob < prob).to(dtype=torch.long)
        frame = frame * (1 - mask.unsqueeze(2))
        return frame, mask

    @staticmethod
    def _nonzero_avg_pool(hidden, mask):
        mask = mask.to(hidden.dtype)
        hidden = hidden * mask[..., None]
        length = mask.sum(dim=1, keepdim=True)
        avg_pool = hidden.sum(dim=1) / (length + 1e-5)
        return avg_pool


class MS(nn.Module):

    def __init__(self, args):

        super(MS, self).__init__()

        self.frame_proj = nn.Sequential(nn.Linear(args.feat_dim, args.bert_dim), nn.LayerNorm(args.bert_dim))

        config = AutoConfig.from_pretrained(args.bert_path)
        config.gradient_checkpointing = args.gradient_checkpointing
        self.bert = AutoModel.from_pretrained(args.bert_path, config=config)
        self.max_frames = args.max_frames

        self.output_proj = nn.Linear(args.bert_dim * 2, 1)

    def forward(self, feats):

        vision_feats = self.frame_proj(feats)  # b, max_frames, h
        masks = feats.abs().sum(dim=2).gt(0)

        # add special tokens
        bz, device = vision_feats.size(0), vision_feats.device

        text = torch.tensor([101, 102], dtype=torch.long)[None].to(device)  # 1, 2
        emb = self.bert.get_input_embeddings()
        text_emb = emb(text).expand((bz, -1, -1))  # bz, 2, hidden
        cls_emb, sep_emb = text_emb[:, 0], text_emb[:, 1]

        inputs_embeds = torch.cat([cls_emb[:, None], vision_feats, sep_emb[:, None]], dim=1)
        masks = torch.cat([torch.ones((bz, 2)).to(device), masks], dim=1)

        states = self.bert(inputs_embeds=inputs_embeds, attention_mask=masks)[0]
        avg_pool = self._nonzero_avg_pool(states, masks)
        cls_pool = states[:, 0]
        cat_pool = torch.cat([cls_pool, avg_pool], dim=1)
        logits = self.output_proj(cat_pool).squeeze(1)

        return logits

    @staticmethod
    def _random_mask_frame(frame, prob=0.15):

        # frame bs len 768
        mask_prob = torch.empty(frame.shape[0], frame.shape[1]).uniform_(0, 1).to(device=frame.device)
        mask = (mask_prob < prob).to(dtype=torch.long)
        frame = frame * (1 - mask.unsqueeze(2))
        return frame, mask

    @staticmethod
    def _nonzero_avg_pool(hidden, mask):
        mask = mask.to(hidden.dtype)
        hidden = hidden * mask[..., None]
        length = mask.sum(dim=1, keepdim=True)
        avg_pool = hidden.sum(dim=1) / (length + 1e-5)
        return avg_pool

