import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
from transformers import AutoModel, AutoConfig
from transformers.models.detr.modeling_detr import DetrHungarianMatcher, DetrLoss


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


class MM(nn.Module):

    def __init__(self, args):

        super(MM, self).__init__()
        self.args = args
        self.frame_proj = nn.Sequential(nn.Linear(args.feat_dim, args.bert_dim), nn.LayerNorm(args.bert_dim))

        config = AutoConfig.from_pretrained(args.bert_path)
        config.gradient_checkpointing = args.gradient_checkpointing
        self.bert = AutoModel.from_pretrained(args.bert_path, config=config)
        self.max_frames = args.max_frames

        self.query_num = args.query_num
        # self.query_embeddings = nn.Sequential(nn.Embedding(self.query_num, args.bert_dim), nn.LayerNorm(args.bert_dim))

        self.output_proj = nn.Linear(args.bert_dim * 2, 1)
        # Object detection heads
        # self.class_labels_classifier = nn.Linear(args.bert_dim, 2)  # We add one for the "no object" class
        #
        # self.bbox_predictor = nn.Sequential(
        #     nn.Linear(args.bert_dim, args.bert_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.bert_dim, args.bert_dim),
        #     nn.ReLU(),
        #     nn.Linear(args.bert_dim, 4)
        # )

        # matcher = DetrHungarianMatcher(
        #     class_cost=args.class_cost, bbox_cost=args.bbox_cost, giou_cost=args.giou_cost
        # )
        # self.obj_criterion = DetrLoss(
        #     matcher=matcher,
        #     num_classes=1,
        #     eos_coef=0.1,
        #     losses=["labels", "boxes", "cardinality"]
        # )

    # @torchsnooper.snoop()
    def forward(self, batch_data):

        feats_q = batch_data["feats_q"]
        feats_r = batch_data["feats_r"]

        length_q = feats_q.size(1)
        length_r = feats_r.size(1)
        feats = torch.cat([feats_q, feats_r], dim=1)
        vision_feats = self.frame_proj(feats)  # b, max_frames, h
        masks = feats.abs().sum(dim=2).gt(0)

        # add special tokens
        bz, device = vision_feats.size(0), vision_feats.device

        text = torch.tensor([101, 102], dtype=torch.long)[None].to(device)  # 1, 2
        emb = self.bert.get_input_embeddings()
        text_emb = emb(text).expand((bz, -1, -1))  # bz, 2, hidden
        cls_emb, sep_emb = text_emb[:, 0], text_emb[:, 1]

        # bz, 10, 768
        # query_embeddings = self.query_embeddings(torch.arange(0, self.query_num).to(device)[None].repeat((bz, 1)))
        # [CLS]Feat_q[SEP]Feat_r[SEP]query_embed[SEP]
        inputs_embeds = torch.cat([
            cls_emb[:, None], vision_feats[:, :length_q],
            sep_emb[:, None], vision_feats[:, length_q:],
            sep_emb[:, None]
        ], dim=1)

        mask_pad_values = torch.ones((bz, 1), dtype=masks.dtype).to(device)
        # query_mask_values = torch.ones((bz, self.query_num), dtype=masks.dtype).to(device)
        attention_mask = torch.cat([
            mask_pad_values, masks[:, :length_q],
            mask_pad_values, masks[:, length_q:],
            mask_pad_values
        ], dim=1)

        token_type_ids = torch.cat([
            torch.zeros((bz, length_q + 2)),  # [CLS]feat_q[SEP]
            torch.ones((bz, length_r + 1))  # query[SEP]
        ], dim=1).to(device).long()

        # 3 + size_q + size_r: [CLS] feat_q [SEP] feat_r [SEP]
        states = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        states = states * attention_mask[..., None].to(states.dtype)

        avg_pool = self._nonzero_avg_pool(states, attention_mask)
        cls_pool = states[:, 0]
        cat_pool = torch.cat([cls_pool, avg_pool], dim=1)
        logits = self.output_proj(cat_pool).squeeze(1)  # matching score
        # query_output = states[:, -(self.query_num + 1): -1]

        # m_logits = self.class_labels_classifier(query_output)
        # m_pred_boxes = self.bbox_predictor(query_output).sigmoid()

        if self.training:
            # targets = batch_data["m_labels"]
            # if isinstance(targets, dict):
            #
            #     keys = list(targets.keys())
            #     values = [[_t.squeeze(0) for _t in targets[k].split(1, dim=0)] for k in keys]
            #
            #     new_target = []
            #     for val in zip(*values):
            #         item = {_k: _v for _k, _v in zip(keys, val)}
            #         new_target.append(item)
            #
            #     targets = new_target

            # tgt_ids = torch.cat([v["class_labels"] for v in targets])

            loss_s = F.binary_cross_entropy_with_logits(logits, batch_data["s_labels"])

            # outputs_loss = {
            #     "logits": m_logits, "pred_boxes": m_pred_boxes
            # }

            # loss_dict = self.obj_criterion(outputs_loss,  targets)
            # loss_ce = loss_dict["loss_ce"] * self.args.class_cost
            # loss_bbox = loss_dict["loss_bbox"] * self.args.bbox_cost
            # loss_giou = loss_dict["loss_giou"] * self.args.giou_cost

            loss_ce = torch.tensor(0.)
            loss_bbox = torch.tensor(0.)
            loss_giou = torch.tensor(0.)

            loss = loss_ce + loss_bbox + loss_giou + loss_s
            outputs = dict(
                loss=loss, loss_ce=loss_ce, loss_bbox=loss_bbox, loss_giou=loss_giou, loss_s=loss_s,
                scores=logits.sigmoid()
            )
        else:
            outputs = dict(
                scores=logits.sigmoid()
            )

        return outputs

    @staticmethod
    def _nonzero_avg_pool(hidden, mask):
        mask = mask.to(hidden.dtype)
        hidden = hidden * mask[..., None]
        length = mask.sum(dim=1, keepdim=True)
        avg_pool = hidden.sum(dim=1) / (length + 1e-5)
        return avg_pool

