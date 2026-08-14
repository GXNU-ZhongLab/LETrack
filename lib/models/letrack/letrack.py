import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.letrack.vit import vit_base_patch16_224
from lib.models.letrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
import torch.nn.functional as F
len_hlist = 16

class LETrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.tome = ToMe(r=6)
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                profile=False,
                ):
        # assert isinstance(search, list), "The type of search is not List"

        if profile:

            out_dict = []
            hlist = torch.tensor([])

            x, aux_dict = self.backbone(template, search,
                                            hlist=hlist, mode='zx')
            if hlist.numel() > 0:
                hlist = torch.cat( (hlist,aux_dict['hlist']) , dim=1 )
            else:
                hlist = aux_dict['hlist']

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                    
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
                # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            # out['backbone_feat'] = x
                
            out_dict.append(out)
                
            return out_dict
        else:
            out_dict = []
            B,_,_,_ = template[0].size()
            hlist = torch.tensor([])
            
            for i in range(len(search)):
                x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                            hlist=hlist, mode='zx')


                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]
                    
                enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

                opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
                # Forward head
                out = self.forward_head(opt, None)


                ## update hlist
                if hlist.numel() == 0:                        
                    attn_s2ts = aux_dict['attn_s2ts']
                    attn_s2t = attn_s2ts[:,:,:64,64:].detach()
                    avg = attn_s2t.mean(dim=2).mean(dim=1)
                    score_map = out['score_map'].reshape(B,256)
                    final_scor = avg * score_map
                    _, indices = torch.sort(final_scor, dim=1, descending=True)
                    x_s = x[:,64:,:]
                    hlist = x_s.gather(dim=1, index=indices[:,:len_hlist].unsqueeze(-1).expand(x_s.shape[0], -1, x_s.shape[-1]))
                else:
                    attn_s2ts = aux_dict['attn_s2ts']
                    attn_s2t = attn_s2ts[:,:,:64,64+len_hlist:].detach()
                    avg = attn_s2t.mean(dim=2).mean(dim=1)
                    score_map = out['score_map'].reshape(B,256)
                    final_scor = avg * score_map
                    _, indices = torch.sort(final_scor, dim=1, descending=True)
                    x_s = x[:,64+len_hlist:,:]
                    dst = x_s.gather(dim=1, index=indices[:,:len_hlist].unsqueeze(-1).expand(x_s.shape[0], -1, x_s.shape[-1]))
                    toMerge = torch.cat([hlist, dst], dim=1)
                    hlist = self.tome(toMerge)
                    hlist = torch.stack(hlist, dim=0)

                out.update(aux_dict)
                out['backbone_feat'] = x
                out_dict.append(out)
                
            return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()

        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError

    def forward_test(self, template: torch.Tensor,
                search: torch.Tensor,
                mode,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                hlist = None
                ):

        out_dict = []
        for i in range(len(search)):
            x, aux_dict = self.backbone.forward_zx(template, x=search[i],
                                        hlist=hlist)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                    
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()

            out = self.forward_head(opt, None)


            ## update hlist
            if hlist.numel() == 0:                        
                attn_s2ts = aux_dict['attn_s2ts']
                attn_s2t = attn_s2ts[:,:,:64,64:].detach()
                avg = attn_s2t.mean(dim=2).mean(dim=1)
                score_map = out['score_map'].reshape(1,256)
                final_scor = avg * score_map
                _, indices = torch.sort(final_scor, dim=1, descending=True)
                x_s = x[:,64:,:]
                hlist = x_s.gather(dim=1, index=indices[:,:len_hlist].unsqueeze(-1).expand(x_s.shape[0], -1, x_s.shape[-1]))
            else:
                attn_s2ts = aux_dict['attn_s2ts']
                attn_s2t = attn_s2ts[:,:,:64,64+len_hlist:].detach()
                avg = attn_s2t.mean(dim=2).mean(dim=1)
                score_map = out['score_map'].reshape(1,256)
                final_scor = avg * score_map
                _, indices = torch.sort(final_scor, dim=1, descending=True)
                x_s = x[:,64+len_hlist:,:]
                dst = x_s.gather(dim=1, index=indices[:,:len_hlist].unsqueeze(-1).expand(x_s.shape[0], -1, x_s.shape[-1]))
                toMerge = torch.cat([hlist, dst], dim=1)
                hlist = self.tome(toMerge)
                hlist = torch.stack(hlist, dim=0)

            out.update(aux_dict)
            out['backbone_feat'] = x
                
            out_dict.append(out)
            aux_dict['hlist'] = hlist
        return out_dict, aux_dict
                



def build_letrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 2
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = LETrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    return model



def partition_indices(n_tokens):
    idxs = torch.arange(n_tokens)
    mid = n_tokens // 2
    first_half = idxs[:mid].tolist()
    second_half = idxs[mid:].tolist()
    return first_half, second_half



def match(tokens, src_idxs, dst_idxs, r):
    """
    tokens: (B, N, D)
    src_idxs: 源 token 索引
    dst_idxs: 目标 token 索引
    r: 需要合并的对数
    返回:
      edges_batch: list，每个 batch 是 [(src_idx, dst_idx, sim), ...]，确保一对一匹配
    """
    B, N, D = tokens.shape
    edges_batch = []

    for b in range(B):
        src_tokens = tokens[b, src_idxs]  # (len(src), D)
        dst_tokens = tokens[b, dst_idxs]  # (len(dst), D)

        # 归一化后计算余弦相似度 (S, D) x (D, T) -> (S, T)
        src_norm = F.normalize(src_tokens, dim=-1)
        dst_norm = F.normalize(dst_tokens, dim=-1)
        sim_matrix = torch.matmul(src_norm, dst_norm.T)  # (S, T)

        # 生成所有可能的(src, dst, sim)组合
        all_edges = []
        for i, s in enumerate(src_idxs):
            for j, d in enumerate(dst_idxs):
                sim = sim_matrix[i, j].item()
                all_edges.append((s, d, sim))

        # 按相似度降序排序
        all_edges.sort(key=lambda x: x[2], reverse=True)

        # 贪心算法选择一对一匹配
        selected_src = set()
        selected_dst = set()
        selected_edges = []
        
        for edge in all_edges:
            s, d, sim = edge
            if s not in selected_src and d not in selected_dst:
                selected_src.add(s)
                selected_dst.add(d)
                selected_edges.append(edge)
                if len(selected_edges) >= r:
                    break  # 达到需要的匹配对数

        edges_batch.append(selected_edges)

    return edges_batch

def merge(tokens, edges_batch, src_idxs, dst_idxs, r):
    """
    tokens: (B, N, D)
    edges_batch: 每个 batch 的匹配边 [(src, dst, sim), ...]
    src_idxs, dst_idxs: 源和目标 token 索引
    r: 合并对数
    返回:
      reduced_tokens_list: list of (N_reduced, D)，其中 N_reduced = (len(src)-r) + r
    """
    B, N, D = tokens.shape
    reduced_tokens_list = []

    for b in range(B):
        edges = edges_batch[b]

        # 取前 r 个匹配 (src->dst)
        selected_edges = edges[:r]
        selected_src = {s for s, _, _ in selected_edges}
        selected_dst = {d for _, d, _ in selected_edges}

        # src 中：保留 n-r 个未被合并的
        kept_src = [i for i in src_idxs if i not in selected_src]

        # dst 中：只保留 r 个匹配到的
        kept_dst = [i for i in dst_idxs if i in selected_dst]

        # 合并后的索引
        kept_indices = kept_src + kept_dst

        reduced_tokens = tokens[b, kept_indices]
        reduced_tokens_list.append(reduced_tokens)

    return reduced_tokens_list



class ToMe(nn.Module):
    def __init__(self, r=8):
        super().__init__()
        self.r = r

    def forward(self, tokens):
        """
        tokens: (N, D)
        attn_fn: 一个函数，用于对 reduced tokens 做 self-attention（或其它处理）
        返回：恢复后的 tokens (N, D)
        """
        B, N, D = tokens.shape
        src_idxs, dst_idxs = partition_indices(N)
        edges_batch = match(tokens, src_idxs, dst_idxs, self.r)
        reduced_tokens_list = merge(tokens, edges_batch, src_idxs, dst_idxs, self.r)

 
        return reduced_tokens_list