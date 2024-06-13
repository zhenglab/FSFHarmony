# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch,math
import torch.nn.functional as F
from torch import nn, Tensor
from .ms_deform_attn import MSDeformAttn, MSDeformMatchV2Attn, MSDeformMatchV3Attn
from .st_transformerv5 import RSTBWithInputConv
# from ..lib import swin_transformer_v2, swin3d_transformer


class DeformableTransformerEncoders(nn.Module):
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, n_levels=4, enc_n_points=4, has_flow=False, t_attn='v0',window_size=4):
        super().__init__()

        # encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   n_levels, nhead, enc_n_points, has_flow=has_flow, t_attn=t_attn)
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          n_levels, nhead, enc_n_points, has_flow=has_flow, t_attn=t_attn)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, srcs, masks=None, pos_embeds=None, flow=None):
        memory,sampling_locations = self.encoder(srcs, pos_embeds, flow=flow)
        return memory,sampling_locations


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, has_flow=False, t_attn='v0',window_size=4):
        super().__init__()
        
        if t_attn == 'v0':
            self.temporal_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points,has_flow=has_flow)
        elif t_attn == 'v2':
            self.temporal_attn = MSDeformMatchV2Attn(d_model, n_levels, n_heads, n_points, has_flow=has_flow)
        elif t_attn == 'v3':
            self.temporal_attn = MSDeformMatchV3Attn(d_model, n_levels, n_heads, n_points, has_flow=has_flow)
        else:
            self.temporal_attn = None
        
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        norm_layer=nn.LayerNorm
        # self.spatial_attn = swin3d_transformer.BasicLayer(
        #             dim=d_model,
        #             depth=2,
        #             num_heads=8,
        #             window_size=(2, 8, 8),
        #             mlp_ratio=2.0,
        #             qkv_bias=True,
        #             qk_scale=None,
        #             drop_path=0.1,
        #             norm_layer=norm_layer
        #             )
        
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm0 = nn.LayerNorm(d_model)
        self.linear0 = nn.Linear(d_model, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None, flow=None):
        bs, t, hw, dim = src.size()
        # h=w=64
        # src2 = src.transpose(3,2).view(bs, t, dim, h, w).transpose(2,1)
        # src2 = self.spatial_attn(src2) #b c d h w [b, dim, t, h, w]
        # src2 = src2.flatten(3).permute(0,2,3,1)  # [b, t, h*w, dim]
        
        # src = self.norm0(src+self.linear0(src2))  # [b, t, h*w, dim]
        
        # temporal_deform_attn
        src = src.flatten(1,2)  # [b, t*h*w, dim]
        tgt = src
        src2, sampling_locations = self.temporal_attn(tgt, reference_points, src, spatial_shapes, level_start_index, padding_mask, flow=flow)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # ffn
        src = self.forward_ffn(src)
        src = src.view(bs, t, hw, dim)
        return src,sampling_locations

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(mask[:, :, 0], 1)  # shape: [1], value: H
        valid_W = torch.sum(mask[:, 0, :], 1)  # shape: [1], value: W
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio  # shape: [1, 2]
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [1, H*W, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [1, H*W*T, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # valid_ratios[:, None]: [1, 1, T, 2]
        return reference_points  # [1, H*W*T, T, 2]

    def forward(self, src, pos_embeds=None, padding_mask=None, flow=None):
        B, T, H, W, C = src.size()   #[8, 96, 16, 64, 64]
        spatial_shape = (H, W)
        spatial_shapes = torch.as_tensor(spatial_shape, dtype=torch.long, device=src.device)  # tensor([H, W])
        spatial_shapes = spatial_shapes.unsqueeze(0).repeat(T, 1)  # [T, 2] T个tensor([H, W])
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))  # [0, 1*64*64, 2*64*64, 3*64*64, 4*64*64] 每一层特征图的起始索引

        m = self.get_valid_ratio(torch.ones([1, H, W], device=src.device))  # [1, 2]
        valid_ratios = m.repeat(T, 1).unsqueeze(0)  # [1, T, 2]

        output = src.flatten(2,3)  # [B, T, H, W, C] → [B, T, H*W, C]
        # if pos_embeds is not None:
        #     pos_embeds = pos_embeds.flatten(1,3)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)  # [1, H*W*T, T, 2]

        for _, layer in enumerate(self.layers):
            output,sampling_locations = layer(output, pos_embeds, reference_points, spatial_shapes, level_start_index, padding_mask,flow=flow)
        output = output.view(B,T,H,W,C)
        location_shape = sampling_locations.size()
        sampling_locations = sampling_locations.view(B,T,H,W, location_shape[2],location_shape[3],location_shape[4],location_shape[5])
        return output,sampling_locations

class DeformableTransformerDecoders(nn.Module):
    
    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, n_levels=1, n_points=4):
        super().__init__()

        decoder_layer = DeformableTransformerDecoderLayer(d_model=d_model, d_ffn=dim_feedforward, dropout=dropout, activation=activation, \
                 n_levels=n_levels, n_heads=nhead, n_points=n_points)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, src_pos=None, tgt_pos=None, flow=None):
        hs = self.decoder(tgt, src, memory_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                          src_pos=src_pos, query_pos=tgt_pos, flow=flow)
        return hs

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, has_flow=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src_pos, tgt_reference_points, src_reference_points, src, \
                tgt_spatial_shapes, src_spatial_shapes, tgt_level_start_index, src_level_start_index, src_padding_mask=None, flow=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # cross attention
        
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               tgt_reference_points,
                               self.with_pos_embed(src, src_pos), src_spatial_shapes, src_level_start_index, src_padding_mask, flow=flow)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(mask[:, :, 0], 1)
        valid_W = torch.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
 
    def forward(self, tgt, src, memory_key_padding_mask=None, tgt_key_padding_mask=None,
                          src_pos=None, query_pos=None, flow=None):
        B, T, H, W, C = tgt.size()
        spatial_shape = (H, W)
        tgt_spatial_shapes = torch.as_tensor(spatial_shape, dtype=torch.long, device=tgt.device)
        
        tgt_spatial_shapes = tgt_spatial_shapes.unsqueeze(0).repeat(T, 1)
        m = self.get_valid_ratio(torch.ones([1, H, W], device=tgt.device))
        valid_ratios = m.repeat(T, 1).unsqueeze(0)
        tgt_reference_points = self.get_reference_points(spatial_shapes=tgt_spatial_shapes, valid_ratios=valid_ratios, device=tgt.device)
        tgt_level_start_index = torch.cat((tgt_spatial_shapes.new_zeros((1, )), tgt_spatial_shapes.prod(1).cumsum(0)[:-1]))
        

        src_spatial_shapes = tgt_spatial_shapes[:1,:].repeat(src.size(1), 1)
        valid_ratios = m.repeat(src.size(1), 1).unsqueeze(0)
        src_reference_points = self.get_reference_points(spatial_shapes=src_spatial_shapes, valid_ratios=valid_ratios, device=tgt.device)
        src_level_start_index = torch.cat((src_spatial_shapes.new_zeros((1, )), src_spatial_shapes.prod(1).cumsum(0)[:-1]))

        output = tgt.flatten(1,3)
        if query_pos is not None:
            query_pos = query_pos.flatten(1,2)
        src = src.flatten(1,3)
        if src_pos is not None:
            src_pos = src_pos.flatten(1,2)
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(output, query_pos, src_pos, tgt_reference_points, src_reference_points, src, \
                tgt_spatial_shapes, src_spatial_shapes, tgt_level_start_index, src_level_start_index, memory_key_padding_mask, flow=flow)

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(tgt_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        output = output.view(B,T,H,W,C)
        return output, tgt_reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

