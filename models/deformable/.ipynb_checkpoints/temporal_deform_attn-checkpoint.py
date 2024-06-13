# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from .ops.functions import MSDeformAttnFunction
from models.lib import spynet
from models.lib.flow import matching,geometry

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.has_flow = has_flow
        if has_flow:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
            # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.has_flow:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, 1, self.n_points, 1)
            # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        else:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        if flow is None:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
        #     # query_t = query.unsqueeze(2).repeat(1,1, self.n_levels, 1)
        #     # input_flatten_tmp = input_flatten.view(N, self.n_levels, -1, dim)
        #     # h = w = int(math.sqrt(input_flatten_tmp.size(2)))
        #     # input_flatten_tmp = input_flatten_tmp.view(N, self.n_levels, h, w, dim).flatten(0,1).permute(0,3,1,2)
        #     # src_warp = spynet.flow_warp(input_flatten_tmp, flow.permute(0,2,3,1))
        #     # src_warp = src_warp.view(N, self.n_levels, dim, h, w).flatten(3).permute(0,3,1,2)
        #     # query_flows = query_t + src_warp
        #     # sampling_offsets = self.sampling_offsets(query_flows).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        #     h = w = int(math.sqrt(query.size(1)))
        #     sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        #     # sampling_offsets = sampling_offsets.permute(0,4,2,1).flatten(0,1).view(-1, h, w, 2)
        #     # flow = flow.unsqueeze(1).repeat(1, self.n_heads*self.n_points, 1, 1, 1).flatten(0,1)   #[2, 5, 5, 2, 64, 64]

            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, 1, self.n_points, 2)
            sampling_offsets = sampling_offsets.repeat(1, 1, 1, self.n_levels, 1, 1)

            
        #     # sampling_offsets = sampling_flow_offsets.flatten(1,2).view(N*self.n_levels, -1, Len_q, 2).view(N, self.n_levels,-1, Len_q,2)\
        #     #     .view(N, self.n_levels, self.n_heads, self.n_points, Len_q, 2).permute(0,4,2,1,3,5).contiguous()
        #     # print(sampling_offsets.is_contiguous())
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # print(sampling_offsets[0][100][0])
        # print(attention_weights[0][100][0])
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output, sampling_locations

class MSDeformQAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.has_flow = has_flow
        if has_flow:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
            self.attention_weights = nn.Linear(d_model*2, n_heads * n_points)
            # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.has_flow:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, 1, self.n_points, 1)
            # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        else:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)


        if flow is None:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:

            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, 1, self.n_points, 2)
            sampling_offsets = sampling_offsets.repeat(1, 1, 1, self.n_levels, 1, 1)

            h = w = input_spatial_shapes[0,0]
            t = int(Len_in/(h*w))
            #利用光流扭曲后的特征
            key_warp = input_flatten.view(N, t, h*w, dim).view(N, 1, t, h, w, dim).repeat(1, t, 1, 1, 1, 1).flatten(1,2).flatten(0,1).permute(0,3,1,2)
            flow = flow.flatten(1,2).flatten(0,1).permute(0,2,3,1)
            key_warp = spynet.flow_warp(key_warp, flow)
            key_warp = key_warp.view(N, t*t, dim, h, w).view(N, t, t, dim, h, w).permute(0,1,2,4,5,3)
            query_split = query.view(N, t, h*w, dim).view(N, t, 1, h, w, dim).repeat(1, 1, t, 1, 1, 1)
            query_key = torch.cat([query_split, key_warp],dim=5).flatten(3,4).flatten(2,3)
            attention_weights = self.attention_weights(query_key).view(N, t, Len_q, self.n_heads,self.n_points).permute(0,2,3,1,4)   #(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
    
class MSDeformOffsetAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.has_flow = has_flow
        if has_flow:
            self.sampling_offsets = nn.Linear(d_model+2, n_heads * n_points * 2)
            # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # attention_weights = torch.zeros((1, 1, self.n_heads, self.n_levels * self.n_points))
        # self.attention_weights = F.softmax(attention_weights, -1).view(1, 1, self.n_heads, self.n_levels, self.n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.has_flow:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, 1, self.n_points, 1)
            # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        else:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        if flow is None:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
            h = w = input_spatial_shapes[0,0]
            t = int(Len_in/(h*w))
                
            query_split = query.view(N, t, h*w, dim).view(N, t, 1, h, w, dim).repeat(1, 1, t, 1, 1, 1)
            query_offset = torch.cat([query_split, flow.permute(0,1,2,4,5,3)],dim=5)
            query_offset = query_offset.permute(0,1,3,4,2,5).flatten(2,3).flatten(1,2)

            # query_split = query.view(N, t, h*w, dim).view(N, 1, t, h, w, dim).repeat(1, t, 1, 1, 1, 1)
            # query_offset = torch.cat([query_split, flow.permute(0,1,2,4,5,3)],dim=5)
            # query_offset = query_offset.flatten(3,4).flatten(2,3).transpose(2,1)

            sampling_offsets = self.sampling_offsets(query_offset)  #b, len_q, t, head*point*2
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_levels , self.n_heads, self.n_points, 2).transpose(3,2)
            # print(sampling_offsets.shape)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # attention_weights = self.attention_weights.repeat(N, Len_q, 1,1,1).to(query.device)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output,sampling_locations

class MSDeformOffsetAttentionWeightAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.has_flow = has_flow
        if has_flow:
            self.sampling_offsets = nn.Linear(d_model*2, n_heads * n_points * 2)
            # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model*2, n_heads * n_points)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.has_flow:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, 1, self.n_points, 1)
            # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        else:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        if flow is None:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
            h = w = input_spatial_shapes[0,0]
            t = int(Len_in/(h*w))
                
            # query_split = query.view(N, t, h*w, dim).view(N, t, 1, h, w, dim).repeat(1, 1, t, 1, 1, 1)
            # query_offset = torch.cat([query_split, flow.permute(0,1,2,4,5,3)],dim=5)
            # query_offset = query_offset.permute(0,1,3,4,2,5).flatten(2,3).flatten(1,2)
            # sampling_offsets = self.sampling_offsets(query_offset)  #b, len_q, t, head*point*2
            # sampling_offsets = sampling_offsets.view(N, Len_q, self.n_levels , self.n_heads, self.n_points, 2).transpose(3,2)
            # print(sampling_offsets.shape)

            input_flatten_split = input_flatten.view(N, t, h*w, dim).view(N, 1, t, h, w, dim).repeat(1, t, 1, 1, 1, 1)
            input_flatten_split = input_flatten_split.flatten(1,2).flatten(0,1).permute(0,3,1,2)
            flow_backward = flow.transpose(2,1).flatten(1,2).flatten(0,1)
            key_warp = spynet.flow_warp(input_flatten_split, flow_backward.permute(0,2,3,1))
            key_warp = key_warp.view(N, t*t, dim, h, w).view(N, t, t, dim, h, w).permute(0,1,2,4,5,3)
            query_split = query.view(N, t, h*w, dim).view(N, t, 1, h, w, dim).repeat(1, 1, t, 1, 1, 1)
            query_key = torch.cat([query_split, key_warp],dim=5).permute(0,1,3,4,2,5).flatten(2,3).flatten(1,2)

            sampling_offsets = self.sampling_offsets(query_key)  #b, len_q, t, head*point*2
            sampling_offsets = sampling_offsets.view(N, Len_q, self.n_levels , self.n_heads, self.n_points, 2).transpose(3,2)

            attention_weights = self.attention_weights(query_key).view(N, Len_q, self.n_levels, self.n_heads,self.n_points)   #(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = attention_weights.transpose(3,2).flatten(3)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)


        # attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output

class MSDeformMultiQAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.has_flow = has_flow
        if has_flow:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
            self.attention_weights = nn.Linear(d_model*2, n_heads * n_points)
            # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.has_flow:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, 1, self.n_points, 1)
            # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        else:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)


        if not self.has_flow:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:

            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, 1, self.n_points, 2)
            sampling_offsets = sampling_offsets.repeat(1, 1, 1, self.n_levels, 1, 1)

            h = w = input_spatial_shapes[0,0]
            t = int(Len_in/(h*w))
            query_feature = query.view(N, t,h*w,dim).transpose(3,2).view(N,t,dim,h,w)   #bs, t, h, w, dim = all_frame_patch_embedding.size()
            flow_ori = matching.feature_cross_matching(query_feature)
            
            #利用光流扭曲后的特征
            key_warp = input_flatten.view(N, t, h*w, dim).view(N, 1, t, h, w, dim).repeat(1, t, 1, 1, 1, 1).flatten(1,2).flatten(0,1).permute(0,3,1,2)
            flow = flow_ori.flatten(1,2).flatten(0,1).permute(0,2,3,1)
            key_warp = spynet.flow_warp(key_warp, flow)
            key_warp = key_warp.view(N, t*t, dim, h, w).view(N, t, t, dim, h, w).permute(0,1,2,4,5,3)
            query_split = query.view(N, t, h*w, dim).view(N, t, 1, h, w, dim).repeat(1, 1, t, 1, 1, 1)
            query_key = torch.cat([query_split, key_warp],dim=5).flatten(3,4).flatten(2,3)
            attention_weights = self.attention_weights(query_key).view(N, t, Len_q, self.n_heads,self.n_points).permute(0,2,3,1,4)   #(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            reference_points = reference_points+flow_ori.permute(0,1,4,5,2,3).flatten(1,3)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output

class MSDeformQOAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.has_flow = has_flow
        if has_flow:
            self.sampling_offsets = nn.Linear(d_model*2, n_heads * n_points * 2)
            self.attention_weights = nn.Linear(d_model*2, n_heads * n_points)
            # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        else:
            self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
            self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        if self.has_flow:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, 1, self.n_points, 1)
            # grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        else:
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)


        if flow is None:
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        else:
            h = w = input_spatial_shapes[0,0]
            t = int(Len_in/(h*w))
            #利用光流扭曲后的特征
            key_warp = input_flatten.view(N, t, h*w, dim).view(N, 1, t, h, w, dim).repeat(1, t, 1, 1, 1, 1).flatten(1,2).flatten(0,1).permute(0,3,1,2)
            flow = flow.flatten(1,2).flatten(0,1).permute(0,2,3,1)
            key_warp = spynet.flow_warp(key_warp, flow)
            key_warp = key_warp.view(N, t*t, dim, h, w).view(N, t, t, dim, h, w).permute(0,1,2,4,5,3)
            #直接利用source计算attention
            # key_warp = input_flatten.view(N, t, h*w, dim).view(N, 1, t, h, w, dim).repeat(1, t, 1, 1, 1, 1)

            query_split = query.view(N, t, h*w, dim).view(N, t, 1, h, w, dim).repeat(1, 1, t, 1, 1, 1)
            query_key = torch.cat([query_split, key_warp],dim=5).flatten(3,4).flatten(2,3)   #(b, t, token_num, dim)

            sampling_offsets = self.sampling_offsets(query_key).view(N, t, Len_q, self.n_heads, self.n_points, 2).permute(0,2,3,1,4,5)
            
            attention_weights = self.attention_weights(query_key).view(N, t, Len_q, self.n_heads,self.n_points).permute(0,2,3,1,4)   #(N, Len_q, self.n_heads, self.n_levels * self.n_points)
            attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class MSDeformMatchAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # attention_weights = torch.zeros((1, 1, self.n_heads, self.n_levels * self.n_points))
        # self.attention_weights = F.softmax(attention_weights, -1).view(1, 1, self.n_heads, self.n_levels, self.n_points)
        
        self.unfold = torch.nn.Unfold(kernel_size=3,
                             dilation=1,
                             padding=1,
                             stride=1)
        
        self._reset_parameters()
        # self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    def _reset_parameters(self):
        # constant_(self.attention_weights.weight.data, 0.)
        # constant_(self.attention_weights.bias.data, 0.)
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None,pos=None):
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        h = w = input_spatial_shapes[0,0]
        t = int(Len_in/(h*w))
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # value = self.value_proj(input_flatten)
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        # value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        value = input_flatten.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # offset (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # correlation = torch.matmul(query-query.mean(dim=-1,keepdim=True), (input_flatten-input_flatten.mean(dim=-1,keepdim=True)).transpose(2,1)).view(N,Len_q, t,h*w)/(dim)
        
        query_p = query.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        input_flatten_p = input_flatten.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        q_v_proj = self.unfold(torch.cat([query_p, input_flatten_p], dim=0))   #[10, 2304, 4096]
        query_proj = q_v_proj[:N*t, ...].view(N, t, -1, h*w).transpose(3,2).flatten(1,2)
        input_flatten_proj = q_v_proj[N*t:, ...].view(N, t, -1, h*w).transpose(3,2).flatten(1,2)
        
        correlation = torch.matmul(query_proj, input_flatten_proj.transpose(2,1)).view(N,Len_q, t,h*w)/(dim)
        # est_norm = torch.nn.functional.normalize(query, dim=-1, p=2)
        # gt_norm = torch.nn.functional.normalize(input_flatten, dim=-1, p=2)
        # correlation = torch.matmul(est_norm, gt_norm.transpose(2,1)).view(N,Len_q, t,h*w)
        

        # print(correlation.shape)
        prob = F.softmax(10*correlation, dim=-1)
        sampleing_values, indices = prob.topk(self.n_points, dim=-1)
        # print(indices[0,10,0,...])
        # init_grid = geometry.coords_grid(N, h, w).to(correlation.device)  # [B, 2, H, W]
        # grid = init_grid.view(N, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

        offsets_w = torch.floor(torch.div(indices.unsqueeze(-1), h))
        offsets_h = indices.unsqueeze(-1)-offsets_w*h
        offsets = torch.cat([offsets_w,offsets_h], dim=-1).unsqueeze(2)
        # print(offsets.shape)
        # print(offsets[0,10,0,0,...])
        sampling_offsets = offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
       
        # print(sampling_offsets.shape)
        # attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        attention_weights = sampleing_values.flatten(2,3).unsqueeze(2).detach()
        # print(attention_weights.shape)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # attention_weights = self.attention_weights.repeat(N, Len_q, 1,1,1).to(query.device)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # sampling_locations = reference_points[:, :, None, :, None, :] \
            #                      + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # print(sampling_locations[0,100,0,1,...])
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output, sampling_locations

class MSDeformMatchV2Attn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        # self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        attention_weights = torch.ones((1, 1, self.n_heads, self.n_levels * self.n_points*9))
        self.attention_weights = F.softmax(attention_weights, -1).view(1, 1, self.n_heads, self.n_levels, self.n_points*9)
        
        # self.unfold = torch.nn.Unfold(kernel_size=3,
        #                      dilation=1,
        #                      padding=1,
        #                      stride=1)
        
        self._reset_parameters()
    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None,pos=None):
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        h = w = input_spatial_shapes[0,0]
        t = int(Len_in/(h*w))
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = input_flatten.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # offset (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        correlation = torch.matmul(query, input_flatten.transpose(2,1)).view(N,Len_q, t,h*w)/(dim)

        prob = F.softmax(10*correlation, dim=-1)
        sampleing_values, indices = prob.topk(self.n_points, dim=-1)
        # print(indices[0,10,0,...])
        
        indices_1 = indices[:,:,:,:]-1-h
        indices_2 = indices[:,:,:,:]-h
        indices_3 = indices[:,:,:,:]+1-h
        indices_4 = indices[:,:,:,:]-1
        indices_5 = indices[:,:,:,:]
        indices_6 = indices[:,:,:,:]+1
        indices_7 = indices[:,:,:,:]-1+h
        indices_8 = indices[:,:,:,:]+h
        indices_9 = indices[:,:,:,:]+1+h

        resample_indices = torch.cat([indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8,indices_9], dim=-1)
        max_grid = (h-1)*(w-1)
        resample_indices[resample_indices>max_grid] = max_grid
        resample_indices[resample_indices<0] = 0
        indices = resample_indices
        offsets_w = torch.floor(torch.div(indices.unsqueeze(-1), h))
        offsets_h = indices.unsqueeze(-1)-offsets_w*h
        offsets = torch.cat([offsets_w,offsets_h], dim=-1).unsqueeze(2)
        # print(offsets.shape)
        # print(offsets[0,10,0,0,...])
        sampling_offsets = offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points*9, 2)
       
        attention_weights = self.attention_weights.repeat(N, Len_q, 1,1,1).to(query.device)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # print(sampling_locations[0,100,0,1,...])
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output, sampling_locations

class MSDeformMatchV3Attn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        # self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        attention_weights = torch.ones((1, 1, self.n_heads, self.n_levels * self.n_points*9))
        self.attention_weights = F.softmax(attention_weights, -1).view(1, 1, self.n_heads, self.n_levels, self.n_points*9)
        
        self.unfold = torch.nn.Unfold(kernel_size=3,
                             dilation=1,
                             padding=1,
                             stride=1)
        self.unfold_2 = torch.nn.Unfold(kernel_size=3,
                             dilation=1,
                             padding=1,
                             stride=1)
        
        self._reset_parameters()
    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None,pos=None):
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        h = w = input_spatial_shapes[0,0]
        t = int(Len_in/(h*w))
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = input_flatten.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # offset (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        query_p = query.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        q_patch = self.unfold(query_p)   #[10, 2304, 4096]
        q_patch = q_patch.view(N, t, -1, h*w).transpose(3,2).flatten(1,2)
        
        input_flatten_p = input_flatten.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        v_patch = self.unfold_2(input_flatten_p)   #[10, 2304, 4096]
        # v_patch = v_patch.view(N, t, -1, int(h*w/4)).transpose(3,2).flatten(1,2)
        # correlation = torch.matmul(q_patch, v_patch.transpose(2,1)).view(N,Len_q, t,int(h*w/4))/(dim)
        v_patch = v_patch.view(N, t, -1, h*w).transpose(3,2).flatten(1,2)
        correlation = torch.matmul(q_patch, v_patch.transpose(2,1)).view(N,Len_q, t,h*w)/(dim)

        prob = F.softmax(10*correlation, dim=-1)
        sampleing_values, indices = prob.topk(self.n_points, dim=-1)
        # print(indices[0,10,0,...])
        # indices = indices*2
        indices_1 = indices[:,:,:,:]-1-h
        indices_2 = indices[:,:,:,:]-h
        indices_3 = indices[:,:,:,:]+1-h
        indices_4 = indices[:,:,:,:]-1
        indices_5 = indices[:,:,:,:]
        indices_6 = indices[:,:,:,:]+1
        indices_7 = indices[:,:,:,:]-1+h
        indices_8 = indices[:,:,:,:]+h
        indices_9 = indices[:,:,:,:]+1+h

        resample_indices = torch.cat([indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8,indices_9], dim=-1)
        max_grid = (h-1)*(w-1)
        resample_indices[resample_indices>max_grid] = max_grid
        resample_indices[resample_indices<0] = 0
        indices = resample_indices
        offsets_w = torch.floor(torch.div(indices.unsqueeze(-1), h))
        offsets_h = indices.unsqueeze(-1)-offsets_w*h
        offsets = torch.cat([offsets_w,offsets_h], dim=-1).unsqueeze(2)
        # print(offsets.shape)
        # print(offsets[0,10,0,0,...])
        sampling_offsets = offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points*9, 2)
       
        attention_weights = self.attention_weights.repeat(N, Len_q, 1,1,1).to(query.device)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # print(sampling_locations[0,100,0,1,...])
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output, sampling_locations

class MSDeformMatchV4Attn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        # self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # attention_weights = torch.ones((1, 1, self.n_heads, self.n_levels * self.n_points*9))
        # self.attention_weights = F.softmax(attention_weights, -1).view(1, 1, self.n_heads, self.n_levels, self.n_points*9)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points*9)
        
        self.unfold = torch.nn.Unfold(kernel_size=3,
                             dilation=1,
                             padding=1,
                             stride=1)
        self.unfold_2 = torch.nn.Unfold(kernel_size=3,
                             dilation=1,
                             padding=1,
                             stride=2)
        
        self._reset_parameters()
    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None,pos=None):
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        h = w = input_spatial_shapes[0,0]
        t = int(Len_in/(h*w))
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = input_flatten.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # offset (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        query_p = query.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        q_patch = self.unfold(query_p)   #[10, 2304, 4096]
        q_patch = q_patch.view(N, t, -1, h*w).transpose(3,2).flatten(1,2)
        
        input_flatten_p = input_flatten.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        v_patch = self.unfold_2(input_flatten_p)   #[10, 2304, 4096]
        v_patch = v_patch.view(N, t, -1, int(h*w/4)).transpose(3,2).flatten(1,2)
        correlation = torch.matmul(q_patch, v_patch.transpose(2,1)).view(N,Len_q, t,int(h*w/4))/(dim)

        prob = F.softmax(10*correlation, dim=-1)
        sampleing_values, indices = prob.topk(self.n_points, dim=-1)
        # print(indices[0,10,0,...])
        indices = indices*2
        indices_1 = indices[:,:,:,:]-1-h
        indices_2 = indices[:,:,:,:]-h
        indices_3 = indices[:,:,:,:]+1-h
        indices_4 = indices[:,:,:,:]-1
        indices_5 = indices[:,:,:,:]
        indices_6 = indices[:,:,:,:]+1
        indices_7 = indices[:,:,:,:]-1+h
        indices_8 = indices[:,:,:,:]+h
        indices_9 = indices[:,:,:,:]+1+h

        resample_indices = torch.cat([indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8,indices_9], dim=-1)
        max_grid = (h-1)*(w-1)
        resample_indices[resample_indices>max_grid] = max_grid
        resample_indices[resample_indices<0] = 0
        indices = resample_indices
        offsets_w = torch.floor(torch.div(indices.unsqueeze(-1), h))
        offsets_h = indices.unsqueeze(-1)-offsets_w*h
        offsets = torch.cat([offsets_w,offsets_h], dim=-1).unsqueeze(2)
        # print(offsets.shape)
        # print(offsets[0,10,0,0,...])
        sampling_offsets = offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points*9, 2)
       
        # attention_weights = self.attention_weights.repeat(N, Len_q, 1,1,1).to(query.device)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points*9)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points*9)
        
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # print(sampling_locations[0,100,0,1,...])
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output, sampling_locations

class MSDeformMatchV5Attn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        # self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        attention_weights = torch.ones((1, 1, self.n_heads, self.n_levels * self.n_points*9))
        self.attention_weights = F.softmax(attention_weights, -1).view(1, 1, self.n_heads, self.n_levels, self.n_points*9)
        
        self.unfold = torch.nn.Unfold(kernel_size=3,
                             dilation=1,
                             padding=1,
                             stride=1)
        self.unfold_2 = torch.nn.Unfold(kernel_size=3,
                             dilation=1,
                             padding=1,
                             stride=1)
        
        self._reset_parameters()
    def _reset_parameters(self):
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None,pos=None):
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        h = w = input_spatial_shapes[0,0]
        t = int(Len_in/(h*w))
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = input_flatten.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # offset (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        query_p = query.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        q_patch = self.unfold(query_p)   #[10, 2304, 4096]
        q_patch = q_patch.view(N, t, -1, h*w).transpose(3,2).flatten(1,2)
        
        input_flatten_p = input_flatten.view(N, t, h, w, dim).permute(0,1,4,2,3).flatten(0,1)
        v_patch = self.unfold_2(input_flatten_p)   #[10, 2304, 4096]
        # v_patch = v_patch.view(N, t, -1, int(h*w/4)).transpose(3,2).flatten(1,2)
        # correlation = torch.matmul(q_patch, v_patch.transpose(2,1)).view(N,Len_q, t,int(h*w/4))/(dim)
        v_patch = v_patch.view(N, t, -1, h*w).transpose(3,2).flatten(1,2)
        correlation = torch.matmul(q_patch, v_patch.transpose(2,1)).view(N,Len_q, t,h*w)/(dim)

        prob = F.softmax(10*correlation, dim=-1)
        sampleing_values, indices = prob.topk(self.n_points, dim=-1)
        # print(indices[0,10,0,...])
        # indices = indices*2
        indices_1 = indices[:,:,:,:]-1-h
        indices_2 = indices[:,:,:,:]-h
        indices_3 = indices[:,:,:,:]+1-h
        indices_4 = indices[:,:,:,:]-1
        indices_5 = indices[:,:,:,:]
        indices_6 = indices[:,:,:,:]+1
        indices_7 = indices[:,:,:,:]-1+h
        indices_8 = indices[:,:,:,:]+h
        indices_9 = indices[:,:,:,:]+1+h

        resample_indices = torch.cat([indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8,indices_9], dim=-1)
        max_grid = (h-1)*(w-1)
        resample_indices[resample_indices>max_grid] = max_grid
        resample_indices[resample_indices<0] = 0
        indices = resample_indices
        offsets_w = torch.floor(torch.div(indices.unsqueeze(-1), h))
        offsets_h = indices.unsqueeze(-1)-offsets_w*h
        offsets = torch.cat([offsets_w,offsets_h], dim=-1).unsqueeze(2)
        # print(offsets.shape)
        # print(offsets[0,10,0,0,...])
        sampling_offsets = offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points*9, 2)
       
        attention_weights = self.attention_weights.repeat(N, Len_q, 1,1,1).to(query.device)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # print(sampling_locations[0,100,0,1,...])
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output, sampling_locations


class MSDeformMatchV2HeaderAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, has_flow=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64  # 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        attention_weights = torch.ones((1, 1, self.n_heads, self.n_levels * self.n_points*9))
        self.attention_weights = F.softmax(attention_weights, -1).view(1, 1, self.n_heads, self.n_levels, self.n_points*9)
        
        # self.unfold = torch.nn.Unfold(kernel_size=3,
        #                      dilation=1,
        #                      padding=1,
        #                      stride=1)
        
        self._reset_parameters()
        # self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    def _reset_parameters(self):
        # constant_(self.attention_weights.weight.data, 0.)
        # constant_(self.attention_weights.bias.data, 0.)
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None, flow=None,pos=None):
        N, Len_q, _ = query.shape
        N, Len_in, dim = input_flatten.shape
        h = w = input_spatial_shapes[0,0]
        t = int(Len_in/(h*w))
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # value = self.value_proj(input_flatten)
        # if input_padding_mask is not None:
        #     value = value.masked_fill(input_padding_mask[..., None], float(0))
        # value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        value = input_flatten.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # offset (N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # correlation = torch.matmul(query, input_flatten.transpose(2,1)).view(N,Len_q, t,h*w)/(dim)
        # prob = F.softmax(10*correlation, dim=-1)
        # sampleing_values, indices = prob.topk(self.n_points, dim=-1)
        # indices_1 = indices[:,:,:,:]-1-h
        # indices_2 = indices[:,:,:,:]-h
        # indices_3 = indices[:,:,:,:]+1-h
        # indices_4 = indices[:,:,:,:]-1
        # indices_5 = indices[:,:,:,:]
        # indices_6 = indices[:,:,:,:]+1
        # indices_7 = indices[:,:,:,:]-1+h
        # indices_8 = indices[:,:,:,:]+h
        # indices_9 = indices[:,:,:,:]+1+h

        # resample_indices = torch.cat([indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8,indices_9], dim=-1)
        # max_grid = (h-1)*(w-1)
        # resample_indices[resample_indices>max_grid] = max_grid
        # resample_indices[resample_indices<0] = 0
        # indices = resample_indices
        # offsets_w = torch.floor(torch.div(indices.unsqueeze(-1), h))
        # offsets_h = indices.unsqueeze(-1)-offsets_w*h
        # offsets = torch.cat([offsets_w,offsets_h], dim=-1).unsqueeze(2)
        # sampling_offsets = offsets.view(N, Len_q, self.n_heads, self.n_levels, self.n_points*9, 2)
        
        sampling_offsets = self.sample_offset(query, input_flatten, input_spatial_shapes)
        # attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # attention_weights = sampleing_values.flatten(2,3).unsqueeze(2).detach()
        # attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        attention_weights = self.attention_weights.repeat(N, Len_q, 1,1,1).to(query.device)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # sampling_locations = reference_points[:, :, None, :, None, :] \
            #                      + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # print(sampling_locations[0,100,0,1,...])
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output, sampling_locations

    def sample_offset(self, query_ori, input_flatten_ori, input_spatial_shapes):
        N, Len_q, _ = query_ori.shape
        N, Len_in, dim = input_flatten_ori.shape
        h = w = input_spatial_shapes[0,0]
        t = int(Len_in/(h*w))
        dim_split = dim//self.n_heads
        sampling_offsets = None
        query_splits = query_ori.split(dim_split, dim=-1)   #B*(T-1)
        input_flatten_splits = input_flatten_ori.split(dim_split, dim=-1)   #B*(T-1)
        
        # query = query.view(N, Len_in, self.n_heads, self.d_model // self.n_heads).transpose(2,1).flatten(0,1)
        # input_flatten = input_flatten.view(N, Len_in, self.n_heads, self.d_model // self.n_heads).transpose(2,1).flatten(0,1)
        
        for query, input_flatten in zip(query_splits, input_flatten_splits):
            correlation = torch.matmul(query, input_flatten.transpose(2,1)).view(N,Len_q, t,h*w)/(dim_split)
            prob = F.softmax(10*correlation, dim=-1)
            sampleing_values, indices = prob.topk(self.n_points, dim=-1)
            # print(indices[0,10,0,...])
            
            indices_1 = indices[:,:,:,:]-1-h
            indices_2 = indices[:,:,:,:]-h
            indices_3 = indices[:,:,:,:]+1-h
            indices_4 = indices[:,:,:,:]-1
            indices_5 = indices[:,:,:,:]
            indices_6 = indices[:,:,:,:]+1
            indices_7 = indices[:,:,:,:]-1+h
            indices_8 = indices[:,:,:,:]+h
            indices_9 = indices[:,:,:,:]+1+h

            resample_indices = torch.cat([indices_1,indices_2,indices_3,indices_4,indices_5,indices_6,indices_7,indices_8,indices_9], dim=-1)
            max_grid = (h-1)*(w-1)
            resample_indices[resample_indices>max_grid] = max_grid
            resample_indices[resample_indices<0] = 0
            indices = resample_indices
            offsets_w = torch.floor(torch.div(indices.unsqueeze(-1), h))
            offsets_h = indices.unsqueeze(-1)-offsets_w*h
            offsets = torch.cat([offsets_w,offsets_h], dim=-1).unsqueeze(2)
            # print(offsets.shape)
            # print(offsets[0,10,0,0,...])
            sampling_offset = offsets.view(N, Len_q, 1, self.n_levels, self.n_points*9, 2)
            if sampling_offsets is None:
                sampling_offsets = sampling_offset
            else:
                sampling_offsets = torch.cat([sampling_offsets, sampling_offset], dim=2)
        return sampling_offsets
