# from tracemalloc import reset_peak
# from tracemalloc import reset_peak
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.optim import lr_scheduler
from torchvision import models
from util.tools import *
from util import util
from . import base_networks as networks_init
from . import transformer
from . import gftransformer
from models.deformable import deform_transformer
# from models.deformable import st_transformer, st_transformerv2, st_transformerv4, st_transformerv5,st_transformerv6
import math
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from models.lib import spynet, spynet_mm
# from models.lib.flow import matching
# from models.lib.gmflow import gmflow, geometry
# from models.lib import swin_transformer_v2

def define_G(netG='retinex',init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    if netG == 'base_td':
        net = BaseTDGenerator(opt)
    elif netG == 's_gf_t_de':
        net = SpatialGlobalFilterTemporalDeAttGenerator(opt)
    elif netG == 's_tre_t_de':
        net = SpatialTRETemporalDeAttGenerator(opt)
    elif netG == 'swin_deformable':
        net = SwinDeformableGenerator(opt)
    elif netG == 'swin_flow_deformable':
        net = SwinFlowDeformableGenerator(opt)
    elif netG == 'swin_td_match':
        net = SwinTDMatchV2Generator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = networks_init.init_weights(net, init_type, init_gain)
    if netG == 'swin_flow_deformable':
        net.flow_net.load_gmflow()
    net = networks_init.build_model(opt, net)
    return net

class SpatialGlobalFilterTemporalDeAttGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SpatialGlobalFilterTemporalDeAttGenerator, self).__init__()
        dim = opt.embedding_dim
        dec_layers =  2
        h, w = opt.crop_size // opt.ksize, opt.crop_size // opt.ksize
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = opt.ksize, p2 = opt.ksize),
            nn.Linear(opt.ksize*opt.ksize*(opt.input_nc+1), dim)
        )
        self.spatial_tre = gftransformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act, h=h, w=w)
        self.temporal_tre = deform_transformer.DeformableTransformerEncoders(d_model=dim, nhead=opt.tr_t_trd_head, num_encoder_layers=opt.tr_t_trd_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, t_attn=opt.t_attn,window_size=opt.window_size)
        self.dec = ContentDecoder(dec_layers, 0, dim, opt.output_nc,64, 'ln', opt.activ, pad_type=opt.pad_type)
        # self.spynet = spynet.SpyNet('models/lib/spynet_sintel_final-3d2a1287.pth', no_backward=True)
        
    def forward(self, inputs=None, composite=None, spatial_pos=None, three_dim_pos=None, key_padding_mask=None):
        bs, t, c, in_h, in_w = inputs.shape
        all_frame_patch_embedding = self.patch_to_embedding(inputs)
        bs, t, h, w, dim = all_frame_patch_embedding.size()
        
        spatial_enc = self.spatial_tre(all_frame_patch_embedding.flatten(2,3).flatten(0,1), src_pos=spatial_pos)
        # spatial_enc = self.spatial_tre(all_frame_patch_embedding.flatten(2,3).flatten(0,1), src_pos=spatial_pos.flatten(2,3).transpose(2,1))
        spatial_enc = spatial_enc.view(bs*t, h, w, dim).view(bs, t, h, w, dim)

        # lqs_downsample = F.interpolate(inputs[:, :, :3, :, :].view(-1, 3, in_h, in_w), scale_factor=0.25, mode='bicubic')\
        #             .view(bs, t, 3, h, w)
        # lqs_1 = lqs_downsample.unsqueeze(2).repeat(1,1,t,1,1,1).flatten(1,2).flatten(0,1)
        # lqs_2 = lqs_downsample.unsqueeze(1).repeat(1,t,1,1,1,1).flatten(1,2).flatten(0,1)
        # flows_backward_2 = self.spynet(lqs_1, lqs_2)
        # warps = spynet_mm.flow_warp(lqs_2, flows_backward_2.permute(0,2,3,1)).view(bs, t*t, 3, h,w).view(bs, t, t, 3, h, w)
        # flows_backward_2 = flows_backward_2.view(bs, t*t, 2, h,w).view(bs, t, t, 2, h, w)  #[2, 5, 5, 2, 64, 64]
        
        # if three_dim_pos is not None:
        #     three_dim_pos = three_dim_pos.permute(0,2,3,4,1)
        temporal_frames,sampling_locations = self.temporal_tre(spatial_enc, pos_embeds=spatial_pos,flow=None)
        if sampling_locations is not None:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), temporal_frames.permute(0,1,4,2,3).detach(),sampling_locations.detach()
        else:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), None, None

class SpatialTRETemporalDeAttGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SpatialTRETemporalDeAttGenerator, self).__init__()
        dim = opt.embedding_dim
        dec_layers =  2
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = opt.ksize, p2 = opt.ksize),
            nn.Linear(opt.ksize*opt.ksize*(opt.input_nc+1), dim)
        )
        self.spatial_tre = transformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act)
        self.temporal_tre = deform_transformer.DeformableTransformerEncoders(d_model=dim, nhead=opt.tr_t_trd_head, num_encoder_layers=opt.tr_t_trd_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, t_attn=opt.t_attn,window_size=opt.window_size)
        self.dec = ContentDecoder(dec_layers, 0, dim, opt.output_nc,64, 'ln', opt.activ, pad_type=opt.pad_type)
        # self.spynet = spynet.SpyNet('models/lib/spynet_sintel_final-3d2a1287.pth', no_backward=True)
        
    def forward(self, inputs=None, composite=None, spatial_pos=None, three_dim_pos=None, key_padding_mask=None):
        bs, t, c, in_h, in_w = inputs.shape
        all_frame_patch_embedding = self.patch_to_embedding(inputs)
        bs, t, h, w, dim = all_frame_patch_embedding.size()
        
        spatial_enc = self.spatial_tre(all_frame_patch_embedding.flatten(2,3).flatten(0,1), src_pos=spatial_pos)
        # spatial_enc = self.spatial_tre(all_frame_patch_embedding.flatten(2,3).flatten(0,1), src_pos=spatial_pos.flatten(2,3).transpose(2,1))
        spatial_enc = spatial_enc.view(bs*t, h, w, dim).view(bs, t, h, w, dim)

        # lqs_downsample = F.interpolate(inputs[:, :, :3, :, :].view(-1, 3, in_h, in_w), scale_factor=0.25, mode='bicubic')\
        #             .view(bs, t, 3, h, w)
        # lqs_1 = lqs_downsample.unsqueeze(2).repeat(1,1,t,1,1,1).flatten(1,2).flatten(0,1)
        # lqs_2 = lqs_downsample.unsqueeze(1).repeat(1,t,1,1,1,1).flatten(1,2).flatten(0,1)
        # flows_backward_2 = self.spynet(lqs_1, lqs_2)
        # warps = spynet_mm.flow_warp(lqs_2, flows_backward_2.permute(0,2,3,1)).view(bs, t*t, 3, h,w).view(bs, t, t, 3, h, w)
        # flows_backward_2 = flows_backward_2.view(bs, t*t, 2, h,w).view(bs, t, t, 2, h, w)  #[2, 5, 5, 2, 64, 64]
        
        # if three_dim_pos is not None:
        #     three_dim_pos = three_dim_pos.permute(0,2,3,4,1)
        temporal_frames,sampling_locations = self.temporal_tre(spatial_enc, pos_embeds=spatial_pos,flow=None)
        if sampling_locations is not None:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), temporal_frames.permute(0,1,4,2,3).detach(),sampling_locations.detach()
        else:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), None, None

class BaseTDGenerator(nn.Module):
    def __init__(self, opt=None):
        super(BaseTDGenerator, self).__init__()
        dim = opt.embedding_dim
        dec_layers =  2
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = opt.ksize, p2 = opt.ksize),
            nn.Linear(opt.ksize*opt.ksize*(opt.input_nc+1), dim)
        )
        self.temporal_tre = st_transformer.STTransformerEncoders(d_model=dim, nhead=opt.tr_t_trd_head, num_encoder_layers=opt.tr_t_trd_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type)
        self.dec = ContentDecoder(dec_layers, 0, dim, opt.output_nc,64, 'ln', opt.activ, pad_type=opt.pad_type)
        
    def forward(self, inputs=None, composite=None, spatial_pos=None, three_dim_pos=None, key_padding_mask=None):
        bs, t, c, in_h, in_w = inputs.shape
        all_frame_patch_embedding = self.patch_to_embedding(inputs)
        bs, t, h, w, dim = all_frame_patch_embedding.size()
        
        if three_dim_pos is not None:
            three_dim_pos = three_dim_pos.permute(0,2,3,4,1)
        temporal_frames,sampling_locations = self.temporal_tre(all_frame_patch_embedding, pos_embeds=three_dim_pos,flow=None)
        location_shape = sampling_locations.size()
        sampling_locations = sampling_locations.view(bs, t, h, w, location_shape[2],location_shape[3],location_shape[4],location_shape[5])
        output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
        harmonized = self.dec(output_reshape)
        return harmonized.view(bs, t, -1, in_h, in_w), temporal_frames.permute(0,1,4,2,3),sampling_locations

class SwinDeformableGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SwinDeformableGenerator, self).__init__()
        dim = opt.embedding_dim
        dec_layers =  2
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = opt.ksize, p2 = opt.ksize),
            nn.Linear(opt.ksize*opt.ksize*(opt.input_nc+1), dim)
        )
        self.temporal_tre = deform_transformer.SwinDeformableTransformerEncoders(d_model=dim, nhead=opt.tr_t_trd_head, num_encoder_layers=opt.tr_t_trd_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, t_attn=opt.t_attn,window_size=opt.window_size)
        self.dec = ContentDecoder(dec_layers, 0, dim, opt.output_nc,64, 'ln', opt.activ, pad_type=opt.pad_type)
        
    def forward(self, inputs=None, composite=None, spatial_pos=None, three_dim_pos=None, key_padding_mask=None):
        bs, t, c, in_h, in_w = inputs.shape
        all_frame_patch_embedding = self.patch_to_embedding(inputs)
        bs, t, h, w, dim = all_frame_patch_embedding.size()
        
        temporal_frames,sampling_locations = self.temporal_tre(all_frame_patch_embedding, pos_embeds=spatial_pos,flow=None)
        if sampling_locations is not None:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), temporal_frames.permute(0,1,4,2,3).detach(),sampling_locations.detach()
        else:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), None, None

class SwinFlowDeformableGenerator(nn.Module):
    def __init__(self, opt=None):
        super(SwinFlowDeformableGenerator, self).__init__()
        dim = opt.embedding_dim
        dec_layers =  2
        self.patch_to_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t h w (p1 p2 c)', p1 = opt.ksize, p2 = opt.ksize),
            nn.Linear(opt.ksize*opt.ksize*(opt.input_nc+1), dim)
        )
        self.temporal_tre = deform_transformer.SwinDeformableTransformerEncoders(d_model=dim, nhead=opt.tr_t_trd_head, num_encoder_layers=opt.tr_t_trd_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=True, t_attn=opt.t_attn,window_size=opt.window_size)
        self.dec = ContentDecoder(dec_layers, 0, dim, opt.output_nc,64, 'ln', opt.activ, pad_type=opt.pad_type)
        
        self.flow_net = spynet_mm.SPyNet()
        # self.flow_net = gmflow.init_gmflow()
        for p in self.flow_net.parameters():
            p.requires_grad = False
    def forward(self, inputs=None, composite=None, spatial_pos=None, three_dim_pos=None, key_padding_mask=None):
        bs, t, c, in_h, in_w = inputs.shape
        all_frame_patch_embedding = self.patch_to_embedding(inputs)
        bs, t, h, w, dim = all_frame_patch_embedding.size()
        lqs_downsample = F.interpolate(composite.flatten(0,1), scale_factor=0.25, mode='bicubic')
        lqs_downsample = lqs_downsample.view(bs, t, 3, h, w)
        lqs_1 = lqs_downsample.unsqueeze(2).repeat(1,1,t,1,1,1).flatten(1,2).flatten(0,1)
        lqs_2 = lqs_downsample.unsqueeze(1).repeat(1,t,1,1,1,1).flatten(1,2).flatten(0,1)

        flows_backward_2 = self.flow_net(lqs_2, lqs_1).detach()

        # warps = spynet_mm.flow_warp(lqs_1, flows_backward_2.permute(0,2,3,1)).view(bs, t*t, 3, h,w).view(bs, t, t, 3, h, w)
        flows_backward_2 = flows_backward_2.view(bs, t*t, 2, h,w).view(bs, t, t, 2, h, w)  #[2, 5, 5, 2, 64, 64]

        if three_dim_pos is not None:
            three_dim_pos = three_dim_pos.permute(0,2,3,4,1)
        temporal_frames,sampling_locations = self.temporal_tre(all_frame_patch_embedding, pos_embeds=spatial_pos,flow=flows_backward_2)
        output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
        harmonized = self.dec(output_reshape)
        return harmonized.view(bs, t, -1, in_h, in_w), temporal_frames.permute(0,1,4,2,3).detach(),sampling_locations.detach()

class SwinTDMatchV2Generator(nn.Module):
    def __init__(self, opt=None):
        super(SwinTDMatchV2Generator, self).__init__()
        dim = opt.embedding_dim
        dec_layers =  2
        self.patch_resolution = 64
        self.patch_to_embedding = swin_transformer_v2.PatchEmbed(256,opt.ksize,4,dim)
        self.temporal_tre = st_transformerv6.STTransformerEncoders(d_model=dim, nhead=opt.tr_t_trd_head, num_encoder_layers=opt.tr_t_trd_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type, t_attn=opt.t_attn,window_size=opt.window_size)
        self.dec = ContentDecoder(dec_layers, 0, dim, opt.output_nc,64, 'ln', opt.activ, pad_type=opt.pad_type)
        
    def forward(self, inputs=None, composite=None, spatial_pos=None, three_dim_pos=None, key_padding_mask=None):
        bs, t, c, in_h, in_w = inputs.shape
        all_frame_patch_embedding = self.patch_to_embedding(inputs.flatten(0,1))  # B Ph*Pw C
        
        temporal_frames,sampling_locations = self.temporal_tre(all_frame_patch_embedding, pos_embeds=spatial_pos,flow=None)
        temporal_frames = temporal_frames.view(bs,t,self.patch_resolution**2, -1).view(bs,t,self.patch_resolution,self.patch_resolution, -1)
        if sampling_locations is not None:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), temporal_frames.permute(0,1,4,2,3).detach(),sampling_locations.detach()
        else:
            output_reshape = temporal_frames.permute(0,1,4,2,3).flatten(0,1)
            harmonized = self.dec(output_reshape)
            return harmonized.view(bs, t, -1, in_h, in_w), None, None


class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
           
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm='ln', activation=activ, pad_type=pad_type)]
        if not dim == output_dim:
            self.model += [Conv2dBlock(dim, output_dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentDecoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, output_dim, dim, norm, activ, pad_type):
        super(ContentDecoder, self).__init__()
        self.model = []
        dim = input_dim
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]

        # upsampling blocks
        for i in range(n_downsample):
            self.model += [
                nn.Upsample(scale_factor=2),
                Conv2dBlock(dim, dim // 2, 5, 1, 2, norm=norm, activation=activ, pad_type=pad_type)
            ]
            dim //= 2

        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_ori':
            self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        elif norm == 'remove_render':
            self.norm = RemoveRender(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', groupcount=16):
        super(ConvTranspose2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'adain_ori':
            self.norm = AdaptiveInstanceNorm2d_IN(norm_dim)
        elif norm == 'adain_dyna':
            self.norm = AdaptiveInstanceNorm2d_Dyna(norm_dim)
        elif norm == 'grp':
            self.norm = nn.GroupNorm(groupcount, norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias))
        else:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=padding, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
