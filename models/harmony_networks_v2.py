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
from . import aptransformer
from models.deformable import st_transformerv6
import math
from models.lib import swin_transformer_v2, swin3d_transformer_v17

def define_G(netG='retinex',init_type='normal', init_gain=0.02, opt=None):
    """Create a generator
    """
    if netG == 's_apswin_t_sgt_f3Dfourier':
        net = APSwinSGTF3DFourierGenerator(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    net = networks_init.init_weights(net, init_type, init_gain)
    if netG == 'swin_flow_deformable':
        net.flow_net.load_gmflow()
    net = networks_init.build_model(opt, net)
    return net

class APSwinSGTF3DFourierGenerator(nn.Module):
    def __init__(self, opt=None):
        super(APSwinSGTF3DFourierGenerator, self).__init__()
        dim = opt.embedding_dim
        dec_layers =  2
        self.patch_resolution = 64
        h, w = opt.crop_size // opt.ksize, opt.crop_size // opt.ksize
        self.patch_to_embedding = swin_transformer_v2.PatchEmbed(256,opt.ksize,4,dim)
        
        self.FrequencyNet_1 = aptransformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act, h=h, w=w)
        self.SpatialNet_1 = st_transformerv6.STTransformerEncoders(d_model=dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type, t_attn=opt.t_attn,window_size=opt.window_size)
        
        self.FrequencyNet_2 = aptransformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act, h=h, w=w)
        self.SpatialNet_2 = st_transformerv6.STTransformerEncoders(d_model=dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type, t_attn=opt.t_attn,window_size=opt.window_size)
        
        self.FrequencyNet_3 = aptransformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act, h=h, w=w)
        self.SpatialNet_3 = st_transformerv6.STTransformerEncoders(d_model=dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type, t_attn=opt.t_attn,window_size=opt.window_size)
        
        self.FrequencyNet_4 = aptransformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act, h=h, w=w)
        self.SpatialNet_4 = st_transformerv6.STTransformerEncoders(d_model=dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type, t_attn=opt.t_attn,window_size=opt.window_size)
        
        self.FrequencyNet_5 = aptransformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act, h=h, w=w)
        self.SpatialNet_5 = st_transformerv6.STTransformerEncoders(d_model=dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type, t_attn=opt.t_attn,window_size=opt.window_size)
        
        self.FrequencyNet_6 = aptransformer.TransformerEncoders(dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, activation=opt.tr_act, h=h, w=w)
        self.SpatialNet_6 = st_transformerv6.STTransformerEncoders(d_model=dim, nhead=opt.tr_s_tre_head, num_encoder_layers=opt.tr_s_tre_layers, dim_feedforward=dim*opt.dim_forward, dropout=0.1, \
                 activation=opt.tr_act, n_levels=opt.n_frames, enc_n_points=opt.tr_t_trd_points, has_flow=False, s_t_type=opt.s_t_type, t_attn=opt.t_attn,window_size=opt.window_size)
        
        self.temporal_s_gt = nn.ModuleList()
        self.temporal_f_gt = nn.ModuleList()

        for i_layer in range(3):
            layer_s_gt = nn.ModuleList([
                swin3d_transformer_v17.TemporalRenderCrossMAEShiftPostLN(dim=dim,window_size=(1,16,16),mlp_ratio=2.0, drop_path=0.1, num=i)
                for i in range(2)
            ])
            self.temporal_s_gt.append(layer_s_gt)

            layer_f_gt = nn.ModuleList([
                swin3d_transformer_v17.Temporal3DFourierPostLN(train=opt.isTrain, dim=dim,window_size=(1,16,16),mlp_ratio=2.0, drop_path=0.1, num=i)
                for i in range(2)
            ])
            self.temporal_f_gt.append(layer_f_gt)
        
        self.dec = ContentDecoder(dec_layers, 0, dim, opt.output_nc,64, 'ln', opt.activ, pad_type=opt.pad_type)
        
    def forward(self, inputs=None, composite=None, spatial_pos=None, three_dim_pos=None, key_padding_mask=None):
        bs, t, c, in_h, in_w = inputs.shape
        all_frame_patch_embedding = self.patch_to_embedding(inputs.flatten(0,1))  # B Ph*Pw C
        
        # layer_1
        frequency_enc = self.FrequencyNet_1(all_frame_patch_embedding, src_pos=spatial_pos)
        spatial_enc, _ = self.SpatialNet_1(all_frame_patch_embedding, pos_embeds=spatial_pos,flow=None)
        fusion = frequency_enc + spatial_enc
        # layer_2
        frequency_enc = self.FrequencyNet_2(fusion, src_pos=spatial_pos)
        spatial_enc, _ = self.SpatialNet_2(fusion, pos_embeds=spatial_pos,flow=None)
        fusion = frequency_enc + spatial_enc
        # layer_3
        frequency_enc = self.FrequencyNet_3(fusion, src_pos=spatial_pos)
        spatial_enc, _ = self.SpatialNet_3(fusion, pos_embeds=spatial_pos,flow=None)
        fusion = frequency_enc + spatial_enc
        # layer_4
        frequency_enc = self.FrequencyNet_4(fusion, src_pos=spatial_pos)
        spatial_enc, _ = self.SpatialNet_4(fusion, pos_embeds=spatial_pos,flow=None)
        fusion = frequency_enc + spatial_enc
        # layer_5
        frequency_enc = self.FrequencyNet_5(fusion, src_pos=spatial_pos)
        spatial_enc, _ = self.SpatialNet_5(fusion, pos_embeds=spatial_pos,flow=None)
        fusion = frequency_enc + spatial_enc
        # layer_6
        frequency_enc = self.FrequencyNet_6(fusion, src_pos=spatial_pos)
        spatial_enc, _ = self.SpatialNet_6(fusion, pos_embeds=spatial_pos,flow=None)
        fusion = frequency_enc + spatial_enc

        fusion = fusion.view(bs,t,self.patch_resolution**2, -1).view(bs,t,self.patch_resolution,self.patch_resolution, -1)


        for s_gt,f_gt in zip(self.temporal_s_gt, self.temporal_f_gt):
            temporal_s_gt, temporal_f_gt = fusion, fusion
            for i_layer_s_gt, i_layer_f_gt in zip(s_gt, f_gt):
                temporal_s_gt = i_layer_s_gt(temporal_s_gt, gpt_mask=(None,None))
                temporal_f_gt = i_layer_f_gt(temporal_f_gt, gpt_mask=(None,None))
            fusion = temporal_s_gt + temporal_f_gt

        output_reshape = fusion.permute(0,1,4,2,3).flatten(0,1)
        harmonized = self.dec(output_reshape)
        return harmonized.view(bs, t, -1, in_h, in_w)
        



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
        # model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out_ = out+residual
        return out_

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
