import torch
import os
import itertools
import torch.nn.functional as F
from .base_model import BaseModel
from util import util
from . import harmony_networks_v2 as networks
from . import base_networks as networks_init
from einops.layers.torch import Rearrange
from util import flops

from .frequency_branch import AMPLoss, PhaseLoss

class FSSingleOutModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='TemporalTre', dataset_mode='hyt')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_T', type=float, default=500.0, help='weight for L1 loss')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--lambda_P', type=float, default=10.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_L1']
        if opt.loss_AP:
            self.loss_names = self.loss_names + ['G_A','G_P']
        if opt.loss_T:
            self.loss_names = self.loss_names + ['G_T']

        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['mask', 'harmonized','comp','real']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G'] 
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        
        if self.isTrain:
            if self.opt.t_attn != 'none':
                self.visual_names = ['mask', 'harmonized','comp','real']
            
            util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionLa = AMPLoss().to(self.opt.device)
            self.criterionLp = PhaseLoss().to(self.opt.device)
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
        self.print_flops = True

    def set_position(self, pos, patch_pos=None):
        b = self.opt.batch_size
        if self.opt.pos_none:
            self.spatial_pos = None
            self.three_dim_pos = None
        else:
            spatial_pos = self.SpatialPatchPositionEmbeddingSine(self.opt)    # [256, 64, 64]
            self.spatial_pos = spatial_pos.unsqueeze(0).flatten(2).transpose(2,1).to(self.device)  # [1, 4096, 256]
            # -----------------------------------------------------------
            spatial_pos_v2 = self.SpatialPatchPositionEmbeddingSineV2(self.opt)    # [256, 64, 64]
            self.spatial_pos_v2 = spatial_pos_v2.unsqueeze(0).flatten(2).transpose(2,1).to(self.device)  # [1, 4096, 256]
            # -----------------------------------------------------------
            three_dim_pos = self.ThreeDPatchPositionEmbeddingSine(self.opt)
            self.three_dim_pos = three_dim_pos.unsqueeze(0).repeat(b, 1, 1, 1,1).to(self.device) 
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.comp = input['comp'].to(self.device)  # [b,t,c,w,h]
        self.real = input['real'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.inputs = torch.cat([self.comp, self.mask], dim=2)
        self.video_object_paths = input['video_object_path']
        self.harmonized = None
        self.revert_mask = 1-self.mask

    def data_dependent_initialize(self, data):
        pass
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.harmonized = self.netG(self.inputs, composite=self.comp, spatial_pos=self.spatial_pos, three_dim_pos=self.three_dim_pos)
        else:
            self.harmonized = self.netG(self.test_inputs, composite=self.test_comp, spatial_pos=self.spatial_pos, three_dim_pos=self.three_dim_pos)
            

    def compute_G_loss(self):
        """Calculate L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        if self.opt.loss_T:
            self.loss_G_T = self.criterionL2(self.harmonized[:,1:,:,:,:]-self.harmonized[:,:-1,:,:,:], self.real[:,1:,:,:,:]-self.real[:,:-1,:,:,:])*self.opt.lambda_T
            self.loss_G = self.loss_G + self.loss_G_T
        if self.opt.loss_AP:
            self.loss_G_A = self.criterionLa(self.harmonized, self.real)*self.opt.lambda_A
            self.loss_G_P = self.criterionLp(self.harmonized, self.real)*self.opt.lambda_P
            self.loss_G = self.loss_G +self.loss_G_A+self.loss_G_P
        return self.loss_G

    def optimize_parameters(self):
        self.forward()
        # update G
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def new_test(self):
        with torch.no_grad():
            outputs = []
            self.test_inputs = None
            
            self.test_inputs = self.inputs
            self.test_comp = self.comp
            self.forward()
            self.harmonized = self.comp*self.revert_mask + self.harmonized*self.mask
    
    def gradient_loss(self, input_1, input_2):
        b,t,c,h,w = input_1.size()
        harmonized_g_x = util.gradient(input_1.flatten(0,1), 'x').view(b,t,1,h,w)
        harmonized_g_y = util.gradient(input_1.flatten(0,1), 'y').view(b,t,1,h,w)
        real_g_x = util.gradient(input_2.flatten(0,1), 'x').view(b,t,1,h,w)
        real_g_y = util.gradient(input_2.flatten(0,1), 'y').view(b,t,1,h,w)
        g_x = self.criterionL1(harmonized_g_x[:,1:,...]-harmonized_g_x[:,:-1,...], real_g_x[:,1:,...]-real_g_x[:,:-1,...])
        g_y = self.criterionL1(harmonized_g_y[:,1:,...]-harmonized_g_y[:,:-1,...], real_g_y[:,1:,...]-real_g_y[:,:-1,...])
        return g_x+g_y

    def PatchPositionEmbeddingSine(self, opt):
        temperature=10000
        feature_z = 20
        num_pos_feats = 256*4
        mask = torch.ones(feature_z)
        z_embed = mask.cumsum(0, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (torch.div(dim_t,2,rounding_mode='trunc')) / num_pos_feats)

        pos_z = z_embed[:, None] / dim_t
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)
        return pos_z

    def SpatialPatchPositionEmbeddingSine(self, opt):
        temperature=10000
        if opt.stride == 1:
            feature_h = int(256/opt.ksize)
        else:
            feature_h = int((256-opt.ksize)/opt.stride)+1
        num_pos_feats = opt.embedding_dim//2
        mask = torch.ones((feature_h, feature_h))
        y_embed = mask.cumsum(0, dtype=torch.float32)
        x_embed = mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (torch.div(dim_t,2,rounding_mode='trunc')) / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos

    def SpatialPatchPositionEmbeddingSineV2(self, opt):
        temperature=10000
        if opt.stride == 1:
            feature_h = int(256/opt.ksize)
        else:
            feature_h = int((256-opt.ksize)/opt.stride)+1
        num_pos_feats = opt.embedding_dim//2//2
        mask = torch.ones((feature_h, feature_h))
        y_embed = mask.cumsum(0, dtype=torch.float32)
        x_embed = mask.cumsum(1, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (torch.div(dim_t,2,rounding_mode='trunc')) / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        return pos

    def ThreeDPatchPositionEmbeddingSine(self, opt):
        temperature=10000
        if opt.stride == 1:
            feature_h = int(256/opt.ksize)
        else:
            feature_h = int((256-opt.ksize)/opt.stride)+1
        feature_t = int(opt.n_frames/opt.tsize)
        num_pos_feats = (opt.embedding_dim-32)//2
        mask = torch.ones((feature_t, feature_h, feature_h))
        t_embed = mask.cumsum(0, dtype=torch.float32)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (torch.div(dim_t,2,rounding_mode='trunc')) / num_pos_feats)
        dim_t_1 = torch.arange(32, dtype=torch.float32)
        dim_t_1 = temperature ** (2 * (torch.div(dim_t_1,2,rounding_mode='trunc')) / 32)
        pos_t = t_embed[:, :, :, None] / dim_t_1
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_t = torch.stack((pos_t[:, :, :, 0::2].sin(), pos_t[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_t, pos_y, pos_x), dim=3).permute(3, 0, 1, 2)
        return pos