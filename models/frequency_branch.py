import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nc,nc,3,1,1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return x+self.block(x)

# class FreBlock(nn.Module):
#     def __init__(self, nc):
#         super(FreBlock, self).__init__()
#         self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
#         self.process1 = nn.Sequential(
#             nn.Conv2d(nc, nc, 1, 1, 0),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, 1, 1, 0))
#         self.process2 = nn.Sequential(
#             nn.Conv2d(nc, nc, 1, 1, 0),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(nc, nc, 1, 1, 0))

#     def forward(self, x):
#         _, _, H, W = x.shape
#         x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
#         mag = torch.abs(x_freq)
#         pha = torch.angle(x_freq)
#         mag = self.process1(mag)
#         pha = self.process2(pha)
#         real = mag * torch.cos(pha)
#         imag = mag * torch.sin(pha)
#         x_out = torch.complex(real, imag)
#         x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

#         return x_out+x
    
class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(nc,nc,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(nc,nc,1,1,0))

    def forward(self,x, mode):
        x = x+1e-8
        mag = torch.abs(x)
        pha = torch.angle(x)
        if mode == 'amplitude':
            mag = self.process(mag)
        elif mode == 'phase':
            pha = self.process(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out

# class ProcessBlock(nn.Module):
#     def __init__(self, in_nc, spatial = True):
#         super(ProcessBlock,self).__init__()
#         self.spatial = spatial
#         self.spatial_process = SpaBlock(in_nc) if spatial else nn.Identity()
#         self.frequency_process = FreBlock(in_nc)
#         self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0) if spatial else nn.Conv2d(in_nc,in_nc,1,1,0)

#     def forward(self, x):
#         xori = x
#         x_freq = self.frequency_process(x)
#         x_spatial = self.spatial_process(x)
#         xcat = torch.cat([x_spatial,x_freq],1)
#         x_out = self.cat(xcat) if self.spatial else self.cat(x_freq)

#         return x_out+xori
    
class ProcessBlock(nn.Module):
    def __init__(self, in_nc):
        super(ProcessBlock,self).__init__()
        self.fpre = nn.Conv2d(in_nc, in_nc, 1, 1, 0)
        self.spatial_process1 = SpaBlock(in_nc)
        self.frequency_process1 = FreBlock(in_nc)
        self.frequency_spatial = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_frequency = nn.Conv2d(in_nc,in_nc,3,1,1)
        self.spatial_process2 = SpaBlock(in_nc)
        self.frequency_process2 = FreBlock(in_nc)
        self.cat = nn.Conv2d(2*in_nc,in_nc,1,1,0)


    def forward(self, x, mode = 'amplitude'):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(self.fpre(x), norm='backward')
        x = self.spatial_process1(x)
        x_freq = self.frequency_process1(x_freq,mode=mode)+1e-8
        x = x+self.frequency_spatial(torch.abs(torch.fft.irfft2(x_freq, s=(H, W), norm='backward'))+1e-8)
        x_freq = x_freq+torch.fft.rfft2(self.spatial_frequency(x), norm='backward')
        x = self.spatial_process2(x)+1e-8
        x_freq = self.frequency_process2(x_freq,mode=mode)+1e-8
        x_freq_spatial = torch.abs(torch.fft.irfft2(x_freq, s=(H, W), norm='backward'))
        xcat = torch.cat([x,x_freq_spatial],1)
        x_out = self.cat(xcat)

        return x_out+xori

class AmplitudeNet(nn.Module):
    def __init__(self, nc):
        super(AmplitudeNet,self).__init__()
        self.conv0 = nn.Conv2d(4,nc,1,1,0)
        self.conv1 = ProcessBlock(nc)
        self.downsample1 = nn.Conv2d(nc,nc*2,stride=2,kernel_size=2,padding=0)
        self.conv2 = ProcessBlock(nc*2)
        self.downsample2 = nn.Conv2d(nc*2,nc*3,stride=2,kernel_size=2,padding=0)
        self.conv3 = ProcessBlock(nc*3)
        self.up1 = nn.ConvTranspose2d(nc*5,nc*2,1,1)
        self.conv4 = ProcessBlock(nc*2)
        self.up2 = nn.ConvTranspose2d(nc*3,nc*1,1,1)
        self.conv5 = ProcessBlock(nc)
        self.convout = nn.Conv2d(nc,3,1,1,0)
        self.convoutfinal = nn.Conv2d(3, 3, 1, 1, 0)
        self.pro = nn.Conv2d(3, 3, 1, 1, 0)

    def forward(self, x):
        x_ori = x
        x = self.conv0(x)
        x01 = self.conv1(x,mode = 'amplitude')
        x01 = torch.nan_to_num(x01, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x1 = self.downsample1(x01)
        x12 = self.conv2(x1,mode = 'amplitude')
        x12 = torch.nan_to_num(x12, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x2 = self.downsample2(x12)
        x3 = self.conv3(x2,mode = 'amplitude')
        x3 = torch.nan_to_num(x3, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x34 = self.up1(torch.cat([F.interpolate(x3,size=(x12.size()[2],x12.size()[3]),mode='bilinear'),x12],1))
        x4 = self.conv4(x34,mode = 'amplitude')
        x4 = torch.nan_to_num(x4, nan=1e-5, posinf=1e-5, neginf=1e-5)
        x4 = self.up2(torch.cat([F.interpolate(x4,size=(x01.size()[2],x01.size()[3]),mode='bilinear'),x01],1))
        x5 = self.conv5(x4,mode = 'amplitude')
        x5 = torch.nan_to_num(x5, nan=1e-5, posinf=1e-5, neginf=1e-5)
        xout = self.convout(x5)
        # --------------------------------------code change--------------------------------------
        xout = x_ori[:,:3] + xout
        xfinal = self.convoutfinal(xout)

        return xfinal
    
class AMPLoss(nn.Module):
    def __init__(self):
        super(AMPLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='ortho')
        x_mag =  torch.abs(x)
        y = torch.fft.rfft2(y, norm='ortho')
        y_mag = torch.abs(y)

        return self.cri(x_mag,y_mag)

class PhaseLoss(nn.Module):
    def __init__(self):
        super(PhaseLoss, self).__init__()
        self.cri = nn.L1Loss()

    def forward(self, x, y):
        x = torch.fft.rfft2(x, norm='ortho')
        x_pha =  torch.angle(x)
        y = torch.fft.rfft2(y, norm='ortho')
        y_pha = torch.angle(y)

        return self.cri(x_pha,y_pha)