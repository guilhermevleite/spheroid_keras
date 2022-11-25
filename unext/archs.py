import torch
from torch import nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb

from typing import Tuple, Dict


__all__ = ['UNext', 'Unet', 'UnetPP', 'MultiResUNet']


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,
            num_classes,
            input_channels=3,
            deep_supervision=False,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dims=[ 128, 160, 256],
            num_heads=[1, 2, 4, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            depths=[1, 1, 1],
            sr_ratios=[8, 4, 2, 1],
            **kwargs
            ):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)


class UNext_S(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(64)
        self.dnorm4 = norm_layer(32)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(128, 64, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)
        
        self.final = nn.Conv2d(8, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
        out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        # TODO Figure out what is this doing here. I need its a reference to nn.Module, since we are inheriting that class.
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # First conv
            # Conv2d(in_cha, out_cha, kernel, stride, padding)
            # When we set stride and padding to one, it is called a SAME CONVOLUTION, the input height and width is the same after the convolution.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # There was no BachNorm at the time uNet was published, but it helps, so we are going to use it, and to do that, Conv2d 'bias' argument has to be False.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    # TODO: What the fuck is this doing? Isn't self.conv just initiated inside __init__?
    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], features=[64, 128, 256, 512], out_channels=1, **kwargs):
    # def __init__(
        # self,
        # input_channels=3,
        # img_size=224,
        # # In the paper, the out channel was 2, we are going to use 1, since all we want is a binary segmentation.
        # # TODO: Check whether out channel > 1 is necessary only when doing semantic segmentation.
        # out_channels=1,
        # # This as the features on every double convolution
        # features=[64, 128, 256, 512]
    # ):
        super(Unet, self).__init__()
        # We can not use self.downs = [], because it stores the convs and we want do do eval on these.
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downward path
        for feature in features:
            self.downs.append(DoubleConv(input_channels, feature))
            input_channels = feature
        
        # Upward path
        # TODO: For a better result we should use Transpose Convolutions
        for feature in reversed(features):
            # First append is the UP
            self.ups.append(
                nn.ConvTranspose2d(
                    # x2 because of the skip connection.
                    feature*2,
                    feature,
                    # These two will double the height, width of the image
                    kernel_size=2,
                    stride=2
                )
            )
            # Second append are the TWO CONVS
            self.ups.append(DoubleConv(feature*2, feature))
        
        # This is the horizontal path between downward and upward
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # This is the very last conv, from 388,388,64 to 388,388 or as in the paper: 388,388,2. It does not change the size of the image, it only changes the channels.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        # Simply reversing the list, because of the upward path will use it in inverse order
        skip_connections = skip_connections[::-1]
        # Step=2 because the upward path has a UP and a DoubleConv, but the skip only applies to the UP part.
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # Integer division by 2 because, altough we want to skip the DoubleConv, we also want to run through the skip_connections one by one.
            skip_connection = skip_connections[idx//2]

            ''' The INPUT needs to be shaped on a multiple of 16, since it is four down ways. If that is not the case, there will be an error to concatenate because of the MAXPOOL, since them both need same height and width.
            One work around this is to check if they are different and resize the X '''
            if x.shape != skip_connection.shape:
                # Shape has: 0 BATCH_SIZE, 1 N_CHANNELS, 2 HEIGHT, 3 WIDTH. With [2:] we are taking only height and width.
                x = TF.resize(x, size=skip_connection.shape[2:])

            # We have 4 dims, 0 BATCH, 1 CHANNEL, 2 HEIGHT, 3 WIDTH. We are concatenating them along the channel dimension
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # This will do the DoubleConv after we did the UP and concatenated the skip connection
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class UnetPP(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output




# class Multiresblock(nn.Module):
  # def __init__(self,input_features : int, corresponding_unet_filters : int ,alpha : float =1.67)->None:
    # """
        # MultiResblock

        # Arguments:
          # x - input layer
          # corresponding_unet_filters - Unet filters for the same stage
          # alpha - 1.67 - factor used in the paper to dervie number of filters for multiresunet filters from Unet filters

        # Returns - None

    # """ 
    # super().__init__()
    # self.corresponding_unet_filters = corresponding_unet_filters
    # self.alpha = alpha
    # self.W = corresponding_unet_filters * alpha
    # self.conv2d_bn_1x1 = Conv2d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5),
    # kernel_size = (1,1),activation='None',padding = 0)

    # self.conv2d_bn_3x3 = Conv2d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167),
    # kernel_size = (3,3),activation='relu',padding = 1)
    # self.conv2d_bn_5x5 = Conv2d_batchnorm(input_features=int(self.W*0.167),num_of_filters = int(self.W*0.333),
    # kernel_size = (3,3),activation='relu',padding = 1)
    # self.conv2d_bn_7x7 = Conv2d_batchnorm(input_features=int(self.W*0.333),num_of_filters = int(self.W*0.5),
    # kernel_size = (3,3),activation='relu',padding = 1)
    # self.batch_norm1 = nn.BatchNorm2d(int(self.W*0.5)+int(self.W*0.167)+int(self.W*0.333) ,affine=False)

  # def forward(self,x: torch.Tensor)->torch.Tensor:

    # temp = self.conv2d_bn_1x1(x)
    # a = self.conv2d_bn_3x3(x)
    # b = self.conv2d_bn_5x5(a)
    # c = self.conv2d_bn_7x7(b)
    # x = torch.cat([a,b,c],axis=1)
    # x = self.batch_norm1(x)
    # x += temp
    # x = self.batch_norm1(x)
    # return x

# class Conv2d_batchnorm(nn.Module):
  # def __init__(self,input_features : int,num_of_filters : int ,kernel_size : Tuple = (2,2),stride : Tuple = (1,1), activation : str = 'relu',padding  : int= 0)->None:
    # """
    # Arguments:
      # x - input layer
      # num_of_filters - no. of filter outputs
      # filters - shape of the filters to be used
      # stride - stride dimension 
      # activation -activation function to be used

    # Returns - None
    # """
    # super().__init__()
    # self.activation = activation
    # self.conv1 = nn.Conv2d(in_channels=input_features,out_channels=num_of_filters,kernel_size=kernel_size,stride=stride,padding = padding)
    # self.batchnorm = nn.BatchNorm2d(num_of_filters,affine=False)
  
  # def forward(self,x : torch.Tensor)->torch.Tensor:
    # x = self.conv1(x)
    # x = self.batchnorm(x)
    # if self.activation == 'relu':
      # return F.relu(x)
    # else:
      # return x


# class Respath(nn.Module):
  # def __init__(self,input_features : int,filters : int,respath_length : int)->None:
    # """
    # Arguments:
    # input_features - input layer filters
    # filters - output channels
    # respath_length - length of the Respath
    
    # Returns - None
    # """
    # super().__init__()
    # self.filters = filters
    # self.respath_length = respath_length
    # self.conv2d_bn_1x1 = Conv2d_batchnorm(input_features=input_features,num_of_filters = self.filters,
    # kernel_size = (1,1),activation='None',padding = 0)
    # self.conv2d_bn_3x3 = Conv2d_batchnorm(input_features=input_features,num_of_filters = self.filters,
    # kernel_size = (3,3),activation='relu',padding = 1)
    # self.conv2d_bn_1x1_common = Conv2d_batchnorm(input_features=self.filters,num_of_filters = self.filters,
    # kernel_size = (1,1),activation='None',padding = 0)
    # self.conv2d_bn_3x3_common = Conv2d_batchnorm(input_features=self.filters,num_of_filters = self.filters,
    # kernel_size = (3,3),activation='relu',padding = 1)
    # self.batch_norm1 = nn.BatchNorm2d(filters,affine=False)
    
  # def forward(self,x : torch.Tensor)->torch.Tensor:
    # shortcut = self.conv2d_bn_1x1(x)
    # x = self.conv2d_bn_3x3(x)
    # # x += shortcut    
    # x = F.relu(x)
    # x = self.batch_norm1(x)
    # if self.respath_length>1:
      # for i in range(self.respath_length):
        # shortcut = self.conv2d_bn_1x1_common(x)
        # x = self.conv2d_bn_3x3_common(x)
        # # x += shortcut
        # x = F.relu(x)
        # x = self.batch_norm1(x)
      # return x
    # else:
      # return x

# class MultiResUnet(nn.Module):
  # def __init__(self,
          # nclasses : int =1,
          # channels : int =3,
          # deep_supervision : bool =False,
          # filters : int =32
          # )->None:

    # """
    # Arguments:
    # channels - input image channels
    # filters - filters to begin with (Unet)
    # nclasses - number of classes

    # Returns - None
    # """
    # super().__init__()
    # self.alpha = 1.67
    # self.filters = filters
    # print('CREIO QUE NAO SUBSTITUI NCLASSES', nclasses, channels)
    # self.nclasses = nclasses
    # self.multiresblock1 = Multiresblock(input_features=channels,corresponding_unet_filters=self.filters)
    # self.pool1 =  nn.MaxPool2d(2,stride= 2)
    # self.in_filters1 = int(self.filters*self.alpha* 0.5)+int(self.filters*self.alpha*0.167)+int(self.filters*self.alpha*0.333)
    # self.respath1 = Respath(input_features=self.in_filters1 ,filters=self.filters,respath_length=4)
    # self.multiresblock2 = Multiresblock(input_features= self.in_filters1,corresponding_unet_filters=self.filters*2)
    # self.pool2 =  nn.MaxPool2d(2, 2)
    # self.in_filters2 = int(self.filters*2*self.alpha* 0.5)+int(self.filters*2*self.alpha*0.167)+int(self.filters*2*self.alpha*0.333)
    # self.respath2 = Respath(input_features=self.in_filters2,filters=self.filters*2,respath_length=3)
    # self.multiresblock3 = Multiresblock(input_features= self.in_filters2,corresponding_unet_filters=self.filters*4)
    # self.pool3 =  nn.MaxPool2d(2, 2)
    # self.in_filters3 = int(self.filters*4*self.alpha* 0.5)+int(self.filters*4*self.alpha*0.167)+int(self.filters*4*self.alpha*0.333)
    # self.respath3 = Respath(input_features=self.in_filters3,filters=self.filters*4,respath_length=2)
    # self.multiresblock4 = Multiresblock(input_features= self.in_filters3,corresponding_unet_filters=self.filters*8)
    # self.pool4 =  nn.MaxPool2d(2, 2)
    # self.in_filters4 = int(self.filters*8*self.alpha* 0.5)+int(self.filters*8*self.alpha*0.167)+int(self.filters*8*self.alpha*0.333)
    # self.respath4 = Respath(input_features=self.in_filters4,filters=self.filters*8,respath_length=1)
    # self.multiresblock5 = Multiresblock(input_features= self.in_filters4,corresponding_unet_filters=self.filters*16)
    # self.in_filters5 = int(self.filters*16*self.alpha* 0.5)+int(self.filters*16*self.alpha*0.167)+int(self.filters*16*self.alpha*0.333)
     
    # #Decoder path
    # self.upsample6 = nn.ConvTranspose2d(in_channels=self.in_filters5,out_channels=self.filters*8,kernel_size=(2,2),stride=(2,2),padding = 0)  
    # self.concat_filters1 = self.filters*8+self.filters*8
    # self.multiresblock6 = Multiresblock(input_features=self.concat_filters1,corresponding_unet_filters=self.filters*8)
    # self.in_filters6 = int(self.filters*8*self.alpha* 0.5)+int(self.filters*8*self.alpha*0.167)+int(self.filters*8*self.alpha*0.333)
    # self.upsample7 = nn.ConvTranspose2d(in_channels=self.in_filters6,out_channels=self.filters*4,kernel_size=(2,2),stride=(2,2),padding = 0)  
    # self.concat_filters2 = self.filters*4+self.filters*4
    # self.multiresblock7 = Multiresblock(input_features=self.concat_filters2,corresponding_unet_filters=self.filters*4)
    # self.in_filters7 = int(self.filters*4*self.alpha* 0.5)+int(self.filters*4*self.alpha*0.167)+int(self.filters*4*self.alpha*0.333)
    # self.upsample8 = nn.ConvTranspose2d(in_channels=self.in_filters7,out_channels=self.filters*2,kernel_size=(2,2),stride=(2,2),padding = 0)  
    # self.concat_filters3 = self.filters*2+self.filters*2
    # self.multiresblock8 = Multiresblock(input_features=self.concat_filters3,corresponding_unet_filters=self.filters*2)
    # self.in_filters8 = int(self.filters*2*self.alpha* 0.5)+int(self.filters*2*self.alpha*0.167)+int(self.filters*2*self.alpha*0.333)
    # self.upsample9 = nn.ConvTranspose2d(in_channels=self.in_filters8,out_channels=self.filters,kernel_size=(2,2),stride=(2,2),padding = 0)  
    # self.concat_filters4 = self.filters+self.filters
    # self.multiresblock9 = Multiresblock(input_features=self.concat_filters4,corresponding_unet_filters=self.filters)
    # self.in_filters9 = int(self.filters*self.alpha* 0.5)+int(self.filters*self.alpha*0.167)+int(self.filters*self.alpha*0.333)
    # self.conv_final = Conv2d_batchnorm(input_features=self.in_filters9,num_of_filters = self.nclasses,
    # kernel_size = (1,1),activation='None')

  # def forward(self,x : torch.Tensor)->torch.Tensor:
    # x_multires1 = self.multiresblock1(x)
    # x_pool1 = self.pool1(x_multires1)
    # x_multires1 = self.respath1(x_multires1)
    # x_multires2 = self.multiresblock2(x_pool1)
    # x_pool2 = self.pool2(x_multires2)
    # x_multires2 = self.respath2(x_multires2)
    # x_multires3 = self.multiresblock3(x_pool2)
    # x_pool3 = self.pool3(x_multires3)
    # x_multires3 = self.respath3(x_multires3)
    # x_multires4 = self.multiresblock4(x_pool3)
    # x_pool4 = self.pool4(x_multires4)
    # x_multires4 = self.respath4(x_multires4)
    # x_multires5 = self.multiresblock5(x_pool4)
    # up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
    # x_multires6 = self.multiresblock6(up6)
    # up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
    # x_multires7 = self.multiresblock7(up7)
    # up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
    # x_multires8 = self.multiresblock8(up8)
    # up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
    # x_multires9 = self.multiresblock9(up9)
    # if self.nclasses > 1:
      # conv_final_layer =  self.conv_final(x_multires9)
    # else:
      # conv_final_layer =  torch.sigmoid(self.conv_final(x_multires9))
    # return conv_final_layer





# import torch


class Conv2d_batchnorm(torch.nn.Module):
	'''
	2D Convolutional layers

	Arguments:
		num_in_filters {int} -- number of input filters
		num_out_filters {int} -- number of output filters
		kernel_size {tuple} -- size of the convolving kernel
		stride {tuple} -- stride of the convolution (default: {(1, 1)})
		activation {str} -- activation function (default: {'relu'})

	'''
	def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), activation = 'relu'):
		super().__init__()
		self.activation = activation
		self.conv1 = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=kernel_size, stride=stride, padding = 'same')
		self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
	
	def forward(self,x):
		x = self.conv1(x)
		x = self.batchnorm(x)
		
		if self.activation == 'relu':
			return torch.nn.functional.relu(x)
		else:
			return x


class Multiresblock(torch.nn.Module):
	'''
	MultiRes Block
	
	Arguments:
		num_in_channels {int} -- Number of channels coming into mutlires block
		num_filters {int} -- Number of filters in a corrsponding UNet stage
		alpha {float} -- alpha hyperparameter (default: 1.67)
	
	'''

	def __init__(self, num_in_channels, num_filters, alpha=1.67):
	
		super().__init__()
		self.alpha = alpha
		self.W = num_filters * alpha
		
		filt_cnt_3x3 = int(self.W*0.167)
		filt_cnt_5x5 = int(self.W*0.333)
		filt_cnt_7x7 = int(self.W*0.5)
		num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
		
		self.shortcut = Conv2d_batchnorm(num_in_channels ,num_out_filters , kernel_size = (1,1), activation='None')

		self.conv_3x3 = Conv2d_batchnorm(num_in_channels, filt_cnt_3x3, kernel_size = (3,3), activation='relu')

		self.conv_5x5 = Conv2d_batchnorm(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3), activation='relu')
		
		self.conv_7x7 = Conv2d_batchnorm(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3), activation='relu')

		self.batch_norm1 = torch.nn.BatchNorm2d(num_out_filters)
		self.batch_norm2 = torch.nn.BatchNorm2d(num_out_filters)

	def forward(self,x):

		shrtct = self.shortcut(x)
		
		a = self.conv_3x3(x)
		b = self.conv_5x5(a)
		c = self.conv_7x7(b)

		x = torch.cat([a,b,c],axis=1)
		x = self.batch_norm1(x)

		x = x + shrtct
		x = self.batch_norm2(x)
		x = torch.nn.functional.relu(x)
	
		return x


class Respath(torch.nn.Module):
	'''
	ResPath
	
	Arguments:
		num_in_filters {int} -- Number of filters going in the respath
		num_out_filters {int} -- Number of filters going out the respath
		respath_length {int} -- length of ResPath
		
	'''

	def __init__(self, num_in_filters, num_out_filters, respath_length):
	
		super().__init__()

		self.respath_length = respath_length
		self.shortcuts = torch.nn.ModuleList([])
		self.convs = torch.nn.ModuleList([])
		self.bns = torch.nn.ModuleList([])

		for i in range(self.respath_length):
			if(i==0):
				self.shortcuts.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (1,1), activation='None'))
				self.convs.append(Conv2d_batchnorm(num_in_filters, num_out_filters, kernel_size = (3,3),activation='relu'))

				
			else:
				self.shortcuts.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (1,1), activation='None'))
				self.convs.append(Conv2d_batchnorm(num_out_filters, num_out_filters, kernel_size = (3,3), activation='relu'))

			self.bns.append(torch.nn.BatchNorm2d(num_out_filters))
		
	
	def forward(self,x):

		for i in range(self.respath_length):

			shortcut = self.shortcuts[i](x)

			x = self.convs[i](x)
			x = self.bns[i](x)
			x = torch.nn.functional.relu(x)

			x = x + shortcut
			x = self.bns[i](x)
			x = torch.nn.functional.relu(x)

		return x


class MultiResUnet(torch.nn.Module):
	'''
	MultiResUNet
	
	Arguments:
		input_channels {int} -- number of channels in image
		num_classes {int} -- number of segmentation classes
		alpha {float} -- alpha hyperparameter (default: 1.67)
	
	Returns:
		[keras model] -- MultiResUNet model
	'''
	def __init__(self,
            num_classes,
            input_channels,
            deep_supervision=False,
            alpha=1.67):
		super().__init__()
		
		self.alpha = alpha
		
		# Encoder Path
		self.multiresblock1 = Multiresblock(input_channels,32)
		self.in_filters1 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)
		self.pool1 =  torch.nn.MaxPool2d(2)
		self.respath1 = Respath(self.in_filters1,32,respath_length=4)

		self.multiresblock2 = Multiresblock(self.in_filters1,32*2)
		self.in_filters2 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
		self.pool2 =  torch.nn.MaxPool2d(2)
		self.respath2 = Respath(self.in_filters2,32*2,respath_length=3)
	
	
		self.multiresblock3 =  Multiresblock(self.in_filters2,32*4)
		self.in_filters3 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
		self.pool3 =  torch.nn.MaxPool2d(2)
		self.respath3 = Respath(self.in_filters3,32*4,respath_length=2)
	
	
		self.multiresblock4 = Multiresblock(self.in_filters3,32*8)
		self.in_filters4 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)
		self.pool4 =  torch.nn.MaxPool2d(2)
		self.respath4 = Respath(self.in_filters4,32*8,respath_length=1)
	
	
		self.multiresblock5 = Multiresblock(self.in_filters4,32*16)
		self.in_filters5 = int(32*16*self.alpha*0.167)+int(32*16*self.alpha*0.333)+int(32*16*self.alpha* 0.5)
	 
		# Decoder path
		self.upsample6 = torch.nn.ConvTranspose2d(self.in_filters5,32*8,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters1 = 32*8 *2
		self.multiresblock6 = Multiresblock(self.concat_filters1,32*8)
		self.in_filters6 = int(32*8*self.alpha*0.167)+int(32*8*self.alpha*0.333)+int(32*8*self.alpha* 0.5)

		self.upsample7 = torch.nn.ConvTranspose2d(self.in_filters6,32*4,kernel_size=(2,2),stride=(2,2))  
		self.concat_filters2 = 32*4 *2
		self.multiresblock7 = Multiresblock(self.concat_filters2,32*4)
		self.in_filters7 = int(32*4*self.alpha*0.167)+int(32*4*self.alpha*0.333)+int(32*4*self.alpha* 0.5)
	
		self.upsample8 = torch.nn.ConvTranspose2d(self.in_filters7,32*2,kernel_size=(2,2),stride=(2,2))
		self.concat_filters3 = 32*2 *2
		self.multiresblock8 = Multiresblock(self.concat_filters3,32*2)
		self.in_filters8 = int(32*2*self.alpha*0.167)+int(32*2*self.alpha*0.333)+int(32*2*self.alpha* 0.5)
	
		self.upsample9 = torch.nn.ConvTranspose2d(self.in_filters8,32,kernel_size=(2,2),stride=(2,2))
		self.concat_filters4 = 32 *2
		self.multiresblock9 = Multiresblock(self.concat_filters4,32)
		self.in_filters9 = int(32*self.alpha*0.167)+int(32*self.alpha*0.333)+int(32*self.alpha* 0.5)

		self.conv_final = Conv2d_batchnorm(self.in_filters9, num_classes, kernel_size = (1,1), activation='None')

	def forward(self,x : torch.Tensor)->torch.Tensor:

		x_multires1 = self.multiresblock1(x)
		x_pool1 = self.pool1(x_multires1)
		x_multires1 = self.respath1(x_multires1)
		
		x_multires2 = self.multiresblock2(x_pool1)
		x_pool2 = self.pool2(x_multires2)
		x_multires2 = self.respath2(x_multires2)

		x_multires3 = self.multiresblock3(x_pool2)
		x_pool3 = self.pool3(x_multires3)
		x_multires3 = self.respath3(x_multires3)

		x_multires4 = self.multiresblock4(x_pool3)
		x_pool4 = self.pool4(x_multires4)
		x_multires4 = self.respath4(x_multires4)

		x_multires5 = self.multiresblock5(x_pool4)

		up6 = torch.cat([self.upsample6(x_multires5),x_multires4],axis=1)
		x_multires6 = self.multiresblock6(up6)

		up7 = torch.cat([self.upsample7(x_multires6),x_multires3],axis=1)
		x_multires7 = self.multiresblock7(up7)

		up8 = torch.cat([self.upsample8(x_multires7),x_multires2],axis=1)
		x_multires8 = self.multiresblock8(up8)

		up9 = torch.cat([self.upsample9(x_multires8),x_multires1],axis=1)
		x_multires9 = self.multiresblock9(up9)

		out =  self.conv_final(x_multires9)
		
		return out
