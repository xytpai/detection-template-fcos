import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layers import *
import math


class FCOSHead(nn.Module):
    def __init__(self, channels, num_class, norm=4):
        super(FCOSHead, self).__init__()
        self.channels = channels
        self.num_class = num_class
        self.norm = norm
        self.conv_cls = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_class, kernel_size=3, padding=1))
        self.conv_reg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), 
            nn.GroupNorm(32, channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 4 + 1, kernel_size=3, padding=1))
        for block in [self.conv_cls, self.conv_reg]:
            for layer in block.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.constant_(layer.bias, 0)
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
        pi = 0.01
        _bias = -math.log((1.0-pi)/pi)
        nn.init.constant_(self.conv_cls[-1].bias, _bias)
    
    def forward(self, x, im_h, im_w):
        '''
        x: F(b, c, h_s, w_s)

        Return: 
        cls_s: F(b, h_s*w_s, num_class)
        reg_s: F(b, h_s*w_s, 4) ymin, xmin, ymax, xmax
        ctr_s: F(b, h_s*w_s)
        '''
        batch_size, c, h_s, w_s = x.shape
        stride = (im_h-1) // (h_s-1)
        cls_s = self.conv_cls(x)
        reg_s, ctr_s = self.conv_reg(x).split([4, 1], dim=1)
        cls_s = cls_s.permute(0,2,3,1).contiguous()
        reg_s = reg_s.permute(0,2,3,1).contiguous()*(self.norm*stride)
        ctr_s = ctr_s.permute(0,2,3,1).contiguous().sigmoid()
        cls_s = cls_s.view(batch_size, -1, self.num_class)
        reg_s = reg_s.view(batch_size, -1, 4)
        ctr_s = ctr_s.view(batch_size, -1)
        stride_s = torch.FloatTensor([stride]).to(cls_s.device)
        center_yx = center_scatter(stride_s, batch_size, h_s, w_s, stride)[..., :2]
        reg_s_ymin_xmin =  center_yx + reg_s[..., 0:2]
        reg_s_ymax_xmax =  center_yx + reg_s[..., 2:4]
        reg_s = torch.cat([reg_s_ymin_xmin, reg_s_ymax_xmax], dim=2)
        return cls_s, reg_s, ctr_s
