import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils

import imageio

from torch.autograd import Variable

import sys

import math

from torch.cuda.amp import autocast, GradScaler

import torchvision

from torch.distributions.normal import Normal
import torch.distributions as dist

from torch.cuda.amp import autocast, GradScaler

torch.set_printoptions(precision=4)
torch.set_printoptions(sci_mode=False)
torch.set_printoptions(threshold=999999999999999)
torch.set_printoptions(linewidth=200)

import torch
import torch.nn.functional as F
import numpy as np
import math

from utils import tri_like_plus #, gaussian_blur, make_grid_plus, symtri_like_plus
#from utils import gaussian_kernel
from utils import sig2dec, dec2sig, unfold_plus


def getBinRate(latz, thval=0.1):

    latz_reg = latz.clamp(-1, 1)
    num_bin = (latz_reg.detach().clone().view(-1).abs() - 1).abs() < thval #1e-6
    num_bin = num_bin.sum()
    rate_bin = num_bin/latz_reg.numel()

    return rate_bin


class ResConvBlock(nn.Module):
    def __init__(self, img_dim=1, mid_dim=32, kernel_size=1, stride=1, padding=0):
        super().__init__()

        self.block = nn.Sequential(
                nn.Conv2d(img_dim, mid_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_dim, img_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                )
    
    def forward(self, inps): return self.block(inps) + inps


class ColUnfold(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(ColUnfold, self).__init__()

        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding


    def forward(self, inps): 
         
        # outs = nn.ZeroPad2d((self.padding)(inps)
        outs = F.unfold(inps, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        size = round(math.sqrt(outs.shape[2]))
        outs = outs.view(outs.shape[0], outs.shape[1], size, size)#inps.shape[2], inps.shape[3])

        return outs


class AutoEnc(nn.Module):
    def __init__(self, lat_dim=8): 
        super().__init__()

        d_model = 512
        n_block = 4
        self.enc_dn = nn.Sequential()
        self.enc_dn.append(ColUnfold(kernel_size=7, stride=1, padding=3))
        self.enc_dn.append(nn.Conv2d(147, d_model, kernel_size=1, stride=1, padding=0))
        for _ in range(n_block): self.enc_dn.append(ResConvBlock(d_model, d_model, kernel_size=1, stride=1, padding=0))
        self.enc_dn.append(nn.Conv2d(d_model, 7, kernel_size=1, stride=1, padding=0))

 
        self.dec_up = nn.Sequential()
        self.dec_up.append(ColUnfold(kernel_size=7, stride=1, padding=3))
        self.dec_up.append(nn.Conv2d(343, d_model, kernel_size=1, stride=1, padding=0))
        for _ in range(n_block): self.dec_up.append(ResConvBlock(d_model, d_model, kernel_size=1, stride=1, padding=0))
        self.dec_up.append(nn.Conv2d(d_model, 3, kernel_size=1, stride=1, padding=0))
       
        self.noise = 'gau'

        self.ifqua = False
        self.ifclp = True


    def enableQua(self): self.ifqua = True 
    def enableClp(self): self.ifclp = True
    def encoder(self, inps): return self.enc_dn(inps)
    def decoder(self, lats): return self.dec_up(lats)

    def forward(self, inps, rate_eps=0.0, slopeval=2.0):

        lats = self.encoder(inps)
        
        lats_bin = nn.Tanh()(lats) if self.ifclp else lats 


        if self.ifqua: lats_bin = torch.where(lats_bin < 0, -1.0, 1.0)

        if self.noise == 'tri': lats_eps = lats_bin + tri_like_plus(lats_bin, 1/slopeval).detach()*rate_eps 
        if self.noise == 'gau': lats_eps = lats_bin + torch.randn_like(lats_bin).detach()*rate_eps 

        if self.ifqua: lats_eps = torch.where(lats_eps < 0, -1.0, 1.0) 
       
        out = self.decoder(lats_eps)
        
        lats_out = lats_eps if self.ifqua else lats_bin       
        
        return out, lats_out




