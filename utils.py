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

from contextlib import contextmanager

from PIL import Image, ImageDraw, ImageFont 
import scipy.io.wavfile

class SimpleArgs(dict):

    
    def __getitem__(self, key):
        if key not in self: self[key] = SimpleArgs()
        return super().__getitem__(key)

    
    def __getattr__(self, key): return self[key]

    
    def __repr__(self): return f"SimpleArgs({super().__repr__()})"

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
            

@contextmanager
def evalTime(label="Operation"):

    start = time.time()
    yield
    end = time.time()
    duration = end - start

    print(""); print("")
    print(f"{label} time:", duration, "s |", duration/60.0, "m |", duration/3600, 'h')
    print(""); print("")




def compute_psnr_torch(imgs, refs, maxval=255.0):
    
    psnr = 10 * torch.log10(maxval**2/ ((imgs - refs)**2).mean(dim=-1).mean(dim=-1).mean(dim=-1))
    
    return psnr

def compute_psnr_sigs(imgs, refs):

    imgs_norm = imgs + 1.0
    refs_norm = refs + 1.0

    imgs_norm *= 0.5
    refs_norm *= 0.5

    imgs_norm *= 255.0
    refs_norm *= 255.0

    psnr_vals = compute_psnr_torch(imgs_norm, refs_norm)

    return psnr_vals

def tri_like_plus(data, rate=1/2):

    U = (torch.rand_like(data) - 0.5)*2.0
    X = torch.where(U < 0, U.abs()**(rate) - 1.0, 1.0 - U**(rate))

    return X


class ColUnfold(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(ColUnfold, self).__init__()

        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding


    def forward(self, inps): 
         
        
        outs = F.unfold(inps, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        size = round(math.sqrt(outs.shape[2]))
        outs = outs.view(outs.shape[0], outs.shape[1], size, size)

        return outs


def make_grid_plus(imgs, nrow=8, padding=1, border=1, bordercolor=0.5, normalize=True):

    imgs_pad = F.pad(imgs, (border, border, border, border), mode='constant', value=bordercolor)
    showimg = vutils.make_grid(imgs_pad.detach().cpu().float(), padding=padding,normalize=normalize, nrow = nrow)
    
    return showimg.detach().cpu().permute(1, 2, 0).numpy() 


def findClsimgs(imgs, trainimgs):
    if imgs[0].size() != trainimgs[0].size(): return torch.zeros_like(imgs)
    
    print("")
    clsimgs_all = []
    for i in range(imgs.shape[0]):
        _, idx = (imgs[i] - trainimgs).abs().reshape(trainimgs.shape[0], -1).sum(-1).sort()
        clsimgs_all.append(trainimgs[idx[0]].unsqueeze(0))
        print("findClsimgs: i = ", i, "|", imgs.shape[0], end='\r')
    
    clsimgs = torch.cat(clsimgs_all, dim=0)

    return clsimgs 


def findClsimgs_two(imgs, trainimgs):
    if imgs[0].size() != trainimgs[0].size(): return torch.zeros_like(imgs)
    
    print("")
    clsimgs_all_1 = []; clsimgs_all_2 = []

    for i in range(imgs.shape[0]):
        _, idx = (imgs[i] - trainimgs).abs().reshape(trainimgs.shape[0], -1).sum(-1).sort()
        clsimgs_all_1.append(trainimgs[idx[0]].unsqueeze(0))
        clsimgs_all_2.append(trainimgs[idx[1]].unsqueeze(0))

        print("findClsimgs: i = ", i, "|", imgs.shape[0], end='\r')
    
    clsimgs_1 = torch.cat(clsimgs_all_1, dim=0)
    clsimgs_2 = torch.cat(clsimgs_all_2, dim=0)
    

    return clsimgs_1, clsimgs_2




def findClsimgs_two_batch(imgs, trainimgs, device, batch_size=128):
    if imgs[0].size() != trainimgs[0].size(): return torch.zeros_like(imgs)
    
    print("")
    clsimgs_all_1 = []; clsimgs_all_2 = []

    for i in range(imgs.shape[0]):

        diffs = []
        for j in range(round(trainimgs.shape[0]/batch_size + 0.5 - 1e-6)):
            diffs_t = (imgs[i] - trainimgs[j*batch_size:(j+1)*batch_size].to(device)).abs().reshape(trainimgs[j*batch_size:(j+1)*batch_size].shape[0], -1).sum(-1)
            diffs.append(diffs_t.detach())
            

        diffs = torch.cat(diffs, dim=0)
        _, idx = diffs.topk(k=2, largest=False)


        clsimgs_all_1.append(trainimgs[idx[0]].unsqueeze(0))
        clsimgs_all_2.append(trainimgs[idx[1]].unsqueeze(0))

        print("findClsimgs: i = ", i, "|", imgs.shape[0], end='\r')
    
    clsimgs_1 = torch.cat(clsimgs_all_1, dim=0)
    clsimgs_2 = torch.cat(clsimgs_all_2, dim=0)
    

    return clsimgs_1, clsimgs_2



def unfold_plus(data, kernel_size=(3,3), stride=1, padding=0):

    data_unfold = F.unfold(data, kernel_size=kernel_size, stride=stride, padding=padding)
    data_unfold = data_unfold.reshape(data_unfold.shape[0], data.shape[1], kernel_size[0], kernel_size[1], data_unfold.shape[-1]).permute(0, 4, 1, 2, 3)

    ts = data_unfold.shape
    data_unfold = data_unfold.reshape(ts[0]*ts[1], ts[2], ts[3], ts[4])

    return data_unfold 




def sig2dec(data, byte=8):
    
    data = (data + 1.0)*0.5
    dims = byte - data.shape[1]
    
    data_pad = F.pad(data, (0, 0, 0, 0, dims, 0))
    
    MAT = 2**torch.tensor(range(byte)).flip(0)
    MAT = MAT.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(data.device)
    data_uint8 = (data_pad*MAT).sum(dim=1, keepdim=True) 
    
    return data_uint8 



def dec2sig(data, byte=8):
    v = data.to(torch.int64)  
    shifts = torch.arange(byte - 1, -1, -1, device=data.device, dtype=v.dtype)
    shifts = shifts.view(1, byte, 1, 1)

    bits = (v >> shifts) & 1

    return bits.to(torch.float32) * 2.0 - 1.0




def sig2dec_batch(data, batch_size=100, byte=8):

    re_data = []
    for i in range(round(data.shape[0]/batch_size + 0.5 - 1e-6)):
        temp = sig2dec(data[i*batch_size:(i+1)*batch_size], byte=byte).to(torch.uint8)
        re_data.append(temp.detach().cpu())

    re_data = torch.cat(re_data, dim=0)

    return re_data 



def dec2sig_batch(data, batch_size=100, byte=8):

    re_data = []
    for i in range(round(data.shape[0]/batch_size + 0.5 - 1e-6)):
        temp = dec2sig(data[i*batch_size:(i+1)*batch_size], byte=byte) 
        re_data.append(temp.detach().cpu().to(torch.int8))

    re_data = torch.cat(re_data, dim=0)

    return re_data 


def saveBatchImgs(imgs, pa_save, keyword='im_'):
    
    os.makedirs(os.path.dirname(pa_save), exist_ok=True)

    for i in range(imgs.shape[0]):
        filename = pa_save + keyword + str(i).zfill(6) + '.png'
        saveImg(filename, (imgs[i].float()*255.0).permute(1, 2, 0).squeeze().detach().cpu().numpy().astype(np.uint8))

        print("saving ", filename, end='\r')

    print("saveBatchImgs complete")



def showMatrixImgs(imgs, gs, mode='c'):
    
    if mode == 'c':
        for p in range(len(imgs)):
            plt.subplot(gs[p//gs.ncols, p%gs.ncols])
            plt.imshow(imgs[p])
            plt.axis('off')
    elif mode == 'r':
        for p in range(len(imgs)):
            plt.subplot(gs[p%gs.nrows,  p//gs.nrows])
            plt.imshow(imgs[p])
            plt.axis('off')
    else:
        print("model value can only be r or c")





