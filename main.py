import math
import os
import torch
import torch.nn as nn
import sys

import time
import argparse

import model as md


from utils import SimpleArgs 
from utils import make_grid_plus 

from utils import saveBatchImgs, showMatrixImgs, sig2dec_batch, dec2sig_batch, dec2sig, sig2dec, findClsimgs

from utils import compute_psnr_sigs
import numpy as np
import random

import local_auto as la

import matplotlib.pyplot as plt


import os
import torch
import torchvision
import torchvision.transforms as transforms

def main_dataset():
    output_dir = "./dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("downloading CIFAR-10 trainimg...")
    train_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    print("downloading CIFAR-10 testing...")
    test_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_images = torch.stack([img for img, _ in train_set])
    test_images = torch.stack([img for img, _ in test_set])
    
    train_path = os.path.join(output_dir, "cifar10_processed.pt")
    test_path = os.path.join(output_dir, "cifar10_processed_test.pt")
    
    torch.save(train_images, train_path)
    torch.save(test_images, test_path)



def setupSeed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main_AE(args):#argc, argv):


    batch_size = args.batch_size
    rate_eps = args.rate_eps
    num_epochs = args.num_epochs
    epochs_save = args.epochs_save
    lr = args.lr_net
    binwei = args.binwei

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


    trainimgs = torch.load('./dataset/cifar10_processed.pt', weights_only=False).to(device)[:1000]
    trainimgs_src = torch.load('./dataset/cifar10_processed.pt', weights_only=False).to(device)
    testimgs = torch.load('./dataset/cifar10_processed_test.pt', weights_only=False).to(device)


    net = md.AutoEnc().to(device)

    try:
        net.load_state_dict(torch.load('./net.pt', map_location=device))
    except:
        print("load net failed")

    optim = torch.optim.Adam( net.parameters(), lr=lr)
    net.enableClp() 
    
    if args.ifqua > 1e-3: net.enableQua()

    fig = plt.figure(figsize=(12, 4)) # row & column
    gs = plt.GridSpec(1, 3) # x & y

    datanum = trainimgs.shape[0]

    
    for epoch in range(num_epochs):
        idx_imgs = torch.randperm(datanum) 
        for i in range(round(datanum/batch_size + 0.4999999999)):

            batch_idx = idx_imgs[i*batch_size:(i+1)*batch_size]

            imgs = trainimgs[batch_idx]

            recs, bins = net(imgs, rate_eps)
            
            loss_rec = ((recs - imgs)**2).mean()
            loss_bin = (bins**2).mean()*binwei

            loss = loss_rec + loss_bin
            
            #optim.zero_grad()
            #loss.backward()
            #optim.step()
            
            avgbin = bins.abs().mean()
            binrate = md.getBinRate(bins)


        str_info = ''
        str_info += f"Epoch [{epoch + 1}/{num_epochs}]"

        
        str_info += ' loss_rec:' + ''.join([f"{t.item():.4f}," for t in [loss_rec]]) 
        str_info += ' loss_bin:' + ''.join([f"{t.item():.4f}," for t in [loss_bin]]) 


        str_info += ' avgbin:' + ''.join([f"{t.item():.4f}," for t in [avgbin]]) 
        str_info += ' binrate:' + ''.join([f"{t.item():.4f}," for t in [binrate]]) 
        
        
        str_info += ' rate_eps_:' + ''.join([f"{t:.4f}," for t in [rate_eps]])
        
        str_info += ' binwei:' + ''.join([f"{t:.4f}," for t in [binwei]])
        
        print(str_info)

        #if epoch % args.epoch_print == 0: print(str_info)

        if epoch % epochs_save == 0:
            torch.save(net.state_dict(), './net.pt')

            
            showlist = []

            showlist.append(make_grid_plus(bins.clamp(-1.0, 1.0).float()[:64, :3], nrow=8))
            showlist.append(make_grid_plus(recs.clamp(-1.0, 1.0).float()[:64], nrow=8))
            showlist.append(make_grid_plus(imgs.clamp(-1.0, 1.0).float()[:64], nrow=8))

            plt.clf(); showMatrixImgs(showlist, gs)
            plt.tight_layout(pad=1.2); plt.pause(0.1)


            psnr_train = compute_psnr_sigs(recs.clamp(-1.0, 1.0).float(), imgs.clamp(-1.0, 1.0).float())
            
            recs_valid, _ = net(trainimgs[-64:], 0.0)
            psnr_valid = compute_psnr_sigs(recs_valid.clamp(-1.0, 1.0).float(), trainimgs[-64:].float())

            recs_test, _ = net(testimgs[:64], 0.0)
            psnr_test = compute_psnr_sigs(recs_test.clamp(-1.0, 1.0).float(), testimgs[:64].float())
            
            
            with torch.no_grad():
                tempsize = 64
                allpsnr = 0
                for t in range(round(trainimgs_src.shape[0]/tempsize + 0.4999999999)):

                    recs_temp, _ = net(trainimgs_src[t*tempsize:(t+1)*tempsize], 0.0)
                    psnr_temp = compute_psnr_sigs(recs_temp.clamp(-1.0, 1.0).float(), trainimgs_src[t*tempsize:(t+1)*tempsize].float())
                    allpsnr = allpsnr + psnr_temp.sum()
                
            
            avgtrainpsnr = allpsnr/trainimgs_src.shape[0]
            

            print("")
            print("Rec PSNR train:", psnr_train.mean())
            print("Rec PSNR valid:", psnr_valid.mean())
            print("Rec PSNR test:", psnr_test.mean())
            print("Rec PSNR average train:", avgtrainpsnr)
            print("")





def main_lats(args):
    #argc, argv):

#     parser = argparse.ArgumentParser()
# 
#     parser.add_argument("--num_epochs",     type=int, default=1001)
#     parser.add_argument("--epochs_save",    type=int, default=20)
#     parser.add_argument("--ifqua",          type=int, default=0)
#     parser.add_argument("--rate_eps",       type=float, default=0.1)
#     parser.add_argument("--lr_net",         type=float, default=1e-4)
#     parser.add_argument("--batch_size",     type=int, default=1024)
#     parser.add_argument("--binwei",         type=float, default=0.1)
# 
#     args = parser.parse_args()

    batch_size = args.batch_size
    rate_eps = args.rate_eps
    num_epochs = args.num_epochs
    epochs_save = args.epochs_save
    lr = args.lr_net
    binwei = args.binwei

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


    trainimgs = torch.load('./dataset/cifar10_processed.pt', weights_only=False).to(device)

    net = md.AutoEnc().to(device)

    try:
        net.load_state_dict(torch.load('./net.pt', map_location=device))
    except:
        print("load net failed")

    
    net.enableClp()
    net.enableQua()

    recs_list = []
    lats_list = []
    with torch.no_grad():
    
        tempsize = 64
        
        for t in range(round(trainimgs.shape[0]/tempsize + 0.4999999999)):
        
            recs, lats = net(trainimgs[t*tempsize:(t+1)*tempsize], 0.0)
            recs_list.append(recs.squeeze().detach().cpu())
            lats_list.append(lats.squeeze().detach().cpu())


    recs = torch.cat(recs_list, dim=0).clamp(-1.0, 1.0)
    lats = torch.cat(lats_list, dim=0).clamp(-1.0, 1.0)

    recs = (recs + 1.0)*0.5
    saveBatchImgs(recs, './recimgs/')
    #torch.save(lats.detach().cpu(), './lats.pt')

    #lats = torch.load('./lats.pt')
    lats = torch.where(lats < 0, -1.0, 1.0)
    lats_uint8 = sig2dec_batch(lats, byte=7).to(torch.uint8)
    torch.save(lats_uint8.detach().cpu(), './lats_uint8.pt')
#    lats_uint8 = lats_uint8.to(device)


def main_gens(argc, argv):

    setupSeed(126)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainimgs = torch.load('./dataset/cifar10_processed.pt', weights_only=False).to(device)


    net = md.AutoEnc().to(device)
    net.load_state_dict(torch.load('./net.pt', map_location=device))[:10000]
    
    net.enableClp()
    net.enableQua()
   

    #lats = torch.load('./lats.pt')
    #lats = torch.where(lats < 0, -1.0, 1.0)
    #lats_uint8 = sig2dec_batch(lats, byte=7).to(torch.uint8)
    #lats_uint8 = lats_uint8.to(device)
    lats_uint8 = torch.load('./lats_uint8.pt').to(device)

    lat_size_list = [[32, 32]]
    rate_eps_list = [0.3474]
    byte_list = [7]
    padding_val = 5
    
    emprical_dis = la.createEmpricalDis(lats_uint8, rate_eps_list, byte_list, padding_val)
    
    
    genimgs_list = []
    clsimgs_list = []

    psnr_thval  = 35
    flag = 0 
    sampnum = 64

    fig = plt.figure(figsize=(12, 4)) # row & column
    gs = plt.GridSpec(1, 3) # x & y


    while flag < 1e-3:
        time_s = time.time()
        genimgs, genlats, genlats_pad = la.inferPSDRVar_T1(emprical_dis, net, sampnum=sampnum)

        clsimgs = findClsimgs(genimgs, trainimgs)
        psnr_vals = compute_psnr_sigs(genimgs, clsimgs)

        idx = psnr_vals < psnr_thval

        genimgs = genimgs[idx]
        clsimgs = clsimgs[idx]

        genimgs_list.append(genimgs.detach().cpu())
        clsimgs_list.append(clsimgs.detach().cpu())

        
        tempgen = torch.cat(genimgs_list, dim=0)
        tempcls = torch.cat(clsimgs_list, dim=0)

        if tempgen.shape[0] > 5000: flag = 1

        
        showlist = []

        showlist.append(make_grid_plus(tempgen[-64:]))
        showlist.append(make_grid_plus(tempcls[-64:]))
        showlist.append(make_grid_plus((tempgen[-64:] - tempcls[-64:]).abs()))

        
        plt.clf(); showMatrixImgs(showlist, gs) #, mode='r') # mode r 行优先显示
        plt.tight_layout(pad=1.2); plt.pause(0.1)

        time_e = time.time()
        infer_time = (time_e - time_s)

        print("infer time:", infer_time, "s |", infer_time/60.0, "m |", infer_time/3600, 'h')

        print(""); print("")
        print("genimgs num:", tempgen.shape[0])


    genimgs = torch.cat(genimgs_list, dim=0)
    genimgs = (genimgs + 1.0)*0.5
    saveBatchImgs(genimgs, './genimgs' + str(psnr_thval) + '/')



if __name__ == '__main__':
    argc = len(sys.argv)
    argv = sys.argv
    
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",        type=int, default=-1)
    parser.add_argument("--autoenc",        type=int, default=-1)
    parser.add_argument("--latent",         type=int, default=-1)
    parser.add_argument("--generation",     type=int, default=1e-3)

    parser.add_argument("--num_epochs",     type=int, default=1001)
    parser.add_argument("--epochs_save",    type=int, default=20)
    parser.add_argument("--ifqua",          type=int, default=0)
    parser.add_argument("--rate_eps",       type=float, default=0.1)
    parser.add_argument("--lr_net",         type=float, default=1e-4)
    parser.add_argument("--batch_size",     type=int, default=128)
    parser.add_argument("--binwei",         type=float, default=0.1)

    args = parser.parse_args()
    

    if args.dataset > 1e-3: main_dataset()
    if args.autoenc > 1e-3: main_AE(args)#:#argc, argv)
    if args.latent > 1e-3: main_lats(args)#argc, argv)
    if args.generation > 1e-3: main_gens(argc, argv)




