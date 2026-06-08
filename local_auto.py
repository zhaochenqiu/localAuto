
import math
import os
import torch
import torch.nn as nn

import trace_file as tf 
from utils import saveBatchImgs, showMatrixImgs, sig2dec_batch, dec2sig_batch, dec2sig, sig2dec, findClsimgs

def logProb_gau_ndtr(sigma, mu=-1): 
    
    sigma = torch.tensor(sigma)
    z = (0 - mu)/sigma
    log_P = torch.special.log_ndtr(z)

    z = (0 + mu)/sigma
    log_P_bar = torch.special.log_ndtr(z)

    return log_P, log_P_bar


class SimpleArgs(dict):

    
    def __getitem__(self, key):
        if key not in self: self[key] = SimpleArgs()
        return super().__getitem__(key)

    
    def __getattr__(self, key): return self[key]

    
    def __repr__(self): return f"SimpleArgs({super().__repr__()})"

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def createEmpricalDis(lats_uint8, rate_eps_list=[0.3474], byte_list=[7], padding_val=7):
    val_in = 0
    val_out = val_in + padding_val


    emprical_dis = SimpleArgs()
    
    
    emprical_dis.lat_size_list_trace = [[lats_uint8.shape[2], lats_uint8.shape[3]]] 
 
    emprical_dis = updateTrace_T1(emprical_dis, val_in=val_in, val_out=val_out)
    emprical_dis.device = lats_uint8.device

    device = lats_uint8.device

    
    emprical_dis.rate_eps_list_imgs = rate_eps_list 
    
    emprical_dis.log_P_bar_list = [t.to(device) for t in [logProb_gau_ndtr(emprical_dis.rate_eps_list_imgs[t])[1] for t in range(len(emprical_dis.rate_eps_list_imgs))]]
    emprical_dis.log_P_list = [t.to(device) for t in [logProb_gau_ndtr(emprical_dis.rate_eps_list_imgs[t])[0] for t in range(len(emprical_dis.rate_eps_list_imgs))]]
    
    emprical_dis.byte_list = byte_list 

    emprical_dis.prb_type = torch.uint8
    emprical_dis.emp_obs = lats_uint8

    print("P    :", torch.cat([t.exp().squeeze().unsqueeze(0) for t in emprical_dis.log_P_list], dim=0))
    print("P_bar:", torch.cat([t.exp().squeeze().unsqueeze(0) for t in emprical_dis.log_P_bar_list], dim=0))

    return emprical_dis









def decs_pad2imgs(decs_chain_pad, net_qua_chain, args, tempnum=64):

    


    _, _, padding_vals= args.trace
    val = padding_vals
    if padding_vals > 1e-6: decs_chain = [t[:, :, val:-val, val:-val] for t in decs_chain_pad]
    else: decs_chain = decs_chain_pad
    

    byte_list = [7]

    bins_chain = dec2sig(decs_chain[0] - 2, byte=7)

    batch_size = 32 
    recs_list = []

    tempnum = decs_chain[0].shape[0]

    for i in range(round(tempnum/batch_size + 0.4999999999)):
        
        
        lat_chain_imgs = bins_chain[i*batch_size:(i+1)*batch_size] 
        recs = net_qua_chain.decoder(lat_chain_imgs)

        recs_list.append(recs)
    
    recs = torch.cat(recs_list, dim=0).clamp(-1.0, 1.0)

    return recs, decs_chain








def fillInferInfo(canv_chain_pad, samps_bias, rects, lv, pos):
    a_s, a_e, r_s, r_e, c_s, c_e = rects[lv]
    pos_a, pos_r, pos_c = pos
    inp_canv = canv_chain_pad[lv][:, a_s:a_e, r_s:r_e, c_s:c_e]
    
    inp_canv[:, pos_a, pos_r, pos_c] = samps_bias.squeeze(1)
    canv_chain_pad[lv][:, a_s:a_e, r_s:r_e, c_s:c_e] = inp_canv

    return canv_chain_pad

def createCanv_P_P_bar(canv_chain, args, byte_list):
        
    log_P_bar_list = args.log_P_bar_list
    log_P_list = args.log_P_list

    log_P_chain = [(canv_chain[t][0].unsqueeze(0) - canv_chain[t][0].unsqueeze(0))*1.0 + log_P_list[t] for t in range(len(canv_chain))] 
    log_P_bar_chain = [(canv_chain[t][0].unsqueeze(0) - canv_chain[t][0].unsqueeze(0))*1.0 + log_P_bar_list[t] for t in range(len(canv_chain))]
    
    
    dim_chain = [(canv_chain[t][0].unsqueeze(0) - canv_chain[t][0].unsqueeze(0))*1.0 + byte_list[t] for t in range(len(canv_chain))]


    return log_P_chain, log_P_bar_chain, dim_chain



def decs2canv(decs_chain, sampnum):
    canv_chain = []
    for i in range(len(decs_chain)):
        t = (decs_chain[i][0] - decs_chain[i][0]) + 2.0
        repeats = (sampnum,) + (1,) * (t.dim())
        t = t.unsqueeze(0).repeat(repeats)

        canv_chain.append(t.detach())

    return canv_chain




def cleanCanv(canv_chain):

    re_canv_chain = []
    for i in range(len(canv_chain)):
        t = canv_chain[i].clone()
        t[t>1e-10] = 1.0
        re_canv_chain.append(t)
    
    return re_canv_chain





def updateTrace_T1(args, val_in=0, val_out=7):

    MX = 20; MI = 0; 
    size_list = args.lat_size_list_trace

    n = len(size_list)
    val = val_in
    offp_list = [[val] * n for _ in range(n)]


    def getTraceFunc(): return tf.getTrace_dynamic_plus(size_list, offp_list, val=val_out)

    args.trace = getTraceFunc(); args.trace_func = getTraceFunc
    














    return args



def flattenChain(chain): return torch.cat([chain[t].reshape(chain[t].shape[0], -1) for t in range(len(chain))], -1)



def getInferPre(rects, pos, lv, canv_chain_pad, decs_chain_pad, log_P_chain_pad, log_P_bar_chain_pad, dim_chain_pad):

    pos_a, pos_r, pos_c = pos 
            

    prior_chain = []
    prior_log_P_chain = []
    prior_log_P_bar_chain = []
    prior_dim_chain = []
    for t in range(len(rects)):
        a_s, a_e, r_s, r_e, c_s, c_e = rects[t]

        prior_temp = canv_chain_pad[t][:, a_s:a_e, r_s:r_e, c_s:c_e]

        prior_log_P_temp        = log_P_chain_pad[t][:, a_s:a_e, r_s:r_e, c_s:c_e]
        prior_log_P_bar_temp    = log_P_bar_chain_pad[t][:, a_s:a_e, r_s:r_e, c_s:c_e]


        prior_dim_temp = dim_chain_pad[t][:, a_s:a_e, r_s:r_e, c_s:c_e]



        prior_chain.append(prior_temp.detach())
        
        prior_log_P_chain.append(prior_log_P_temp)
        prior_log_P_bar_chain.append(prior_log_P_bar_temp)

        prior_dim_chain.append(prior_dim_temp)


    
    
    
    condi_chain = []
    for t in range(len(rects)):
        a_s, a_e, r_s, r_e, c_s, c_e = rects[t]
        condi_temp = decs_chain_pad[t][:, a_s:a_e, r_s:r_e, c_s:c_e]
        condi_chain.append(condi_temp.detach())


    prior_vec = flattenChain(prior_chain)
    condi_vec = flattenChain(condi_chain)

    prior_log_P_vec     = flattenChain(prior_log_P_chain)
    prior_log_P_bar_vec = flattenChain(prior_log_P_bar_chain)
    prior_dim_vec       = flattenChain(prior_dim_chain)
    

    idx_msk = prior_vec[0] > 1.0 + 1e-3

    
    a_s, a_e, r_s, r_e, c_s, c_e = rects[lv]
    patch_temp = decs_chain_pad[lv][:, a_s:a_e, r_s:r_e, c_s:c_e]
    obser_temp = patch_temp[:, pos_a, pos_r, pos_c]

    
    patch_log_P = log_P_chain_pad[lv][:, a_s:a_e, r_s:r_e, c_s:c_e]
    log_P_temp = patch_log_P[:, pos_a, pos_r, pos_c]

    patch_log_P_bar = log_P_bar_chain_pad[lv][:, a_s:a_e, r_s:r_e, c_s:c_e]
    log_P_bar_temp = patch_log_P_bar[:, pos_a, pos_r, pos_c]

    patch_dim = dim_chain_pad[lv][:, a_s:a_e, r_s:r_e, c_s:c_e]
    cur_dim = patch_dim[:, pos_a, pos_r, pos_c]


    log_P_cur       = log_P_temp
    log_P_bar_cur   = log_P_bar_temp

    condi = condi_vec[:, idx_msk]
    prior = prior_vec[:, idx_msk]
    obser = obser_temp 
        
    prior_log_P     = prior_log_P_vec[:, idx_msk]
    prior_log_P_bar = prior_log_P_bar_vec[:, idx_msk]
    
    prior_dim = prior_dim_vec[:, idx_msk]

    

    return prior, obser, condi, prior_log_P, prior_log_P_bar, log_P_cur, log_P_bar_cur, prior_dim, cur_dim




def getSampDis_t1(prior_u, condi_u, obser_u, c_X, prior_log_P, prior_log_P_bar, log_P_cur, log_P_bar_cur, prior_dim, cur_dim, _BITCOUNT_LUT):

    condi_diff = None
    if condi_u.shape[-1] < 1e-4:
        condi_diff = torch.zeros_like(obser_u).squeeze(-1).unsqueeze(-1)
        log_condi_prb = torch.zeros_like(obser_u.view(-1)[0]) 
        log_condi_prb = obser_u.new_zeros(prior_u.shape[0], 1, 1)
        
        condi_diff_xor = []
    else:
        
        prior_u_e = prior_u.unsqueeze(1)
        condi_diff_xor = _BITCOUNT_LUT[(prior_u_e^condi_u).to(torch.int64)]

        log_condi_prb = condi_diff_xor*prior_log_P_bar[0] + (prior_dim - condi_diff_xor)*prior_log_P[0]
        log_condi_prb = log_condi_prb.sum(-1, keepdim=True)

    c_X_u = c_X.unsqueeze(0).to(torch.uint8)
    obser_diff_xor = _BITCOUNT_LUT[(obser_u ^ c_X_u).to(torch.int64)]
    obser_diff = obser_diff_xor 

    log_obser_prb = obser_diff*log_P_bar_cur[0] + (cur_dim - obser_diff)*log_P_cur[0]
    log_infer_prb = log_obser_prb  + log_condi_prb
    
    norm_log_infer_prb = torch.logsumexp(log_infer_prb, dim=1)
    infer_prb = torch.softmax(norm_log_infer_prb, dim=1)

    return infer_prb 






def samplingFunc(decs_chain, c_X_list, byte_list, args, decs_imgs, sampnum=64):


    trace, padding_funs, padding_vals = args.trace_func()


    decs_chain_subset = [t[:4] for t in decs_chain]
    
    
    canv_chain = decs2canv(decs_chain_subset, sampnum)
    log_P_chain, log_P_bar_chain, dim_chain = createCanv_P_P_bar(canv_chain, args, byte_list)



    canv_chain_pad = [padding_funs[i](canv_chain[i]) for i in range(len(canv_chain))]
    decs_chain_pad = [padding_funs[i](decs_chain[i]) for i in range(len(decs_chain))]
    log_P_chain_pad     = [padding_funs[i](log_P_chain[i])      for i in range(len(log_P_chain))]
    log_P_bar_chain_pad = [padding_funs[i](log_P_bar_chain[i])  for i in range(len(log_P_bar_chain))]
    dim_chain_pad = [padding_funs[i](dim_chain[i]) for i in range(len(dim_chain))]


    canv_chain_pad = cleanCanv(canv_chain_pad)


    canv_chain_pad = [t.to(args.prb_type) for t in canv_chain_pad]
    decs_chain = [t.to(args.prb_type) for t in decs_chain]
    
    time1 = 0
    time2 = 0 



    device = args.device
    _BITCOUNT_LUT = torch.tensor(    [bin(i).count('1') for i in range(256)],  dtype=torch.uint8 ).to(device)



    for step in range(len(trace)):

        lv      = trace[step]['lv']
        pos     = trace[step]['pos']
        rects   = trace[step]['rects']

        prior, obser, condi, prior_log_P, prior_log_P_bar, log_P_cur, log_P_bar_cur, prior_dim, cur_dim = getInferPre(rects, pos, lv, canv_chain_pad, decs_chain_pad, log_P_chain_pad, log_P_bar_chain_pad, dim_chain_pad) 

         
        condi_inp = condi
        obser_inp = obser.unsqueeze(-1)

        prior = (prior - 2).to(torch.uint8)

        
        c_X = c_X_list[lv]


        samping_prb_t = getSampDis_t1(prior, condi_inp, obser_inp, c_X, prior_log_P, prior_log_P_bar, log_P_cur, log_P_bar_cur, prior_dim, cur_dim, _BITCOUNT_LUT)


        samping_prb = samping_prb_t

        samps = torch.multinomial(samping_prb, 1).float()*1.0
        
        samps_bias = samps + 2.0 
         
        canv_chain_pad = fillInferInfo(canv_chain_pad, samps_bias, rects, lv, pos)


        print(step, "|", len(trace), end='\r')


    canv_recs_list, canv_decs = decs_pad2imgs(canv_chain_pad, decs_imgs, args, tempnum=sampnum) 
        

    return canv_recs_list, canv_chain_pad, canv_decs






def inferPSDRVar_T1(args=None, decs_imgs=None, sampnum=64):
    
    device = args.device
    emp_obs_chain = [args.emp_obs.to(device)]


    byte_list = args.byte_list
    c_X_list = [torch.tensor(range(2**lat_dim)).to(device) for lat_dim in byte_list]
    
        
    genimgs, canv_chain_pad, canv_decs = samplingFunc(emp_obs_chain, c_X_list, byte_list, args, decs_imgs, sampnum=sampnum)

    return genimgs, canv_decs, canv_chain_pad
        

        




