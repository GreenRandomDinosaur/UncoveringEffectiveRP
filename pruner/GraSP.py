import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np
import time 
from .pruner_utils import *


def GraSP(model, sample_mode, abs_val, sample_ratio, dataloader):
    if abs_val: temperature = 1.
    else: temperature = 200.

    model.train()    
    weights = gather_weights(model, sample_mode)
    Hg = {}

    for batch_idx, (inputs1, targets1) in enumerate(dataloader):
        model.zero_grad()
        inputs1 = inputs1.cuda()
        targets1 = targets1.cuda()   
        
        outputs = model(inputs1)
        outputs /= temperature
        loss1 = F.nll_loss(F.log_softmax(outputs, dim=1), targets1, reduction='sum')
        grads1 = autograd.grad(loss1, weights, create_graph=False)
        grads1 = [g1.detach().clone() for g1 in grads1]
        
        model.zero_grad()
        outputs = model(inputs1)
        outputs /= temperature
        loss2 = F.nll_loss(F.log_softmax(outputs, dim=1), targets1, reduction='sum')        
        grads2 = autograd.grad(loss2, weights, create_graph=True)
        
        loss = 0.
        for g1, g2 in zip(grads1, grads2):
            # print(g2)        
            loss += (g1*g2).sum()
        loss.backward()        

        if abs_val: 
            hg1 = gather_grads(model, sample_mode, take_abs=True)
        else: 
            hg1 = gather_grads(model, sample_mode, take_abs=False)
        update_grads(Hg, hg1)        
    
        
    all_params = []
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)        
        if cond1:
            if abs_val: 
                scores = (Hg[m] * m.weight).abs()
            else: 
                scores = Hg[m] * m.weight
            scores = copy.deepcopy(scores.detach())
            buffer_keys = [key for key, buffer in m.named_buffers()]
            if 'score' in buffer_keys:
                m.score = scores
            else: 
                m.register_buffer('score', scores)
            all_params.append(scores.view(-1))
            

    all_params = torch.cat(all_params, 0)
    all_params = torch.sort(all_params, descending=True)[0]
    num_sample = int(np.floor(all_params.numel()*sample_ratio))
    threshold_val = all_params[num_sample]
    threshold_val_weak = all_params[all_params.numel()-num_sample]
    
    
    num_dict = {}
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
    
        if cond1:
            score_mask = (m.score >= threshold_val).long()
            if 'score_mask' in buffer_keys:
                m.score_mask = score_mask
            else:
                m.register_buffer('score_mask', score_mask) 
            num_dict[n] = m.score_mask.sum().item()
            
            '''
            copy_mask = score_mask.detach().clone()
            if 'cumulated_scores' in buffer_keys:
                m.cumulated_scores += copy_mask
            else: 
                m.register_buffer('cumulated_scores', copy_mask)   


            copy_score = m.score.detach().clone()
            if 'cumulated_scores_conti' in buffer_keys:
                m.cumulated_scores_conti += copy_score
            else: 
                m.register_buffer('cumulated_scores_conti', copy_score)    
            '''                
            
    return num_dict, num_sample, threshold_val.item()
    