import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np


def reset_buffers(model, sample_mode):
    for m in model.modules(): 
        if isinstance(m, sample_mode): 
            delattr(m, 'score')
            delattr(m, 'score_mask')
            

def add_buffers(model, sample_mode):
    for m in model.modules(): 
        if isinstance(m, sample_mode): 
            m.register_buffer('score', m.weight.new_zeros(m.weight.size()))
            m.register_buffer('score_mask', m.weight.new_zeros(m.weight.size()))
       
    
def save_add_masks(model, sample_mode):    
    score_mask = {}
    rand_mask = {}
    min_mask = {}
    
    for n, m in model.named_modules():
        cond1 = isinstance(m, sample_mode)
        if cond1: 
            curr_score = m.score_mask.detach().clone()
            score_mask[n] = curr_score
        
            mask_num = (curr_score==1).long().sum()
            rand_idx = np.random.permutation(curr_score.numel())
            rand_mask[n] = curr_score.new_zeros([curr_score.numel()])
            rand_mask[n][rand_idx[:mask_num]] = 1
            rand_mask[n] = rand_mask[n].reshape_as(curr_score)            
            
            _, sorted_idx = m.score.view(-1).sort()
            min_mask[n] = curr_score.new_zeros([curr_score.numel()])
            min_mask[n][sorted_idx[:mask_num]]=1
            min_mask[n] = min_mask[n].reshape_as(curr_score)
    
    return score_mask, rand_mask, min_mask
             

