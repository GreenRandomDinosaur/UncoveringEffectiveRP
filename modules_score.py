import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np


'''
def choose_weights(grads_dir): 
    grads1 = grads_dir.sum(dim=0) # R x Cout x Cin x H x W -> Cout x Cin x H x W
    weight_size = grads1.size()
    weight_size = torch_size_to_list(weight_size)
    
    sorted_idx = grads1.reshape(-1).sort()[1]
    idx_max = sorted_idx[-1].item()
    idx_med = sorted_idx[len(sorted_idx)//2].item()
    idx_min = sorted_idx[0].item()
    # print(idx_max)
    
    indices = {}
    indices['max'] = to_multi_dim_idx(idx_max, weight_size)
    indices['med'] = to_multi_dim_idx(idx_med, weight_size)
    indices['min'] = to_multi_dim_idx(idx_min, weight_size)
    
    # print(indices)
    return indices


def choose_random_weights(grads_dir, num_weights=30): 
    grads1 = grads_dir.sum(dim=0) # R x Cout x Cin x H x W -> Cout x Cin x H x W
    weight_size = grads1.size()
    weight_size = torch_size_to_list(weight_size)
    
    rand_idx = np.random.permutation(grads1.numel())
    rand_idx = rand_idx[:num_weights]
    
    indices = []
    for idx1 in rand_idx:
        indices.append(to_multi_dim_idx(idx1, weight_size))
    
    # print(indices)
    return indices
    
    
def torch_size_to_list(x):
    y = []
    for x1 in x: 
        y.append(x1)
    return y    
    
        
def prod_list(x):
    y = 1
    for x1 in x: 
        y = y*x1
    return y  
        
        
def to_multi_dim_idx(idx, w_size): 
    indices = []
    for i in range(1,len(w_size)): 
        curr_size = w_size[i:]
        curr_dim = prod_list(curr_size)
        curr_idx = idx // curr_dim
        indices.append(curr_idx)
        idx = idx - curr_idx * curr_dim
    indices.append(idx)
    return indices
    
    
def select_grads(grads_dir, idx):
    curr_grads = grads_dir
    
    idx_len = len(idx)
    for i in range(1, len(idx)+1):
        idx1 = curr_grads.new_tensor([idx[idx_len-i]]).int()
        curr_grads = torch.index_select(curr_grads, idx_len-i+1, idx1)
        
    curr_grads = curr_grads.reshape(-1)
    
    return curr_grads.tolist()


def get_scores(model, dataset, sample_mode, target_layer, pruner1, abs_val, args):    
    grads_dir = []
    grads_total = []
    
    idx_train = [i for i in range(len(dataset))]
    
    for r in range(args.rounds):
        # print(r)
        model.train()
        
        trainset = torch.utils.data.Subset(dataset, idx_train[r*args.batch_size:(r+1)*args.batch_size])        
        trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        _, _, _ = pruner1(model, sample_mode, abs_val, args.sample_ratio, trainloader)    
  
        
        for n, m in model.named_modules(): 
            if n == target_layer:
                grads_total.append(m.score.unsqueeze(0))
                weight = m.weight.detach()
                assert args.pruning_method == 'SNIP' or args.pruning_method == 'GraSP_abs'
                grads_dir.append((m.score/(weight.abs()+1e-5)).unsqueeze(0))
            
        reset_buffers(model, sample_mode)        


    grads_total = torch.cat(grads_total, 0)      
    grads_dir = torch.cat(grads_dir, 0)
    
    return grads_total, grads_dir


def update_score(stats1, key, val):
    # val = val.detach().clone()
    
    if key in stats1: 
        stats1[key].append(val.unsqueeze(0))
    else: 
        stats1[key] = [val.unsqueeze(0)]
'''


def reset_buffers(model, sample_mode):
    for m in model.modules(): 
        if isinstance(m, sample_mode): 
            # m.score = m.score.new_zeros(m.score.size())
            # m.score_mask = m.score_mask.new_zeros(m.score_mask.size())
            
            delattr(m, 'score')
            delattr(m, 'score_mask')
            

def add_buffers(model, sample_mode):
    for m in model.modules(): 
        if isinstance(m, sample_mode): 
            m.register_buffer('score', m.weight.new_zeros(m.weight.size()))
            m.register_buffer('score_mask', m.weight.new_zeros(m.weight.size()))
            # m.register_buffer('cumulated_scores', m.weight.new_zeros(m.weight.size()))
            # m.register_buffer('cumulated_scores_conti', m.weight.new_zeros(m.weight.size()))
                  
         
    
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
             

