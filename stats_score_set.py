# ========== Thanks https://github.com/Eric-mingjie/rethinking-network-pruning ============
# ========== we adopt the code from the above link and did modifications ============
# ========== the comments as #=== === were added by us, while the comments as # were the original one ============

from __future__ import print_function

import argparse
import math
import os
import random
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import models as models

from data_piles import prepare_dataset
from pruner.SNIP import SNIP
from pruner.GraSP import GraSP
from modules_score import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', default='stats/score_hist_set', type=str)
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--tinyimagenet_dir', default='/workspace/data/tiny-imagenet-200', type=str)
parser.add_argument('--batch_size', default=128, type=int, metavar='N')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                    choices=model_names)
parser.add_argument('--depth', default=20, type=int, metavar='N',
                    help='resnet depth')
                    
parser.add_argument('--pruning_method', default='SNIP', type=str)
parser.add_argument('--pruneset_ratio', default=0.1, type=float, metavar='N') 
parser.add_argument('--sample_ratio', default=0.05, type=float, metavar='N',
                    help='1-prune_ratio')
parser.add_argument('--rounds', default=30, type=int, metavar='N',
                    help='number of total rounds to run')    
parser.add_argument('--ckpt', default=None, type=str)                    
args = parser.parse_args()


def main():
    if args.dataset=='tinyimagenet':
        dataset, num_classes = prepare_dataset(args, train_set=True, tiny_dir=args.tinyimagenet_dir)
    else: 
        dataset, num_classes = prepare_dataset(args, train_set=True) 
    
    idx = np.random.permutation(len(dataset))
    pruneset_size = int(args.pruneset_ratio*len(dataset))
    trainset_size = args.rounds * args.batch_size
    idx_prune = idx[:pruneset_size]
    idx_train = idx[-trainset_size:]  

    pruneset = torch.utils.data.Subset(dataset, idx_prune)
    trainset = torch.utils.data.Subset(dataset, idx_train)

    
    if 'resnet2x' in args.arch:
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth
                )  
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)
    model.cuda()
    cudnn.benchmark = True
    
    sample_mode = (nn.Conv2d, nn.Linear)
    
    if args.pruning_method == 'SNIP': pruner1 = SNIP
    else: pruner1 = GraSP
    
    abs_val = False
    if args.pruning_method == 'GraSP_abs':
        abs_val = True
        
        
    model.train()
    if args.ckpt is None:
        pruneloader = data.DataLoader(pruneset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        _, _, _ = pruner1(model, sample_mode, abs_val, args.sample_ratio, pruneloader)
        score_mask, rand_mask, min_mask = save_add_masks(model, sample_mode)
        reset_buffers(model, sample_mode)        
    else: 
        ckpt = torch.load(args.ckpt)
        model.load_state_dict(ckpt, strict=False)
        score_mask, rand_mask, min_mask = get_masks(model, sample_mode, ckpt)
        add_buffers(model, sample_mode)
            
    cnts1, cnts_total = 0, 0
    cnts_fc = 0    
    for k, v in score_mask.items(): 
        print('{}: {}/{}'.format(k, v.sum().item(), v.numel()))
        cnts1 += v.sum().item()
        cnts_total += v.numel()
        cnts_fc = v.sum().item()
    print('\ntotal: {:.2f}\n'.format(cnts1/cnts_total))    
   
   
    idx_train = [i for i in range(len(trainset))]   
    stats_sample_mode = (nn.Conv2d, nn.Linear)
    
    scores_impt, scores_rand = [], []
 
    
    for r in range(args.rounds):  
        model.train()
        
        batch1 = torch.utils.data.Subset(trainset, idx_train[r*args.batch_size:(r+1)*args.batch_size])
        batch1_loader = data.DataLoader(batch1, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        _, _, _ = pruner1(model, sample_mode, abs_val, args.sample_ratio, batch1_loader)
        
        scores_impt1 = collect_scores(model, score_mask, stats_sample_mode)
        scores_rand1 = collect_scores(model, rand_mask, stats_sample_mode)
        
        scores_impt += subsample_data(scores_impt1)
        scores_rand += subsample_data(scores_rand1)        
        
        reset_buffers(model, sample_mode)

        print('round: {}'.format(r))

    stats_total = {}
    stats_total['max_set'] = scores_impt
    stats_total['rand_set'] = scores_rand

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    arch = args.arch
    if 'resnet2x' in args.arch: arch = '{}_{}'.format(args.arch, args.depth)
    sio.savemat('{}/{}_{}_{}_total.mat'.format(save_dir, args.dataset, arch, args.pruning_method), stats_total)
    print('{}_{}'.format(args.dataset, arch))
    

def collect_scores(model, mask, stats_sample_mode): 
    
    scores = []
    for n, m in model.named_modules(): 
        if isinstance(m, stats_sample_mode):
            mask1 = mask[n]
            score1 = m.score[mask1==1]
            scores.append(score1.view(-1))
           
    scores = torch.cat(scores, 0)
    return scores.tolist()
    

def get_masks(model, sample_mode, ckpt):
    score_mask = {}
    rand_mask = {}
    min_mask = {}    

    for n, m in model.named_modules(): 
        if isinstance(m, sample_mode):            
            score_mask[n] = ckpt['module.{}.score_mask'.format(n)]
            rand_mask[n] = ckpt['module.{}.rand_mask'.format(n)]          
            min_mask[n] = ckpt['module.{}.min_mask'.format(n)]
    
    return score_mask, rand_mask, min_mask


def subsample_data(data1, ratio=0.01):
    np.random.shuffle(data1)
    return data1[:int(ratio*len(data1))]


if __name__ == '__main__':
    main()
