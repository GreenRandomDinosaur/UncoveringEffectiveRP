#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch vgg19_bn --save_dir stats/score_hist_all
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch vgg19_no_bn --save_dir stats/score_hist_all  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch resnet2x --depth 32 --save_dir stats/score_hist_all   
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch resnet2x_no_bn --depth 32 --save_dir stats/score_hist_all


CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch vgg19_bn --save_dir stats/score_hist_all --sample_ratio 0.1  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch vgg19_no_bn --save_dir stats/score_hist_all --sample_ratio 0.1  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch resnet2x --depth 32 --save_dir stats/score_hist_all   
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch resnet2x_no_bn --depth 32 --save_dir stats/score_hist_all


CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch vgg19_bn --save_dir stats/score_hist_all  --sample_ratio 0.1
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch vgg19_no_bn --save_dir stats/score_hist_all --sample_ratio 0.1  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch resnet2x --depth 56 --save_dir stats/score_hist_all 
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch resnet2x_no_bn --depth 56 --save_dir stats/score_hist_all




###GraSP_abs

CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch vgg19_bn --pruning_method GraSP_abs --save_dir stats/score_hist_all
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch vgg19_no_bn --pruning_method GraSP_abs --save_dir stats/score_hist_all  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch resnet2x --depth 32 --pruning_method GraSP_abs --save_dir stats/score_hist_all   
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar10 --batch_size 128 --arch resnet2x_no_bn --depth 32 --pruning_method GraSP_abs --save_dir stats/score_hist_all


CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch vgg19_bn --pruning_method GraSP_abs --save_dir stats/score_hist_all  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch vgg19_no_bn --pruning_method GraSP_abs --save_dir stats/score_hist_all  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch resnet2x --depth 32 --pruning_method GraSP_abs --save_dir stats/score_hist_all   
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset cifar100 --batch_size 128 --arch resnet2x_no_bn --depth 32 --pruning_method GraSP_abs --save_dir stats/score_hist_all


CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch vgg19_bn --pruning_method GraSP_abs --save_dir stats/score_hist_all  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch vgg19_no_bn --pruning_method GraSP_abs --save_dir stats/score_hist_all  
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch resnet2x --depth 56 --pruning_method GraSP_abs --save_dir stats/score_hist_all --sample_ratio 0.1 --rounds 30 --sample_ratio 0.1
CUDA_VISIBLE_DEVICES=0 python3 stats_score_set.py --dataset tinyimagenet --batch_size 128 --arch resnet2x_no_bn --depth 56 --pruning_method GraSP_abs --save_dir stats/score_hist_all --sample_ratio 0.1 --rounds 30 --sample_ratio 0.1

