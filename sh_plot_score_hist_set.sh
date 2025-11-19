#!/bin/bash

# SNIP
python3 plot_score_hist_set.py --dataset cifar10 --arch vgg19_bn --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar10 --arch vgg19_no_bn --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar10 --arch resnet2x --depth 32 --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar10 --arch resnet2x_no_bn --depth 32 --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all


python3 plot_score_hist_set.py --dataset cifar100 --arch vgg19_bn --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar100 --arch vgg19_no_bn --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar100 --arch resnet2x --depth 32 --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar100 --arch resnet2x_no_bn --depth 32 --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all


python3 plot_score_hist_set.py --dataset tinyimagenet --arch vgg19_bn --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset tinyimagenet --arch vgg19_no_bn --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset tinyimagenet --arch resnet2x --depth 56 --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset tinyimagenet --arch resnet2x_no_bn --depth 56 --filename stats/score_hist_all --pruning_method SNIP --save_dir images/score_hist_all



# GraSP_abs
python3 plot_score_hist_set.py --dataset cifar10 --arch vgg19_bn --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar10 --arch vgg19_no_bn --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar10 --arch resnet2x --depth 32 --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar10 --arch resnet2x_no_bn --depth 32 --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all


python3 plot_score_hist_set.py --dataset cifar100 --arch vgg19_bn --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar100 --arch vgg19_no_bn --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar100 --arch resnet2x --depth 32 --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset cifar100 --arch resnet2x_no_bn --depth 32 --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all


python3 plot_score_hist_set.py --dataset tinyimagenet --arch vgg19_bn --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset tinyimagenet --arch vgg19_no_bn --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset tinyimagenet --arch resnet2x --depth 56 --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all
python3 plot_score_hist_set.py --dataset tinyimagenet --arch resnet2x_no_bn --depth 56 --filename stats/score_hist_all --pruning_method GraSP_abs --save_dir images/score_hist_all


