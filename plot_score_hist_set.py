import os
import argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.stats as statistics
import numpy as np
from matplotlib.ticker import MaxNLocator

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filename', default='stats/score_hist_set', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--arch', default='vgg19_bn', type=str)
parser.add_argument('--depth', default=32, type=int)
parser.add_argument('--pruning_method', default='SNIP', type=str)
parser.add_argument('--save_dir', default='images/score_hist_set', type=str)
args = parser.parse_args()
    

def main():
    save_dir = '{}/mixture_{}_{}_{}'.format(args.save_dir, args.pruning_method, args.dataset, args.arch)
    if 'resnet' in args.arch: 
        save_dir = '{}_{}_samples'.format(save_dir, args.depth)
    else: 
        save_dir = '{}_samples'.format(save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if 'vgg' in args.arch:
        stats = '{}/{}_{}_{}_total.mat'.format(args.filename, args.dataset, args.arch, args.pruning_method)
    else: 
        stats = '{}/{}_{}_{}_{}_total.mat'.format(args.filename, args.dataset, args.arch, args.depth, args.pruning_method)  
    stats = sio.loadmat(stats)

    max_set = stats['max_set'][0]
    rand_set = stats['rand_set'][0]
    
    draw_histogram(save_dir, max_set, rand_set)
    print(save_dir)        

        
def draw_histogram(save_dir, max1, rand1): 
    n_bins = 50

    min_min = min([min(max1), min(rand1)])
    max_max = max([max(max1), max(rand1)])
    
    bins1 = np.linspace(min_min, max_max, n_bins)
    cnts_max, bins_max = np.histogram(max1, bins1)
    cnts_rand, bins_rand = np.histogram(rand1, bins1)     

    total_max = sum(cnts_max)
    total_rand = sum(cnts_rand)

    cnts_max = [c/total_max for c in cnts_max]
    cnts_rand = [c/total_rand for c in cnts_rand]

    width, height = 6.4*1.5, 4.8
    plt.figure(constrained_layout=True, figsize=(width, height))
    adjust_font_size(args.save_dir)
          
    plt.stairs(cnts_max, bins_max, fill=True, alpha=0.5, color='r',label='Most Impt.')
    plt.stairs(cnts_rand, bins_rand, fill=True, alpha=0.5, color='b', label='Random')
    
    plt.xlabel('Score')
    plt.ylabel('Proportion')
    plt.legend(loc='upper right')
    plt.yscale('log', base=2)
    # plt.ylim([-0.05, 1.05])
    
    ax = plt.gca()
    ax.set_rasterized(True)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))    

    plt.title('KS2 stat.: {:.2f}'.format(k_distance(max1, rand1)))
    
    image_extension = 'pdf'
    if 'tinyimagenet' in save_dir: 
        save_dir = save_dir.replace('tinyimagenet', 'ti')
    plt.savefig('{}.{}'.format(save_dir, image_extension), bbox_inches='tight')
    plt.close()
    

def k_distance(d1, d2):
    return statistics.ks_2samp(d1, d2).statistic


def adjust_font_size(save_dir):
    SMALL_SIZE = 30
    if 'yolk' in save_dir: SMALL_SIZE = 30
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    
    
if __name__ == '__main__':
    main()