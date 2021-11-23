# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Sun Nov 21 16:29:08 2021

@author: Christopher Salazar
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 13:54:53 2018

@author: yonghong, luo
"""
import sys
sys.path.append("..")
import WGAN_GRUI_KDD
import tensorflow as tf
import argparse
import numpy as np
from readKDD import ReadKDD_Data
import os
from utils import show_all_variables

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--impute-iter', type=int, default=800)
    parser.add_argument('--pretrain-epoch', type=int, default=20)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path-miss', type=str, default='../KDD_data/kdd_w_miss.txt')
    parser.add_argument('--data-path-wo-miss', type=str, default='../KDD_data/kdd_wo_miss.txt')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--g-loss-lambda',type=float,default=0.1)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--lr', type=float, default=0.002)
    #lr 0.001的时候 pretrain_loss降的很快，4个epoch就行了
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--n-inputs', type=int, default=6)
    parser.add_argument('--n-hours', type=int, default=48)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--z-dim', type=int, default=64)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory name to save training logs')
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)
    parser.add_argument('--disc-iters',type=int,default=8)
    args = parser.parse_args()
    
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True

    #make the max step length of two datasett the same
    epochs=[30]
    g_loss_lambdas=[0.15]
    beta1s=[0.5]
    for beta1 in beta1s:
        for e in epochs:
            for g_l in g_loss_lambdas:
                args.epoch=e
                args.beta1=beta1
                args.g_loss_lambda=g_l
                tf.reset_default_graph()
                dt_train=ReadKDD_Data(args.data_path_miss, args.data_path_wo_miss, args.n_inputs, args.n_hours)
                dt_train.gen_data()
                tf.reset_default_graph()
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True 
                with tf.Session(config=config) as sess:
                    gan = WGAN_GRUI_KDD.WGAN(sess,
                                args=args,
                                datasets=dt_train,
                                )
            
                    # build graph
                    gan.build_model()
            
                    # show network architecture
                    show_all_variables()
            
                    # launch the graph in a session
                    gan.train()
                    print(" [*] Training finished!")
                    
                    
                    print(" [*] Test dataset Imputation finished!")
                tf.reset_default_graph()
if __name__ == '__main__':
    main()