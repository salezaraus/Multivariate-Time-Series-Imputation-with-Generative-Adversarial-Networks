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
import WGAN_GRUI_KDD_T
import tensorflow as tf
import argparse
import numpy as np
from readKDD_train import ReadKDD_Data
import os
from utils import show_all_variables

"""main"""
def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gen-length', type=int, default=96)
    parser.add_argument('--impute-iter', type=int, default=1)
    parser.add_argument('--pretrain-epoch', type=int, default=50)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path', type=str, default='../KDD_data/beijing_17_18_aq.csv')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--g-loss-lambda',type=float,default=0.1)
    parser.add_argument('--beta1',type=float,default=0.5)
    parser.add_argument('--lr', type=float, default=0.002)
    #lr 0.001的时候 pretrain_loss降的很快，4个epoch就行了
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--n-stations', type=int, default=11)
    parser.add_argument('--n-steps', type=int, default=24)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--z-dim', type=int, default=256)
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
    epochs=[90]
    g_l = 0.0
    beta1s=[0.5]
    counter = 1
    RMSE_tot = []
    dt_train=ReadKDD_Data(args.data_path, args.n_steps, args.n_stations)
    dt_train.partition_data()
    for beta1 in beta1s:
        for e in epochs:
            for fold_idx in dt_train.fold_idx:
                args.epoch=e
                args.beta1=beta1
                args.g_loss_lambda=g_l
                tf.reset_default_graph()
                dt_train.gen_data(fold_idx)
                tf.reset_default_graph()
                config = tf.ConfigProto() 
                config.gpu_options.allow_growth = True 
                with tf.Session(config=config) as sess:
                    gan = WGAN_GRUI_KDD_T.WGAN(sess,
                                args=args,
                                datasets=dt_train,
                                )
            
                    # build graph
                    gan.build_model()
            
                    # show network architecture
                    show_all_variables()
            
                    # launch the graph in a session
                    RMSEs = gan.train()
                    print(" [*] Training finished!")
                    
                    gan.plot_loss(counter)
                    
                    counter +=1
                    
                    #x_imputed, x_real, M_batch, deltas, fake_data,  rand_idx, norm_params, RMSE = gan.imputation(dt_train)
                    
                    RMSE_tot.append(RMSEs)
                    
                    
                    print(" [*] Test dataset Imputation finished!")
                tf.reset_default_graph()
                
    #return x_imputed, x_real, M_batch, deltas, fake_data, rand_idx, norm_params
    return RMSE_tot
if __name__ == '__main__':
    #x_imputed, x_real, M_batch, deltas, fake_data, rand_idx, norm_params = main()
    RMSE_tot = main()
    
    
# =============================================================================
# mean_val = norm_params['mean']
# std_val = norm_params['std']
#    
# x_i_renorm = [(x*std_val + mean_val) for x in x_imputed]
# x_r_renorm = [(x*std_val + mean_val) for x in x_real]
# 
# from itertools import chain
# 
# x_i_order = np.array(list(chain(*x_i_renorm)))
# x_r_order = np.array(list(chain(*x_r_renorm)))
# M_order = np.array(list(chain(*M_batch)))
# 
# x_i_order = [x_i_order[np.where(rand_idx == i)] for i in range(len(rand_idx))]
# x_r_order =[x_r_order[np.where(rand_idx == i)] for i in range(len(rand_idx))]
# M_order = [M_order[np.where(rand_idx == i)] for i in range(len(rand_idx))]
# 
# x_i_order = np.array(list(chain(*x_i_order)))
# x_r_order = np.array(list(chain(*x_r_order)))
# M_order = np.array(list(chain(*M_order)))
# 
# x_i_order = np.array(list(chain(*x_i_order)))
# M_order = np.array(list(chain(*M_order)))
# 
# feat = 5
# 
# t_real = np.where(M_order[:,feat] == 1)
# x_real = x_i_order[:,feat][t_real]
# 
# t_imp = np.where(M_order[:,feat] == 0)
# x_imp = x_i_order[:,feat][t_imp]
# 
# from matplotlib import pyplot as plt
# %matplotlib qt5
# 
# plt.figure(1)
# plt.scatter(t_real, x_real, s = 10)
# plt.xlabel("Time (hrs)")
# plt.ylabel("PM10")
# plt.legend(["Real"])
# plt.title('Full time series for SO2 at Beijing:aotizhongxin_aq')
# 
# plt.figure(2)
# plt.scatter(t_real, x_real, s = 10)
# plt.scatter(t_imp, x_imp, s = 10)
# plt.xlabel("Time (hrs)")
# plt.ylabel("PM10")
# plt.title('Full time series for SO2 at Beijing:aotizhongxin_aq')
# plt.legend(["Real", "Imputed"])
# =============================================================================

# =============================================================================
# min_val = norm_params['min_val']
# max_val = norm_params['max_val']
# #    
# x_i_renorm = [(x*(max_val + 1e-6) + min_val) for x in x_imputed]
# x_r_renorm = [(x*(max_val + 1e-6) + min_val) for x in x_real]
# =============================================================================

#PM2.5,PM10,NO2,CO,O3,SO2
    
# =============================================================================
# RMSEs = np.array([np.asarray(RMSE_tot[0]), np.asarray(RMSE_tot[1]), np.asarray(RMSE_tot[2]), np.asarray(RMSE_tot[3]), np.asarray(RMSE_tot[4])])
# RMSE_ave = np.mean(RMSEs, axis = 0)
# 
# from matplotlib import pyplot as plt
# %matplotlib qt5
# 
# epochs = np.arange(1,len(RMSE_ave)+1)
# 
# plt.figure(1)
# plt.plot(epochs, RMSE_ave[:,0])
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title('Average Validation RMSE for PM2.5 at station: miyunshuiku_aq')  
# 
# plt.figure(2)
# plt.plot(epochs, RMSE_ave[:,1])
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title('Average Validation RMSE for PM10 at station: miyunshuiku_aq')  
# 
# plt.figure(3)
# plt.plot(epochs, RMSE_ave[:,2])
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title('Average Validation RMSE for NO2 at station: miyunshuiku_aq') 
# 
# plt.figure(4)
# plt.plot(epochs, RMSE_ave[:,3])
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title('Average Validation RMSE for CO at station: miyunshuiku_aq')
# 
# plt.figure(5)
# plt.plot(epochs, RMSE_ave[:,4])
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title('Average Validation RMSE for O3 at station: miyunshuiku_aq')   
# 
# plt.figure(6)
# plt.plot(epochs, RMSE_ave[:,5])
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title('Average Validation RMSE for SO2 at station: miyunshuiku_aq') 
#     
# =============================================================================
    
  
