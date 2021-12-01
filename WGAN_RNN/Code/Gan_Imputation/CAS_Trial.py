# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 20:06:48 2021

@author: Christopher Salazar
"""

from __future__ import print_function
import sys
sys.path.append("..")
import WGAN_GRUI 
import tensorflow as tf
import argparse
import numpy as np
from Physionet2012Data import readData, readTestData
import os


# parse arguments
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpus', type=str, default = None)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--gen-length', type=int, default=96)
parser.add_argument('--impute-iter', type=int, default=400)
parser.add_argument('--pretrain-epoch', type=int, default=5)
parser.add_argument('--run-type', type=str, default='train')
parser.add_argument('--data-path', type=str, default="../set-a/")
parser.add_argument('--model-path', type=str, default=None)
parser.add_argument('--result-path', type=str, default=None)
parser.add_argument('--dataset-name', type=str, default=None)
parser.add_argument('--g-loss-lambda',type=float,default=0.1)
parser.add_argument('--beta1',type=float,default=0.5)
parser.add_argument('--lr', type=float, default=0.001)
#lr 0.001的时候 pretrain_loss降的很快，4个epoch就行了
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--n-inputs', type=int, default=41)
parser.add_argument('--n-hidden-units', type=int, default=64)
parser.add_argument('--n-classes', type=int, default=2)
parser.add_argument('--z-dim', type=int, default=64)
parser.add_argument('--checkpoint-dir', type=str, default='checkpoint',
                    help='Directory name to save the checkpoints')
parser.add_argument('--result-dir', type=str, default='results',
                    help='Directory name to save the generated images')
parser.add_argument('--log-dir', type=str, default='logs',
                    help='Directory name to save training logs')
parser.add_argument('--isNormal',type=int,default=1)
#0 false 1 true
parser.add_argument('--isBatch-normal',type=int,default=1)
parser.add_argument('--isSlicing',type=int,default=1)
parser.add_argument('--disc-iters',type=int,default=8)
args = parser.parse_args()

if args.isBatch_normal==0:
        args.isBatch_normal=False
if args.isBatch_normal==1:
        args.isBatch_normal=True
if args.isNormal==0:
        args.isNormal=False
if args.isNormal==1:
        args.isNormal=True
if args.isSlicing==0:
        args.isSlicing=False
if args.isSlicing==1:
        args.isSlicing=True
        
args.epoch=30
g_loss_lambdas=0.15
beta1s=0.5

tf.reset_default_graph()
dt_train=readData.ReadPhysionetData(os.path.join(args.data_path,"train"), os.path.join(args.data_path,"train","list.txt"),isNormal=args.isNormal,isSlicing=args.isSlicing)
tf.reset_default_graph()
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 

with tf.Session(config=config) as sess:
    gan = WGAN_GRUI.WGAN(sess,
                args=args,
                datasets=dt_train,
                )
    
    # build graph
    gan.build_model()
    
    # show network architecture
    #show_all_variables()
    
    # launch the graph in a session
    gan.train()
    print(" [*] Training finished!")


# =============================================================================
# n_in = 41
# n_hid = 64
# n_steps = 48
# 
# wr_h=tf.get_variable("d_wr_h",shape=[n_in,n_hid],initializer=tf.random_normal_initializer())
# w_out= tf.get_variable("d_w_out",shape=[n_hid, 1],initializer=tf.random_normal_initializer())
# br_h= tf.get_variable("d_br_h",shape=[n_hid, ],initializer=tf.constant_initializer(0.001))
# b_out= tf.get_variable("d_b_out",shape=[1, ],initializer=tf.constant_initializer(0.001))
# 
# M = dt.m[0:4]
# X = dt.x[0:4]
# deltaP = dt.deltaPre[0:4]
# xlengths1 = X_lengths[0:4]
# 
# X = tf.reshape(X, [-1, n_in])
# DeltaPre=tf.reshape(deltaP,[-1,n_in])
# rth= tf.matmul(DeltaPre, wr_h)+br_h
# rth=math_ops.exp(-tf.maximum(0.0,rth))
# 
# 
# X=tf.concat([X,rth],1)
# X_in = tf.reshape(X, [4, n_steps , n_in+n_hid])
# grud_cell_d = mygru_cell.MyGRUCell15(n_hid)
# init_state = grud_cell_d.zero_state(4, dtype=tf.float32)
# 
# 
# outputs, final_state = tf.nn.dynamic_rnn(grud_cell_d, X_in, \
#                     initial_state=init_state,\
#                     sequence_length=xlengths1,
#                     time_major=False)
# 
# keep_prob = 0.5
# 
# out_logit=tf.matmul(tf.nn.dropout(final_state,keep_prob), w_out) + b_out
# out =tf.nn.sigmoid(out_logit)    #选取最后一个 output
# =============================================================================

