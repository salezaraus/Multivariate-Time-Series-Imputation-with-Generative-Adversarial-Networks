# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 12:45:09 2021

Gan Imputation using KDD dataset, code is adapted from 
https://github.com/Luoyonghong/Multivariate-Time-Series-Imputation-with-Generative-Adversarial-Networks

@author: Christopher Salazar
"""

from __future__ import division
import os
import math
import time
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from ops import *
from utils import *
import mygru_cell

class WGAN(object):
    model_name = "WGAN_no_mask"     # name for checkpoint

    def __init__(self, sess, args, datasets):
        self.sess = sess
        self.isbatch_normal=args.isBatch_normal
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.dataset_name=args.dataset_name
        self.run_type=args.run_type
        self.lr = args.lr                 
        self.epoch = args.epoch     
        self.batch_size = args.batch_size
        self.n_inputs = args.n_inputs                 # MNIST data input (img shape: 28*28)
        self.n_steps = datasets.n_hours                               # time steps
        self.n_hidden_units = args.n_hidden_units        # neurons in hidden layer
        self.n_classes = args.n_classes                # MNIST classes (0-9 digits)
        self.gpus=args.gpus
        self.run_type=args.run_type
        self.result_path=args.result_path
        self.model_path=args.model_path
        self.pretrain_epoch=args.pretrain_epoch
        self.impute_iter=args.impute_iter
        self.g_loss_lambda=args.g_loss_lambda
        
        self.datasets=datasets
        self.z_dim = args.z_dim         # dimension of noise-vector
        self.gen_length=args.gen_length
        
        # WGAN_GP parameter
        self.lambd = 0.25       # The higher value, the more stable, but the slower convergence
        self.disc_iters = args.disc_iters     # The number of critic iterations for one-step of generator

        # train
        self.learning_rate = args.lr
        self.beta1 = args.beta1
        if "1.5" in tf.__version__ or "1.9" in tf.__version__ :
            self.grud_cell_d = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g = mygru_cell.MyGRUCell15(self.n_hidden_units)
        elif "1.5" in tf.__version__ or "1.7" in tf.__version__ :
            self.grud_cell_d = mygru_cell.MyGRUCell15(self.n_hidden_units)
            self.grud_cell_g = mygru_cell.MyGRUCell15(self.n_hidden_units)    
        elif "1.4" in tf.__version__:
            self.grud_cell_d = mygru_cell.MyGRUCell4(self.n_hidden_units)
            self.grud_cell_g = mygru_cell.MyGRUCell4(self.n_hidden_units)
        elif "1.2" in tf.__version__:
            self.grud_cell_d = mygru_cell.MyGRUCell2(self.n_hidden_units)
            self.grud_cell_g = mygru_cell.MyGRUCell2(self.n_hidden_units)
        # test
        self.sample_num = 64  # number of generated images to be saved

        self.num_batches = len(datasets.x) // self.batch_size

      
    def pretrainG(self, X, M, Delta, X_lengths, Keep_prob, reuse=False):
        
        with tf.variable_scope("g_enerator", reuse=reuse):
            
            """
            the rnn cell's variable scope is defined by tensorflow,
            if we want to update rnn cell's weights, the variable scope must contains 'g_' or 'd_'
            
            """
            
            wr_h=tf.get_variable("g_wr_h",shape=[self.n_inputs,self.n_hidden_units],initializer=tf.random_normal_initializer())
            w_out= tf.get_variable("g_w_out",shape=[self.n_hidden_units, self.n_inputs],initializer=tf.random_normal_initializer())
            
            br_h= tf.get_variable("g_br_h",shape=[self.n_hidden_units, ],initializer=tf.constant_initializer(0.001))
            b_out= tf.get_variable("g_b_out",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            w_z=tf.get_variable("g_w_z",shape=[self.z_dim,self.n_inputs],initializer=tf.random_normal_initializer())
            b_z=tf.get_variable("g_b_z",shape=[self.n_inputs, ],initializer=tf.constant_initializer(0.001))
            
            
            X = tf.reshape(X, [-1, self.n_inputs])
            Delta=tf.reshape(Delta,[-1,self.n_inputs])
            
            rth= tf.matmul(Delta, wr_h)+br_h
            rth=math_ops.exp(-tf.maximum(0.0,rth))
            
            X=tf.concat([X,rth],1)
            
            X_in = tf.reshape(X, [-1, self.n_steps, self.n_inputs+self.n_hidden_units])
         
            init_state = self.grud_cell_g.zero_state(self.batch_size, dtype=tf.float32) # 初始化全零 state
            outputs, final_state = tf.nn.dynamic_rnn(self.grud_cell_g, X_in, \
                                initial_state=init_state,\
                                sequence_length=X_lengths,
                                time_major=False)
            #outputs: batch_size*n_steps*n_hiddensize
            outputs=tf.reshape(outputs,[-1,self.n_hidden_units])
            out_predict=tf.matmul(tf.nn.dropout(outputs,Keep_prob), w_out) + b_out
            out_predict=tf.reshape(out_predict,[-1,self.n_steps,self.n_inputs])
            return out_predict

        
    def impute(self):
        with tf.variable_scope("impute", reuse=tf.AUTO_REUSE):
            z_need_tune=tf.get_variable("z_needtune",shape=[self.batch_size,self.z_dim],initializer=tf.random_normal_initializer(mean=0,stddev=0.1) )
            return z_need_tune
            
    def build_model(self):
        
        self.keep_prob = tf.placeholder(tf.float32) 
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.m = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.deltaPre = tf.placeholder(tf.float32, [None, self.n_steps, self.n_inputs])
        self.x_lengths = tf.placeholder(tf.int32,  shape=[self.batch_size,])
     
        """ Loss Function """

        Pre_out=self.pretrainG(self.x, self.m, self.deltaPre, \
                                                      self.x_lengths,self.keep_prob, \
                                                      reuse=False)
        
        self.pretrain_loss=tf.reduce_sum(tf.square(tf.multiply(Pre_out,self.m)-self.x)) / tf.cast(tf.reduce_sum(self.x_lengths),tf.float32)
        
        

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        g_vars = [var for var in t_vars if 'g_' in var.name]

        '''
        print("d vars:")
        for v in d_vars:
            print(v.name)
        print("g vars:")
        for v in g_vars:
            print(v.name)
        print("z vars:")
        for v in z_vars:
            print(v.name)
        '''
        
        #don't need normalization because we have adopted the dropout
        """
        ld = 0.0
        for w in d_vars:
            ld += tf.contrib.layers.l2_regularizer(1e-4)(w)
        lg = 0.0
        for w in g_vars:
            lg += tf.contrib.layers.l2_regularizer(1e-4)(w)
        
        self.d_loss+=ld
        self.g_loss+=lg
        """
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        # this code have used batch normalization, so the upside line should be executed
            #self.g_optim=self.optim(self.learning_rate, self.beta1,self.g_loss,g_vars)
            self.g_pre_optim=tf.train.AdamOptimizer(self.learning_rate*2,beta1=self.beta1) \
                        .minimize(self.pretrain_loss,var_list=g_vars)

    
        
        

        #clip weight
        self.clip_all_vals = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in t_vars]
        self.clip_G = [p.assign(tf.clip_by_value(p, -0.99, 0.99)) for p in g_vars]
        
       

        """ Summary """

        g_pretrain_loss_sum=tf.summary.scalar("g_pretrain_loss", self.pretrain_loss)
        # final summary operations
        self.g_pretrain_sum=tf.summary.merge([g_pretrain_loss_sum])

        
    def optim(self,learning_rate,beta,loss,var):
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta)
        grads = optimizer.compute_gradients(loss,var_list=var)
        for i, (g, v) in enumerate(grads):
            if g is not None:
                grads[i] = (tf.clip_by_norm(g, 5), v)  # clip gradients
        train_op = optimizer.apply_gradients(grads)
        return train_op
    def pretrain(self, start_epoch,counter,start_time):
        
        if start_epoch < self.pretrain_epoch:
            #todo
            for epoch in range(start_epoch, self.pretrain_epoch):
            # get batch data
                self.datasets.shuffle(True)
                idx=0
                #x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
                for x_in, M_in, delta_in, x_step in self.datasets.nextBatch():
                    
                    # pretrain
                    _, summary_str, p_loss = self.sess.run([self.g_pre_optim, self.g_pretrain_sum, self.pretrain_loss],
                                                   feed_dict={self.x: x_in,
                                                              self.m: M_in,
                                                              self.deltaPre: delta_in,
                                                              self.x_lengths: x_step,
                                                              self.keep_prob: 0.5})
    
    
                    counter += 1
    
                    # display training status
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, pretrain_loss: %.8f" \
                          % (epoch, idx, self.num_batches, time.time() - start_time, p_loss))
                    idx+=1
                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                #调好之后再保存
                #if epoch%10==0:
                #    self.save(self.checkpoint_dir, counter)

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()
        start_epoch = 0
        start_batch_id = 0
        counter = 1

        # loop for epoch
        start_time = time.time()
        
        self.pretrain(start_epoch,counter,start_time)
        if start_epoch < self.pretrain_epoch:
            start_epoch=self.pretrain_epoch


    
    
    
