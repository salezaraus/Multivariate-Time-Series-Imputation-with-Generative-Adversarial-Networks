# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 00:15:52 2021

@author: Christopher Salazar
"""


import numpy as np
import random 

# Inputs
data_path = 'C:\\Users\\Christopher Salazar\\Desktop\\GAIN Research\\WGAN_RNN\\Code\\KDD_data\\beijing_17_18_aq.csv'
n_steps = 24
n_stations = 11

class ReadKDD_Data():
    # first read all dataset
    # before call, determine wheher shuffle
    # produce next batch
    def __init__(self, data_path,
                 n_steps, 
                 n_stations, 
                 batch_size = 16):
        
        # Read data with missing values
        kdd = np.genfromtxt(data_path, delimiter=',',skip_header=1,dtype = 'str')   
        
        self.n_inputs = n_stations*6 # six features for each station
        self.n_stations = n_stations
        self.n_steps = n_steps # Total steps that will be generated per batch
        self.total_steps = 8886 # total times steps for each station (raw)
        self.batch_size = batch_size
        self.data_path = data_path
        
        # Prepare all known data with full 24 hr periods
        valid_idx = self.get_valid_idx(kdd[:self.total_steps])
        kdd = self.concatenate_values(kdd)[valid_idx]
        
        self.kdd = kdd.astype(np.float)
        self.M_real = np.ones(np.shape(kdd))
        self.t_steps_v = np.shape(kdd)[0] # number of total steps after valid
                                          # entry
        
    def concatenate_values(self, data):
        # Concatentates data together based on how many stations that should  
        # be included as features
        
        for sta in range(self.n_stations):
            if sta == 0: 
                data_con = data[:self.total_steps][:,2:]
            else:
                next_sta = data[sta*self.total_steps:sta*self.total_steps \
                               + self.total_steps][:,2:]
                
                data_con = np.concatenate((data_con, next_sta), axis = 1)
                  
        return data_con
        
    def get_valid_idx(self, data):
        # Returns the indeces from full data for days that have full 24 hour 
        # measurements. Note all stations have the equivalent temporal 
        # measurements
        
        timestamps = data[:self.total_steps] 
        date_dict = {}
        
        # Gather valid dates with 24 hour measurements
        for date_stamp in timestamps[:,1]: 
            date = date_stamp.split()[0]
            
            if date not in date_dict.keys(): 
                date_dict[date] = 1
            else: 
                date_dict[date] += 1 
                
        valid_dates = [k for k,v in date_dict.items() if v == 24]
        valid_dates.sort()

        counter = 0
        for date in valid_dates:
            # Checks if date is in timestamp of kdd array
            if not counter:
                valid_idx  = np.flatnonzero(np.core.defchararray.find(data[:,1],date)!=-1)
                counter = 1
            else:
                idx = np.flatnonzero(np.core.defchararray.find(data[:,1],date)!=-1)
                valid_idx = np.concatenate((valid_idx, idx))
                
        return valid_idx
    
    def partition_data(self, nfolds = 5):
        #creates n folds of random indeces that will be used to partition
        # mask matrix into missing data and non-missing data
        
        total_ndata = self.t_steps_v*self.n_inputs
        idx_list = np.arange(total_ndata)
        random.shuffle(idx_list)
        
        idx_list = [idx_list[i::nfolds] for i in range(nfolds)]
        
        self.fold_idx = idx_list
            
    
    def data_sliced_hrs(self, fold_idx):
        # partition data into n_steps chunks
        
        # Shape of grouped time series data 
        partition_shape = (int(self.t_steps_v/self.n_steps), 
                          self.n_steps, self.n_inputs)
        
        M = np.ones(np.shape(self.kdd)[0]*np.shape(self.kdd)[1])
        M[fold_idx] = 0
        M = np.reshape(M, (self.t_steps_v, self.n_inputs))
        
        #M_real = np.reshape(self.M_real.copy(), partition_shape)
        M_real = np.ones((self.t_steps_v, self.n_inputs))
        
        delta = np.array([self.make_delta(x) for x in M.T]).T
        delta_real = np.array([self.make_delta(x) for x in M_real.T]).T
        
        ######
        x_miss = self.kdd.copy()
        x_miss[M == 0] = np.nan
        
        x_miss = np.reshape(x_miss, partition_shape)
        x_real = np.reshape(self.kdd.copy(), partition_shape)
        delta = np.reshape(delta,partition_shape)
        delta_real = np.reshape(delta_real, partition_shape)
        M = np.reshape(M, partition_shape)
        M_real = np.reshape(M_real, partition_shape)
        
        
        return x_miss, x_real, M, M_real, delta, delta_real

        
    def make_delta(self, mask_vec): 
        # Constructs time decay matrix from mask time series, delta
        # ref: https://papers.nips.cc/paper/2018/file/96b9bff013acedfb1d140579e2fbeb63-Paper.pdf
        delta_vec = np.zeros(len(mask_vec))
        last_known_idx = 0 
        for t in range(1,len(mask_vec)): 
            if t == 0:
                continue
            elif mask_vec[t] == 1: 
                delta_vec[t] = (t - last_known_idx)*60
                last_known_idx = t
            else: 
                delta_vec[t] = (t - last_known_idx)*60
        
        return delta_vec
    
    
    def normalize(self, x_data, parameters=None):
      '''Normalize data in [0, 1] range.
      
      Args:
        - data: original data
      
      Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
      '''
      if parameters is None:
        min_val = np.nanmin(np.nanmin(x_data, axis = 1), axis = 0)
        max_val = np.nanmax(np.nanmax(x_data, axis = 1), axis = 0)
        
        # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                         'max_val': max_val} 
    
      else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        
    
        
        norm_parameters = {'min_val': min_val,
                     'max_val': max_val} 
    
      normalized_data = (x_data-min_val)/(max_val + 1e-6)
      #print(norm_parameters)
      
      return normalized_data, norm_parameters
  
    def standardize(self, x_data, parameters=None):
      '''Stardardize data in [0, 1] range.
      
      Args:
        - data: original data
      
      Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
      '''
      if parameters is None:
        mean_val = np.nanmean(np.nanmean(x_data, axis = 1), axis = 0)
        std_val = np.nanstd(np.nanstd(x_data, axis = 1), axis = 0)
        
        # Return norm_parameters for renormalization
        stad_parameters = {'mean': mean_val,
                         'std': std_val} 
    
      else:
        mean_val = parameters['mean']
        std_val = parameters['std']
        
    
        
        stad_parameters = {'mean': mean_val,
                     'std': std_val} 
    
      standardized_data = (x_data-mean_val)/std_val
      #print(norm_parameters)
      
      return standardized_data, stad_parameters
    
       
    def gen_data(self, fold_idx):
        # Generate all relevant data
        x_miss, x_real, M, M_real, delta, delta_real = self.data_sliced_hrs(fold_idx)
        
        x_norm, x_params = self.standardize(x_miss)
        x_norm_real, xr_params = self.standardize(x_real, x_params)
        x_norm[np.isnan(x_norm)] = 0
        
        self.x = x_miss
        self.M = M
        self.delta = delta 
        
        self.x_real = x_real
        self.M_real = M_real
        self.delta_real = delta_real 
        
        self.x_norm = x_norm
        self.x_norm_real = x_norm_real
        self.norm_params = x_params 
        
    def shuffle(self, isShuffle=False):
        # Shuffles data for each epoch
        if isShuffle:
            # creat random indexing of data
            rand_idx = np.random.permutation(len(self.x_norm))
            
            x_shuffle = self.x_norm[rand_idx]
            M_shuffle = np.asarray(self.M)[rand_idx]
            delta_shuffle = np.asarray(self.delta)[rand_idx]
            
            x_r_shuffle = self.x_norm_real[rand_idx]
            M_r_shuffle = np.asarray(self.M_real)[rand_idx]
            delta_r_shuffle = np.asarray(self.delta_real)[rand_idx]
            
            self.x_shuf = x_shuffle
            self.M_shuf = M_shuffle
            self.delta_shuf = delta_shuffle
            
            self.x_r_shuf = x_r_shuffle
            self.M_r_shuf = M_r_shuffle
            self.delta_r_shuf = delta_r_shuffle
            
            self.isShuffle = isShuffle
            self.rand_idx = rand_idx 
            
    def nextBatch(self):
        if self.isShuffle: 
            for b_idx_s in range(0, int(len(self.x_shuf)/self.batch_size)* self.batch_size, self.batch_size):
                 b_idx_e = b_idx_s + self.batch_size
                 
                 x_b =  self.x_shuf[b_idx_s:b_idx_e]
                 M_b =  self.M_shuf[b_idx_s:b_idx_e]
                 delta_b =  self.delta_shuf[b_idx_s:b_idx_e]
                 
                 x_r_b =  self.x_r_shuf[b_idx_s:b_idx_e]
                 M_r_b =  self.M_r_shuf[b_idx_s:b_idx_e]
                 delta_r_b =  self.delta_r_shuf[b_idx_s:b_idx_e]
                 
                 #n_batch = len(x_b)
                 
                 x_steps = [self.n_steps] * self.batch_size
                 
                 #print(np.shape(x_b))
                 #print(M_b)
                 #print(delta_b)
                 #print('\n\n\n')
                 
                 yield x_b, M_b, delta_b, x_r_b, M_r_b, delta_r_b, x_steps
        else: 
            for b_idx_s in range(0, int(len(self.x_shuf)/self.batch_size)* self.batch_size, self.batch_size):
                 b_idx_e = b_idx_s + self.batch_size
                 
                 x_b =  self.x_norm[b_idx_s:b_idx_e]
                 M_b =  self.M[b_idx_s:b_idx_e]
                 delta_b =  self.delta[b_idx_s:b_idx_e]
                 
                 x_r_b =  self.x_real[b_idx_s:b_idx_e]
                 M_r_b =  self.M_real[b_idx_s:b_idx_e]
                 delta_r_b =  self.delta_real[b_idx_s:b_idx_e]
                 
                # n_batch= len(x_b)
                 
                 x_steps = [self.n_steps] * self.batch_size
                 
                 #print(np.shape(x_b))
                 #print(np.shape(M_b))
                 #print(np.shape(delta_b))
                 #print('\n\n\n')
                 
                 yield x_b, M_b, delta_b, x_r_b, M_r_b, delta_r_b, x_steps
            
        
        
# =============================================================================
# dt = ReadKDD_Data(data_path, n_steps, n_stations)
# dt.partition_data()
# fold_idx = dt.fold_idx[0]
# dt.gen_data(fold_idx)
# dt.shuffle(True)
# =============================================================================
# =============================================================================
# for x_b, M_b, delta_b, x_r_b, M_r_b, delta_r_b, x_steps, batch_dyn in dt.nextBatch(): 
#     print(batch_dyn)
# =============================================================================

    
