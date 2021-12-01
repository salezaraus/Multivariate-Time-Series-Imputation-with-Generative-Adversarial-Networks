# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 00:15:52 2021

@author: Christopher Salazar
"""


import numpy as np 

# Inputs
kdd_path_missing = 'C:\\Users\\Christopher Salazar\\Desktop\\GAIN Research\\WGAN_RNN\\Code\\KDD_data\\kdd_w_miss.txt'
kdd_path_wo_missing = 'C:\\Users\\Christopher Salazar\\Desktop\\GAIN Research\\WGAN_RNN\\Code\\KDD_data\\kdd_wo_miss.txt'
total_steps = 10898
n_inputs = 66
n_steps = 48

class ReadKDD_Data():
    # first read all dataset
    # before call, determine wheher shuffle
    # produce next batch
    def __init__(self, dataPath_miss, 
                 dataPath_wo_miss,
                 n_inputs,
                 n_steps, 
                 batch_size = 16):
        
        # Read data with missing values
        with open(dataPath_miss) as f:
            self.missing_lines = f.readlines()
        
            
        # Read data without missing values
        with open(dataPath_wo_miss) as g:
            self.non_missing_lines = g.readlines()
        
        
            
        self.n_inputs = n_inputs
        self.n_steps = n_steps
        self.total_steps = 10898
        self.batch_size = batch_size
    
      
    def data_full_time(self, data_lines, n_inputs, n_steps):
        '''
        Reads and places KDD time series dataset into numpy matrices. Stores 
        data matrix, mask matrix and delta time decay matrix for entire times steps. 
        
        '''
        x = [] # Data matrix
        M = [] # Mask matrix 
        delta = [] # Delta Matrix
        
        # Isolate only values from China 
        data_full_chn = data_lines[20:230]
    
    
        # Every 6 time series are air quality inputs for one station 
        for sta_idx in range(0,len(data_full_chn),6): 
            sta_end_idx = sta_idx + 5
            
            m_arr = np.ones((n_steps, n_inputs))
            sta_arr = np.empty((n_steps, n_inputs), dtype = '<U10')
            
            for idx, inpt in enumerate([*range(sta_idx, sta_end_idx + 1)]): 
                sta_arr[:,idx] = data_full_chn[inpt].split(':')[5].split(',')
                
                # strip last entry of '\n'
                sta_arr[-1,idx]  =  sta_arr[-1,idx].strip('\n')
            
                
            m_arr = np.ones((n_steps, n_inputs)) # Initialize mask array
            m_arr[sta_arr == '?'] = 0 # find missing values
            sta_arr[sta_arr == '?'] = 0 # replace missing values with 0 
            
            # Construct decay matrix
            delta_arr = np.array([self.make_delta(x) for x in m_arr.T]).T
            
            # Convert to float
            sta_arr = sta_arr.astype(np.float)
                
            x.append(sta_arr)
            M.append(m_arr)
            delta.append(delta_arr)
            
        return x, M, delta
    
    def data_48_hrs(self, data_lines, n_inputs, n_steps, real = False):
        x = [] # Data matrix
        M = [] # Mask matrix 
        delta = [] # Delta Matrix
        
        
        if real:
            # Read Real data that has slight offset in text file
            data_full_chn = data_lines[22:232]
        else: 
            # Isolate only values from China 
            data_full_chn = data_lines[20:230]
    
        # Every 6 time series are air quality inputs for one station 
        for sta_idx in range(0,n_inputs,n_inputs): 
            sta_end_idx = sta_idx + n_inputs
            
            # Split time series into every 48 hours
            for st_hr in range(0, int(self.total_steps/n_steps)*n_steps, 48): 
                end_hr = st_hr + 48
            
                m_arr = np.ones((n_steps, n_inputs))
                sta_arr = np.empty((n_steps, n_inputs), dtype = '<U10')
            
                for idx, inpt in enumerate([*range(sta_idx, sta_end_idx)]): 
                    sta_arr[:,idx] = data_full_chn[inpt].split(':')[5].split(',')[st_hr:end_hr]
                    
                    # strip last entry of '\n'
                    sta_arr[-1,idx]  =  sta_arr[-1,idx].strip('\n')
            
                
                m_arr = np.ones((n_steps, n_inputs)) # Initialize mask array
                m_arr[sta_arr == '?'] = 0 # find missing values
                sta_arr[sta_arr == '?'] = np.nan # replace missing val with null
                
                # Construct decay matrix
                delta_arr = np.array([self.make_delta(x) for x in m_arr.T]).T
                
                # Convert to float
                sta_arr = sta_arr.astype(np.float)
                    
                x.append(sta_arr)
                M.append(m_arr)
                delta.append(delta_arr)
                
        return x, M, delta

        
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
    
    
    
    def gen_data(self):
        # Generate all relevant data
        
        x, M, delta = self.data_48_hrs(self.missing_lines, 
                                  self.n_inputs, 
                                  self.n_steps)
        
        x_real, M_real, delta_real = self.data_48_hrs(self.non_missing_lines, 
                                                 self.n_inputs, 
                                                 self.n_steps, 
                                                 real =True)
        
        
        # Recover one full year worth of hourly data
        x = x[0:192]
        M = M[0:192]
        delta = delta[0:192]
        
        x_real = x_real[0:192]
        M_real = M_real[0:192]
        delta_real = delta_real[0:192]
        
        x_norm, x_params = self.standardize(x)
        x_norm_real, xr_params = self.standardize(x_real, x_params)
        x_norm[np.isnan(x_norm)] = 0
        
        self.x = x
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
            for b_idx_s in range(0, len(self.x_shuf), self.batch_size):
                 b_idx_e = b_idx_s + self.batch_size
                 
                 x_b =  self.x_shuf[b_idx_s:b_idx_e]
                 M_b =  self.M_shuf[b_idx_s:b_idx_e]
                 delta_b =  self.delta_shuf[b_idx_s:b_idx_e]
                 
                 x_r_b =  self.x_r_shuf[b_idx_s:b_idx_e]
                 M_r_b =  self.M_r_shuf[b_idx_s:b_idx_e]
                 delta_r_b =  self.delta_r_shuf[b_idx_s:b_idx_e]
                 
                 x_steps = [self.n_steps] * self.batch_size
                 
                 #print(np.shape(x_b))
                 #print(M_b)
                 #print(delta_b)
                 #print('\n\n\n')
                 
                 yield x_b, M_b, delta_b, x_r_b, M_r_b, delta_r_b, x_steps
        else: 
            for b_idx_s in range(0, len(self.x_norm), self.batch_size):
                 b_idx_e = b_idx_s + self.batch_size
                 
                 x_b =  self.x_norm[b_idx_s:b_idx_e]
                 M_b =  self.M[b_idx_s:b_idx_e]
                 delta_b =  self.delta[b_idx_s:b_idx_e]
                 
                 x_r_b =  self.x_real[b_idx_s:b_idx_e]
                 M_r_b =  self.M_real[b_idx_s:b_idx_e]
                 delta_r_b =  self.delta_real[b_idx_s:b_idx_e]
                 
                 x_steps = [self.n_steps] * self.batch_size
                 
                 #print(np.shape(x_b))
                 #print(np.shape(M_b))
                 #print(np.shape(delta_b))
                 #print('\n\n\n')
                 
                 yield x_b, M_b, delta_b, x_r_b, M_r_b, delta_r_b, x_steps
            
        
        
# =============================================================================
#         batch_size = 16
#         
#         for i in range(2): 
#             rand_idx = np.random.permutation(len(x_norm))
#             x_shuffle = x_norm[rand_idx]
#             M_shuffle = np.asarray(M)[rand_idx]
#             delta_shuffle = np.asarray(delta)[rand_idx]
#             for b_idx_s in range(0, len(x_shuffle), batch_size):
#                 b_idx_e = b_idx_s + batch_size
#                 print(np.shape(x_shuffle[b_idx_s:b_idx_e]))
#                 print(np.shape(M_shuffle[b_idx_s:b_idx_e]))
#                 print(np.shape(delta_shuffle[b_idx_s:b_idx_e]))
#                 print('\n\n\n')
# =============================================================================
        
# =============================================================================
# dt = ReadKDD_Data(kdd_path_missing, kdd_path_wo_missing, n_inputs, n_steps)
# 
# dt.gen_data()
# dt.shuffle(isShuffle = True)
# 
# for i in range(3): 
#     for x_in, M_in, delta_in, x_step in dt.nextBatch():
#         print(np.shape(x_in))
#         print(np.shape(M_in))
#         print(np.shape(delta_in))
#         print('\n\n\n')
# =============================================================================
    
    
