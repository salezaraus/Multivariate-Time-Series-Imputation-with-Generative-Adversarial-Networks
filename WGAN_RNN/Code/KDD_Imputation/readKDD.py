# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 00:15:52 2021

@author: Christopher Salazar
"""


import numpy as np 
import os

kdd_path = 'C:\\Users\\Christopher Salazar\\Desktop\\GAIN Research\\WGAN_RNN\\Code\\KDD_data\\kdd_w_miss.txt'

with open(kdd_path ) as f:
    lines = f.readlines()
f.close

# Isolate only values from China 
data_full_chn = lines[20:230]


# Check if time series are uniform
# =============================================================================
# for t_s in data_full_chn:
#     n_steps = np.shape(t_s.split(':')[5].split(','))[0]
#     if n_steps != 10898: 
#         print('Time State %s is not uniform' % t_s.split(':')[0])
#         break
# =============================================================================

n_steps = 10898
n_inputs = 6
x = [] # Data matrix
M = [] # Mask matrix 
delta = []

# Every 6 time series are air quality inputs for one station 
for sta_idx in range(0,210,6): 
    sta_end_idx = sta_idx + 5
    
    m_arr = np.ones((n_steps, n_inputs))
    sta_arr = np.empty((n_steps, n_inputs), dtype = '<U10')
    
    for idx, inpt in enumerate([*range(sta_idx, sta_end_idx + 1)]): 
        sta_arr[:,idx] = data_full_chn[inpt].split(':')[5].split(',')
        
        # strip last entry of '\n'
        sta_arr[-1,idx]  =  sta_arr[-1,idx].strip('\n')
    
        
    m_arr = np.ones((n_steps, n_inputs)) # Initialize mask array
    m_arr[sta_arr == '?'] = 0 # find missing values
    sta_arr[sta_arr == '?'] = '0.0' # replace missing values with 0 
    
    # Construct decay matrix
    delta_arr = np.array([make_delta(x) for x in m_arr.T]).T
    
    # Convert to float
    sta_arr = sta_arr.astype(np.float)
        
    x.append(sta_arr)
    M.append(m_arr)
    delta.append(delta_arr)
    
def make_delta(mask_vec): 
    # Constructs time decay matrix from mask time series, delta
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
            
