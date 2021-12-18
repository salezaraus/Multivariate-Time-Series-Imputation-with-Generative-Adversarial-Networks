# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 08:22:11 2021

@author: Christopher Salazar
"""

import numpy as np
import os

#PM2.5,PM10,NO2,CO,O3,SO2

RMSE_1 = np.loadtxt('Results/2021_12_07/RMSE_out1.txt')
RMSE_2 = np.loadtxt('Results/2021_12_07/RMSE_out2.txt')
RMSE_3 = np.loadtxt('Results/2021_12_07/RMSE_out3.txt')
RMSE_4 = np.loadtxt('Results/2021_12_07/RMSE_out4.txt')
RMSE_5 = np.loadtxt('Results/2021_12_07/RMSE_out5.txt')

RMSE_1_ni = np.loadtxt('Results/2021_12_07_No_Imp/RMSE_out1.txt')
RMSE_2_ni = np.loadtxt('Results/2021_12_07_No_Imp/RMSE_out2.txt')
RMSE_3_ni = np.loadtxt('Results/2021_12_07_No_Imp/RMSE_out3.txt')
RMSE_4_ni = np.loadtxt('Results/2021_12_07_No_Imp/RMSE_out4.txt')
RMSE_5_ni = np.loadtxt('Results/2021_12_07_No_Imp/RMSE_out5.txt')

RMSEs = np.array([RMSE_1, RMSE_2, RMSE_3, RMSE_4, RMSE_5])
RMSE_ave = np.mean(RMSEs, axis = 0)

RMSEs = np.array([RMSE_1_ni, RMSE_2_ni, RMSE_3_ni, RMSE_4_ni, RMSE_5_ni])
RMSE_ave_ni = np.mean(RMSEs, axis = 0)
    
    

from matplotlib import pyplot as plt
%matplotlib qt5

Sta_num = 3

sta_dict = {1 : [0,6], 
            2: [6,12], 
            3: [12,18], 
            4: [18,24], 
            5: [24,30], 
            6: [30,36], 
            7: [36,42], 
            8: [42,48], 
            9: [48,54], 
            10: [54,60], 
            11: [60,66]}

sta_ind = [i for i in range(66)]
sta_sl = sta_ind[sta_dict[Sta_num][0]: sta_dict[Sta_num][1]]

epochs = np.arange(1,len(RMSE_ave)+1)

plt.figure(1)
plt.plot(epochs, RMSE_ave[:,sta_sl[0]])
plt.plot(epochs, RMSE_ave_ni[:,sta_sl[0]])
plt.legend(["400 Imp Iter", "No Iter Imp"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title('Average Validation RMSE for PM2.5 at station: miyunshuiku_aq')  

plt.figure(2)
plt.plot(epochs, RMSE_ave[:,sta_sl[1]])
plt.plot(epochs, RMSE_ave_ni[:,sta_sl[1]])
plt.legend(["400 Imp Iter", "No Iter Imp"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title('Average Validation RMSE for PM10 at station: miyunshuiku_aq')  

plt.figure(3)
plt.plot(epochs, RMSE_ave[:,sta_sl[2]])
plt.plot(epochs, RMSE_ave_ni[:,sta_sl[2]])
plt.legend(["400 Imp Iter", "No Iter Imp"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title('Average Validation RMSE for NO2 at station: miyunshuiku_aq') 

plt.figure(4)
plt.plot(epochs, RMSE_ave[:,sta_sl[3]])
plt.plot(epochs, RMSE_ave_ni[:,sta_sl[3]])
plt.legend(["400 Imp Iter", "No Iter Imp"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title('Average Validation RMSE for CO at station: miyunshuiku_aq')

plt.figure(5)
plt.plot(epochs, RMSE_ave[:,sta_sl[4]])
plt.plot(epochs, RMSE_ave_ni[:,sta_sl[4]])
plt.legend(["400 Imp Iter", "No Iter Imp"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title('Average Validation RMSE for O3 at station: miyunshuiku_aq')   

plt.figure(6)
plt.plot(epochs, RMSE_ave[:,sta_sl[5]])
plt.plot(epochs, RMSE_ave_ni[:,sta_sl[5]])
plt.legend(["400 Imp Iter", "No Iter Imp"])
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title('Average Validation RMSE for SO2 at station: miyunshuiku_aq')      