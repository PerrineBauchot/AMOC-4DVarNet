#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:46:33 2022
Last update: 12 December 2023
@author: Perrine Bauchot - Lab-STICC (ENSTA Bretagne + IMT Atlantique) x Chaire Oceanix
"""
""" IMPORT LIBRAIRIES """

import numpy as np
import os
import random
import matplotlib.pyplot as plt 
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from sklearn.feature_extraction import image

from scipy.integrate import odeint
from scipy.interpolate import interp1d

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

""" PARAMETRES """

num_experiment = 1 #choice of the sampling strategy (1: regular; 2: regular cluster; 3: random; 4: random cluster)
flagTypeMissData = 1 #sampling strategy (1: every variable is observed; 2: only x1 is observed; 3: only (x2, x3) are observed )
time_step_obs = 50 #sampling period
flagAEType = 'ode' #choice of the assimilation method ('unet': fully data driven - phi as a NN; 'ode': physics-informed - phi as physical equations)
epoch = 300
step_calcul = 10 #undersampling to speed up computation
directory = '/home/bauchope/'
results_name='varnet_ode_f50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt=1
eps=0.001 #small coefficient for derivation

#Constants
ssf = 35/1000 #physical constant
l = 0.01 #friction coefficient
K = 10**(-4) #diffusion coefficient
omega0 = -2.5*10**(-2) #thermally and wind-driven circulation component
epsilon = 0.35 #buoyancy coefficient
beta = 7*10**(-4) #haline contraction coefficient
dim = beta*epsilon/l #scaling coefficient

""" GENERATION OF DATA"""

def model(V, t, ssf, l, K, omega0, epsilon, beta):
  """
  V : vecteur des variables à modéliser
  l, K, omega0, epsilon, beta : paramètres
  ssf : surface salt flux (ssf=F0*S0/h avec F0 l'intensité du flux de d'eau douce; S0 la salinité de référence; h l'épaisseur de la boucle)
  """
  x_1 = -l*V[0] - epsilon*beta*V[2]
  x_2 = (omega0+V[0])*V[2] - K*V[1] + ssf
  x_3 = -(omega0+V[0])*V[1] - K*V[2]
  dV = np.array([x_1, x_2, x_3])
  return dV

y0 = np.array([0.,0.,0.]) #initial conditions
tt = np.arange(0.,100000.,1) #temporal series
S_init = odeint(model,y0,tt,args=(ssf,l,K,omega0,epsilon,beta)) #initial integration
S_test=np.zeros((100,100000,3)) #test dataset
S_train=np.zeros((100,100000,3)) #training dataset
list_CI_test = np.linspace(65000,95000,100) #time limits for initial conditions of the test dataset
list_CI_train = np.random.randint(5000,60000,100) #time limits for the initial conditions of the training dataset

#filling in S_test as a basis to build a various and representative test dataset of 100 trajectories
for i in range(100):
    k_test = int(list_CI_test[i])
    y0_test=S_init[k_test,:]
    S_temp_test = odeint(model,y0_test,tt,args=(ssf,l,K,omega0,epsilon,beta))
    S_temp_test[:,1]=dim*S_temp_test[:,1]
    S_temp_test[:,2]=dim*S_temp_test[:,2]
    S_test[i,:,:]=S_temp_test[:,:]

#filling in S_train as a basis to build a various and representative training dataset of 100 trajectories
for i in range(100):
    k_train = int(list_CI_train[i])
    y0_train=S_init[k_train,:]
    S_temp_train = odeint(model,y0_train,tt,args=(ssf,l,K,omega0,epsilon,beta))
    S_temp_train[:,1]=dim*S_temp_train[:,1]
    S_temp_train[:,2]=dim*S_temp_train[:,2]
    S_train[i,:,:]=S_temp_train[:,:]


""" OBSERVATIONS """

###############################################
## Generation of training and test dataset
## Extraction of time series of dT time steps 

NbTraining = 50
NbTest     = 1
time_step = 1
dT        = 2500 #duration of one trajectory

#Observation noise as a 10% ratio of the variance of the small-scale oscillations of each variable
bruit_1 = 0.1*np.std(S_test[0,900:2000,0])**2
bruit_2 = 0.1*np.std(S_test[0,900:2000,1])**2
bruit_3 = 0.1*np.std(S_test[0,900:2000,2])**2

# extract subsequences
dataTrainingNoNaN = np.zeros((5000,2500,3))
dk=0
for i in range(100):
    dataTrainingNoNaN[dk:dk+50,:,:] = image.extract_patches_2d(S_train[i,5000:60000:time_step,:],(dT,3),max_patches=NbTraining)
    dk+=50

dataTestNoNaN = np.zeros((100,2500,3))
t_depart=65000
for i in range(100):
    t_fin=t_depart+2500
    dataTestNoNaN[i,:,:] = S_test[i,t_depart:t_fin:time_step,:]
    t_depart=t_depart+325

# create missing data depending on the sampling strategy
if num_experiment == 1:
    if flagTypeMissData == 3:
        print('..... Sampling strategy: Regular / Observation pattern: Only x1 is observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,::time_step_obs,0] = dataTrainingNoNaN[:,::time_step_obs,0]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,::time_step_obs,0] = dataTestNoNaN[:,::time_step_obs,0]
    
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
       
    elif flagTypeMissData == 2:
        print('..... Regular sampling: Observation pattern: Only (x2, x3) are observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,::time_step_obs,2] = dataTrainingNoNaN[:,::time_step_obs,2]
        dataTraining[:,::time_step_obs,1] = dataTrainingNoNaN[:,::time_step_obs,1]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,::time_step_obs,2] = dataTestNoNaN[:,::time_step_obs,2]
        dataTest[:,::time_step_obs,1] = dataTestNoNaN[:,::time_step_obs,1]
        
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_3)**2)
        
    else:
        print('..... Sampling strategy: Regular / Observation pattern: All components are observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]
        
        genSuffixObs    = '_ObsSub_%02d_%02d'%(1/time_step_obs,0)
        
elif num_experiment == 2:
    if flagTypeMissData == 3:
        print('..... Sampling strategy: Regular cluster / Observation pattern: Only x1 is observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,:-20:time_step_obs,0] = dataTrainingNoNaN[:,:-20:time_step_obs,0]
        dataTraining[:,10:-10:time_step_obs,0] = dataTrainingNoNaN[:,10:-10:time_step_obs,0]
        dataTraining[:,20::time_step_obs,0] = dataTrainingNoNaN[:,20::time_step_obs,0]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,:-20:time_step_obs,0] = dataTestNoNaN[:,:-20:time_step_obs,0]
        dataTest[:,10:-10:time_step_obs,0] = dataTestNoNaN[:,10:-10:time_step_obs,0]
        dataTest[:,20::time_step_obs,0] = dataTestNoNaN[:,20::time_step_obs,0]
    
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
    
    elif flagTypeMissData == 2:
        print('..... Sampling strategy: Regular cluster / Observation pattern: Only (x2, x3) are observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,:-20:time_step_obs,1] = dataTrainingNoNaN[:,:-20:time_step_obs,1]
        dataTraining[:,10:-10:time_step_obs,1] = dataTrainingNoNaN[:,10:-10:time_step_obs,1]
        dataTraining[:,20::time_step_obs,1] = dataTrainingNoNaN[:,20::time_step_obs,1]
        dataTraining[:,:-20:time_step_obs,2] = dataTrainingNoNaN[:,:-20:time_step_obs,2]
        dataTraining[:,10:-10:time_step_obs,2] = dataTrainingNoNaN[:,10:-10:time_step_obs,2]
        dataTraining[:,20::time_step_obs,2] = dataTrainingNoNaN[:,20::time_step_obs,2]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,:-20:time_step_obs,1] = dataTestNoNaN[:,:-20:time_step_obs,1]
        dataTest[:,10:-10:time_step_obs,1] = dataTestNoNaN[:,10:-10:time_step_obs,1]
        dataTest[:,20::time_step_obs,1] = dataTestNoNaN[:,20::time_step_obs,1]
        dataTest[:,:-20:time_step_obs,2] = dataTestNoNaN[:,:-20:time_step_obs,2]
        dataTest[:,10:-10:time_step_obs,2] = dataTestNoNaN[:,10:-10:time_step_obs,2]
        dataTest[:,20::time_step_obs,2] = dataTestNoNaN[:,20::time_step_obs,2]
    
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
        
    else:
        print('..... Sampling strategy: Regular cluster / Observation pattern: All components observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTraining[:,:-20:time_step_obs,:] = dataTrainingNoNaN[:,:-20:time_step_obs,:]
        dataTraining[:,10:-10:time_step_obs,:] = dataTrainingNoNaN[:,10:-10:time_step_obs,:]
        dataTraining[:,20::time_step_obs,:] = dataTrainingNoNaN[:,20::time_step_obs,:]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        dataTest[:,:-20:time_step_obs,:] = dataTestNoNaN[:,:-20:time_step_obs,:]
        dataTest[:,10:-10:time_step_obs,:] = dataTestNoNaN[:,10:-10:time_step_obs,:]
        dataTest[:,20::time_step_obs,:] = dataTestNoNaN[:,20::time_step_obs,:]
    
        genSuffixObs    = '_ObsSub_%02d_%02d'%(1/time_step_obs,0)
        
elif num_experiment == 3:
    if flagTypeMissData == 3:
        print('..... Sampling strategy: Random / Observation pattern: Only x1 is observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        for j in range(5000):
            points_obs_train = random.sample(range(dT)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_train:
                dataTraining[j,k,0] = dataTrainingNoNaN[j,k,0]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        for j in range(100):
            points_obs_test = random.sample(range(dT)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_test:
                dataTest[j,k,0] = dataTestNoNaN[j,k,0]
    
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
    elif flagTypeMissData == 2:
        print('..... Sampling strategy: Random / Observation pattern: Only (x2, x3) are observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        for j in range(5000):
            points_obs_train = random.sample(range(dT)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_train:
                dataTraining[j,k,1] = dataTrainingNoNaN[j,k,1]
                dataTraining[j,k,2] = dataTrainingNoNaN[j,k,2]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        for j in range(100):
            points_obs_test = random.sample(range(dT)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_test:
                dataTest[j,k,1] = dataTestNoNaN[j,k,1]
                dataTest[j,k,2] = dataTestNoNaN[j,k,2]
    
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
        
    else:
        print('..... Sampling strategy: Random / Observation pattern: All components are observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        for j in range(5000):
            points_obs_train = random.sample(range(dT)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_train:
                dataTraining[j,k,:] = dataTrainingNoNaN[j,k,:]
        
        for j in range(100):
            points_obs_test = random.sample(range(dT)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_test:
                dataTest[j,k,:] = dataTestNoNaN[j,k,:]
    
        genSuffixObs    = '_ObsSub_%02d_%02d'%(1/time_step_obs,0)
        
else:
    if flagTypeMissData == 3:
        print('..... Sampling strategy: Random cluster / Observation pattern: Only x1 is observed')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        for j in range(5000):
            points_obs_train = random.sample(range(10,dT-10)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_train:
                dataTraining[j,k-10,0] = dataTrainingNoNaN[j,k-10,0]
                dataTraining[j,k,0] = dataTrainingNoNaN[j,k,0]
                dataTraining[j,k+10,0] = dataTrainingNoNaN[j,k+10,0]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        for j in range(100):
            points_obs_test = random.sample(range(10,dT-10)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_test:
                dataTest[j,k-10,0] = dataTestNoNaN[j,k-10,0]
                dataTest[j,k,0] = dataTestNoNaN[j,k,0]
                dataTest[j,k+10,0] = dataTestNoNaN[j,k+10,0]
            
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
    
    elif flagTypeMissData == 2:
        print('.....  Sampling strategy: Random cluster / Observation pattern: Only (x2,x3) are osberved')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        for j in range(5000):
            points_obs_train = random.sample(range(10,dT-10)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_train:
                dataTraining[j,k-10,1] = dataTrainingNoNaN[j,k-10,1]
                dataTraining[j,k,1] = dataTrainingNoNaN[j,k,1]
                dataTraining[j,k+10,1] = dataTrainingNoNaN[j,k+10,1]
                dataTraining[j,k-10,2] = dataTrainingNoNaN[j,k-10,2]
                dataTraining[j,k,2] = dataTrainingNoNaN[j,k,2]
                dataTraining[j,k+10,2] = dataTrainingNoNaN[j,k+10,2]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        for j in range(100):
            points_obs_test = random.sample(range(10,dT-10)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_test:
                dataTest[j,k-10,1] = dataTestNoNaN[j,k-10,1]
                dataTest[j,k,1] = dataTestNoNaN[j,k,1]
                dataTest[j,k+10,1] = dataTestNoNaN[j,k+10,1]
                dataTest[j,k-10,2] = dataTestNoNaN[j,k-10,2]
                dataTest[j,k,2] = dataTestNoNaN[j,k,2]
                dataTest[j,k+10,2] = dataTestNoNaN[j,k+10,2]
            
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
        
    else:
        print('.....  Sampling strategy: Random cluster / Observation pattern: All components are osberved')
        dataTraining    = np.zeros((dataTrainingNoNaN.shape))
        dataTraining[:] = float('nan')
        for j in range(5000):
            points_obs_train = random.sample(range(10,dT-10)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_train:
                dataTraining[j,k-10,:] = dataTrainingNoNaN[j,k-10,:]
                dataTraining[j,k,:] = dataTrainingNoNaN[j,k,:]
                dataTraining[j,k+10,:] = dataTrainingNoNaN[j,k+10,:]
        
        dataTest    = np.zeros((dataTestNoNaN.shape))
        dataTest[:] = float('nan')
        for j in range(100):
            points_obs_test = random.sample(range(10,dT-10)[::step_calcul], int(dT//time_step_obs))
            for k in points_obs_test:
                dataTest[j,k-10,:] = dataTestNoNaN[j,k-10,:]
                dataTest[j,k,:] = dataTestNoNaN[j,k,:]
                dataTest[j,k+10,:] = dataTestNoNaN[j,k+10,:]
            
        genSuffixObs    = '_ObsDim0_%02d_%02d'%(1/time_step_obs,10*np.mean(bruit_1)**2)
    
# set to NaN patch boundaries
dataTraining[:,0:10,:] =  float('nan')
dataTest[:,0:10,:]     =  float('nan')
dataTraining[:,dT-10:dT,:] =  float('nan')
dataTest[:,dT-10:dT,:]     =  float('nan')

# mask for NaN
maskTraining = (dataTraining == dataTraining).astype('float')
maskTest     = ( dataTest    ==  dataTest   ).astype('float')

dataTraining = np.nan_to_num(dataTraining)
dataTest     = np.nan_to_num(dataTest)

# Permutation to have channel as #1 component
dataTraining      = np.moveaxis(dataTraining,-1,1)
maskTraining      = np.moveaxis(maskTraining,-1,1)
dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)

dataTest      = np.moveaxis(dataTest,-1,1)
maskTest      = np.moveaxis(maskTest,-1,1)
dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)

############################################
## raw data
X_train         = dataTrainingNoNaN
X_train_missing = dataTraining
mask_train      = maskTraining

X_test         = dataTestNoNaN
X_test_missing = dataTest
mask_test      = maskTest

############################################
## normalized data
meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train) 
stdTr           = np.sqrt( np.mean( (X_train_missing-meanTr)**2 ) / np.mean(mask_train) )

x_train_missing = ( X_train_missing - meanTr ) / stdTr
x_test_missing  = ( X_test_missing - meanTr ) / stdTr

# scale wrt std

x_train = (X_train - meanTr) / stdTr
x_test  = (X_test - meanTr) / stdTr

print('.... MeanTr = %.3f --- StdTr = %.3f '%(meanTr,stdTr))


# Generate noisy observsation
sigNoise_train_1  = np.random.normal(0,bruit_1,(maskTraining.shape[0],maskTraining.shape[2]))
sigNoise_train_2  = np.random.normal(0,bruit_2,(maskTraining.shape[0],maskTraining.shape[2]))
sigNoise_train_3  = np.random.normal(0,bruit_3,(maskTraining.shape[0],maskTraining.shape[2]))
sigNoise_test_1  = np.random.normal(0,bruit_1,(maskTest.shape[0],maskTest.shape[2]))
sigNoise_test_2  = np.random.normal(0,bruit_2,(maskTest.shape[0],maskTest.shape[2]))
sigNoise_test_3  = np.random.normal(0,bruit_3,(maskTest.shape[0],maskTest.shape[2]))
sigNoise_train  = np.random.normal(0,bruit_1,maskTraining.shape)
sigNoise_test  = np.random.normal(0,bruit_1,maskTest.shape)
X_train_obs = X_train_missing + sigNoise_train * maskTraining * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[1],X_train_missing.shape[2])
X_test_obs  = X_test_missing  + sigNoise_test * maskTest * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[1],X_test_missing.shape[2])
X_train_obs[:,0,:] = X_train_missing[:,0,:] + sigNoise_train_1 * maskTraining[:,0,:] * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[2])
X_test_obs[:,0,:] = X_test_missing[:,0,:] + sigNoise_test_1 * maskTest[:,0,:] * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[2])
X_train_obs[:,1,:] = X_train_missing[:,1,:] + sigNoise_train_2 * maskTraining[:,1,:] * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[2])
X_test_obs[:,1,:] = X_test_missing[:,1,:] + sigNoise_test_2 * maskTest[:,1,:] * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[2])
X_train_obs[:,2,:] = X_train_missing[:,2,:] + sigNoise_train_3 * maskTraining[:,2,:] * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[2])
X_test_obs[:,2,:] = X_test_missing[:,2,:] + sigNoise_test_3 * maskTest[:,2,:] * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[2])
x_train_obs = (X_train_obs - meanTr) / stdTr
x_test_obs  = (X_test_obs - meanTr) / stdTr
print('..... Training dataset: %dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2]))
print('..... Test dataset    : %dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2]))

# Initial interpolation (linear interpolation or zeros) for missing data

mx_train = np.sum( np.sum( X_train , axis = 2 ) , axis = 0 ) / (X_train.shape[0]*X_train.shape[2])

flagInit = 1

if flagInit == 0: 
  X_train_Init = mask_train * X_train_obs + (1. - mask_train) * (np.zeros(X_train_missing.shape) + meanTr)
  X_test_Init  = mask_test * X_test_obs + (1. - mask_test) * (np.zeros(X_test_missing.shape) + meanTr)
else:
  X_train_Init = np.zeros(X_train.shape)
  for ii in range(0,X_train.shape[0]):
    # Initial linear interpolation for each component
    XInit = np.zeros((X_train.shape[1],X_train.shape[2]))

    for kk in range(0,3):
      indt  = np.where( mask_train[ii,kk,:] == 1.0 )[0]
      indt_ = np.where( mask_train[ii,kk,:] == 0.0 )[0]
      if len(indt) > 1:
        indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
        indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
        fkk = interp1d(indt, X_train_obs[ii,kk,indt])
        XInit[kk,indt]  = X_train_obs[ii,kk,indt]
        XInit[kk,indt_] = fkk(indt_)
      else:
        XInit[kk,:] = XInit[kk,:] + mx_train[kk]

    X_train_Init[ii,:,:] = XInit

  X_test_Init = np.zeros(X_test.shape)
  for ii in range(0,X_test.shape[0]):
    # Initial linear interpolation for each component
    XInit = np.zeros((X_test.shape[1],X_test.shape[2]))

    for kk in range(0,3):
      indt  = np.where( mask_test[ii,kk,:] == 1.0 )[0]
      indt_ = np.where( mask_test[ii,kk,:] == 0.0 )[0]

      if len(indt) > 1:
        indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
        indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
        fkk = interp1d(indt, X_test_obs[ii,kk,indt])
        XInit[kk,indt]  = X_test_obs[ii,kk,indt]
        XInit[kk,indt_] = fkk(indt_)
      else:
        XInit[kk,:] = XInit[kk,:] + mx_train[kk]

    X_test_Init[ii,:,:] = XInit

x_train_Init = ( X_train_Init - meanTr ) / stdTr
x_test_Init = ( X_test_Init - meanTr ) / stdTr

# reshape to 2D tensors
x_train = x_train.reshape((-1,3,dT,1))
mask_train = mask_train.reshape((-1,3,dT,1))
x_train_Init = x_train_Init.reshape((-1,3,dT,1))
x_train_obs = x_train_obs.reshape((-1,3,dT,1))

x_test = x_test.reshape((-1,3,dT,1))
mask_test = mask_test.reshape((-1,3,dT,1))
x_test_Init = x_test_Init.reshape((-1,3,dT,1))
x_test_obs = x_test_obs.reshape((-1,3,dT,1))

print('..... Training dataset: %dx%dx%dx%d'%(x_train.shape[0],x_train.shape[1],x_train.shape[2],x_train.shape[3]))
print('..... Test dataset    : %dx%dx%dx%d'%(x_test.shape[0],x_test.shape[1],x_test.shape[2],x_test.shape[3]))

#%%% Visual check of the data generation

plt.figure(figsize=(8,8),dpi=300)
plt.suptitle('Sampling strategy '+str(num_experiment),fontsize=20)
plt.subplot(311)
plt.grid()
plt.tick_params(axis = 'both', labelsize = 15)
plt.plot(x_test[0,0,:,0], color='blue', linewidth=2)
plt.plot(np.where(x_test_obs[0,0,:,0]!=0.), x_test_obs[0,0,np.where(x_test_obs[0,0,:,0]!=0.),0],'o', label='obs', color='black')
#plt.legend()
plt.ylabel('x1 ($yr^{-1}$)', fontsize=15)
plt.xlim(-0.1,2510)

plt.subplot(312)
plt.tick_params(axis = 'both', labelsize = 15)
plt.grid()
plt.plot(x_test[0,1,:,0], color='blue',linewidth=2)
plt.plot(np.where(x_test_obs[0,1,:,0]!=0.), x_test_obs[0,1,np.where(x_test_obs[0,0,:,0]!=0.),0],'o', label='obs', color='black')
#plt.legend()
plt.xlim(-0.1,2510)
plt.ylabel('x2 ($yr^{-1}$)',fontsize=15)

plt.subplot(313)
plt.tick_params(axis = 'both', labelsize = 15)
plt.grid()
plt.plot(x_test[0,2,:,0], color='blue',linewidth=2)
plt.plot(np.where(x_test_obs[0,2,:,0]!=-0.), x_test_obs[0,2,np.where(x_test_obs[0,0,:,0]!=-0.),0],'o', label='obs', color='black')
#plt.legend()
plt.ylabel('x3 ($yr^{-1}$)',fontsize=15)
plt.xlim(-0.1,2510)
plt.xlabel('Time ($yr$)',fontsize=15)
plt.tight_layout()

#%%%
""" 
Trainable projection-based prior PHI 
"""

DimAE = 5
shapeData  = x_train.shape[1:]

if flagAEType == 'ode': ## ode_L63

    class Phi_r(torch.nn.Module):
        def __init__(self):
              super(Phi_r, self).__init__()
              self.l = torch.nn.Parameter(torch.Tensor([l]))
              self.epsilon    = torch.nn.Parameter(torch.Tensor([epsilon]))
              self.beta   = torch.nn.Parameter(torch.Tensor([beta]))
              self.omega0   = torch.nn.Parameter(torch.Tensor([omega0]))
              self.K   = torch.nn.Parameter(torch.Tensor([K]))
              self.ssf   = torch.nn.Parameter(torch.Tensor([ssf]))

              self.dt        = 1
              self.IntScheme = 'rk4'
              self.stdTr     = stdTr
              self.meanTr    = meanTr  
              self.dim = torch.nn.Parameter(torch.Tensor([dim]))
              self.dt = torch.nn.Parameter(torch.Tensor([dt]))
              self.dT = dT
              
        def _ode(self, xin):
        
            dx_1 = (-self.l*xin[:,0,:]-self.epsilon*self.beta*xin[:,2,:])
            dx_2 = ((self.omega0+xin[:,0,:])*xin[:,2,:]-self.K*xin[:,1,:]+self.ssf)
            dx_3 = (-(self.omega0+xin[:,0,:])*xin[:,1,:]-self.K* xin[:,2,:])

            return torch.cat((dx_1.view(xin.size(0),1,xin.size(2)), dx_2.view(xin.size(0),1,xin.size(2)), dx_3.view(xin.size(0),1,xin.size(2))),1)


        def _RK4Solver(self, X_sol):
            
            k1 = self._ode(X_sol)
        
            x2_1 = X_sol[:,0,:] + 0.5*self.dt*k1[:,0,:]
            x2_2 = X_sol[:,1,:] + 0.5*self.dt*k1[:,1,:]
            x2_3 = X_sol[:,2,:] + 0.5*self.dt*k1[:,2,:]
             
            x2 = torch.cat((x2_1.view(-1,1,X_sol.size(2)),x2_2.view(-1,1,X_sol.size(2)),x2_3.view(-1,1,X_sol.size(2))),1)
            k2 = self._ode(x2)
              
            x3_1 = X_sol[:,0,:] + 0.5*self.dt*k2[:,0,:]
            x3_2 = X_sol[:,1,:] + 0.5*self.dt*k2[:,1,:]
            x3_3 = X_sol[:,2,:] + 0.5*self.dt*k2[:,2,:]
        
            x3 = torch.cat((x3_1.view(-1,1,X_sol.size(2)),x3_2.view(-1,1,X_sol.size(2)),x3_3.view(-1,1,X_sol.size(2))),1)
              
            k3 = self._ode(x3)
        
            x4_1 = X_sol[:,0,:] + 0.5*self.dt*k3[:,0,:]
            x4_2 = X_sol[:,1,:] + 0.5*self.dt*k3[:,1,:]
            x4_3 = X_sol[:,2,:] + 0.5*self.dt*k3[:,2,:]
            
            x4 = torch.cat((x4_1.view(-1,1,X_sol.size(2)),x4_2.view(-1,1,X_sol.size(2)),x4_3.view(-1,1,X_sol.size(2))),1)
              
            k4 = self._ode(x4)
            
            dx = (k1 + 2.*k2 + 2.*k3 + k4)
        
            xnew = X_sol + (1/6.)*self.dt*dx
        
            x_return = torch.cat((X_sol[:,:,0].view(-1,3,1),xnew),2)
                  
            return x_return
        
        def _scaling(self,x,dim):
            x_1 = x[:,0,:]
            x_2 = torch.mul((1/dim),x[:,1,:])
            x_3 = torch.mul((1/dim),x[:,2,:])
            X = torch.cat((x_1.view(-1,1,x.size(2)),x_2.view(-1,1,x.size(2)),x_3.view(-1,1,x.size(2))),1)
            return X

        def _rescaling(self,x,dim):
            x_1 = x[:,0,:]
            x_2 = torch.mul(dim,x[:,1,:])
            x_3 = torch.mul(dim,x[:,2,:])
            X = torch.cat((x_1.view(-1,1,x.size(2)),x_2.view(-1,1,x.size(2)),x_3.view(-1,1,x.size(2))),1)
            return X
      
        def forward(self, x):
            X = self.stdTr * x.view(x.size(0),3,int(shapeData[1]/step_calcul))
            X = X + self.meanTr
            
            x_init = self._scaling(X, dim)
            
            xpred = self._RK4Solver(x_init[:,:,:-1])

            xnew = self._rescaling(xpred, dim)

            x_result = xnew - self.meanTr
            x_result_final = x_result / self.stdTr

            #xnew  = torch.cat((x[:,:,0].view(-1,x.size(1),1),xpred),dim=2)
            
            #xnew = xnew.view(-1,x.size(1),x.size(2),1)

            return x_result_final.view(-1,3,x.size(2),1)

elif flagAEType == 'unet': ## Conv model with no use of the central point
  dW = 5
  class Phi_r(torch.nn.Module):
      def __init__(self):
          super(Phi_r, self).__init__()
          self.pool1  = torch.nn.AvgPool2d((10,1)) #8 10
          self.conv1  = torch.nn.Conv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.conv2  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          
          self.conv21 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv22 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv23 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.conv3  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          
          self.conv2Tr = torch.nn.ConvTranspose2d(shapeData[0]*DimAE,shapeData[0],(10,1),stride=(10,1),bias=False)          

          self.convHR1  = torch.nn.Conv2d(shapeData[0],2*shapeData[0]*DimAE,(2*dW+1,1),padding=(dW,0),bias=False)
          self.convHR2  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          
          self.convHR21 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.convHR22 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.convHR23 = torch.nn.Conv2d(shapeData[0]*DimAE,shapeData[0]*DimAE,1,padding=0,bias=False)
          self.convHR3  = torch.nn.Conv2d(2*shapeData[0]*DimAE,shapeData[0],1,padding=0,bias=False)

      def forward(self, xinp):
          x = self.pool1( xinp )
          x = self.conv1( x )
          x = self.conv2( F.relu(x) )
          x = torch.cat((self.conv21(x), self.conv22(x) * self.conv23(x)),dim=1)
          x = self.conv3( x )
          
          x = self.conv2Tr( x )
          
          xHR = self.convHR1( xinp )
          xHR = self.convHR2( F.relu(xHR) )
          xHR = torch.cat((self.convHR21(xHR), self.convHR22(xHR) * self.convHR23(xHR)),dim=1)
          xHR = self.convHR3( xHR )
          x   = torch.add(x,xHR)
          
          x = x.view(-1,shapeData[0],int(shapeData[1]/step_calcul),1)
          return x

""" 
Observation model 
"""
class Model_H(torch.nn.Module):
    def __init__(self):
        super(Model_H, self).__init__()
        self.dim_obs = 1
        self.dim_obs_channel = np.array([shapeData[0]])

        self.DimObs = 1
        self.dimObsChannel = np.array([shapeData[0]])

    def forward(self, x, y, mask):
        dyout = (x - y) * mask
        return dyout
    
""" 
Lightning class for 4DVarNet data assimilation
"""
class HParam:
    def __init__(self):
        self.iter_update     = []
        self.nb_grad_update  = []
        self.lr_update       = []
        self.n_grad          = 1
        self.dim_grad_solver = 10
        self.dropout         = 0.25
        self.w_loss          = []
        self.automatic_optimization = True

        self.alpha_proj    = 0.5
        self.alpha_mse = 10.

        self.k_batch = 1
        
"""
4DVarNet
"""

class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3,padding_mode='zeros'):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding,padding_mode=padding_mode)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

class ConvLSTM1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3,padding_mode='zeros'):
        super(ConvLSTM1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv1d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding,padding_mode=padding_mode)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(device),
                torch.autograd.Variable(torch.zeros(state_size)).to(device)
            )

        # prev_state has two components
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell


# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()
 
    def forward(self,x,w,eps=0.):
        loss_ = torch.nansum( x**2 , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_


# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class model_GradUpdateLSTM(torch.nn.Module):
    def __init__(self,ShapeData,periodicBnd=False,DimLSTM=0,rateDropout=0.,padding_mode='zeros'):
        super(model_GradUpdateLSTM, self).__init__()

        with torch.no_grad():
            self.shape     = ShapeData
            if DimLSTM == 0 :
                self.DimState  = 5*self.shape[0]
            else :
                self.DimState  = DimLSTM
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False

        self.convLayer     = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)

        self.dropout = torch.nn.Dropout(rateDropout)

        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0],self.DimState,3,padding_mode=padding_mode)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0],self.DimState,3,padding_mode=padding_mode)

    def _make_ConvGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.DimState, self.shape[0], 1, padding=0,bias=False))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(torch.nn.Conv2d(self.DimState, self.shape[0], (1,1), padding=0,bias=False))

        return torch.nn.Sequential(*layers)
    
    def _make_LSTMGrad(self):
        layers = []

        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0],self.DimState,3))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(ConvLSTM2d(self.shape[0],self.DimState,3))

        return torch.nn.Sequential(*layers)

    def forward(self,hidden,cell,grad,gradnorm=1.0):

        # compute gradient
        grad  = grad / gradnorm
        grad  = self.dropout( grad )

        if self.PeriodicBnd == True :
            dB     = 7
            #
            grad_  = torch.cat((grad[:,:,x.size(2)-dB:,:],grad,grad[:,:,0:dB,:]),dim=2)
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_,None)
            else:
                hidden_  = torch.cat((hidden[:,:,x.size(2)-dB:,:],hidden,hidden[:,:,0:dB,:]),dim=2)
                cell_    = torch.cat((cell[:,:,x.size(2)-dB:,:],cell,cell[:,:,0:dB,:]),dim=2)
                hidden_,cell_ = self.lstm(grad_,[hidden_,cell_])

            hidden_ = hidden_[:,:,dB:x.size(2)+dB,:]
            cell_   = cell_[:,:,dB:x.size(2)+dB,:]
        else:
            if hidden is None:
                hidden_,cell_ = self.lstm(grad,None)
            else:
                hidden_,cell_ = self.lstm(grad,[hidden,cell])

        grad = self.dropout( hidden_ )
        grad = self.convLayer( grad )

        return grad,hidden_,cell_


# New module for the definition/computation of the variational cost
class Model_Var_Cost(nn.Module):
    def __init__(self ,m_NormObs, m_NormPhi, ShapeData,DimObs=1,dimObsChannel=0,dimState=0):
        super(Model_Var_Cost, self).__init__()
        self.dimObsChannel = dimObsChannel
        self.DimObs        = DimObs
        if dimState > 0 :
            self.DimState      = dimState
        else:
            self.DimState      = ShapeData[0]
            
        # parameters for variational cost
        self.alphaObs    = torch.nn.Parameter(torch.Tensor(0.1 * np.ones((self.DimObs,1))))
        self.alphaReg    = torch.nn.Parameter(torch.Tensor([0.9]))
        #self.log("alphaObs", self.alphaObs , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("alphaReg", self.alphaReg , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.dimObsChannel[0] == 0 :
            self.WObs           = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs,ShapeData[0]))))
            self.dimObsChannel  = ShapeData[0] * np.ones((self.DimObs,))
        else:
            self.WObs            = torch.nn.Parameter(torch.Tensor(np.ones((self.DimObs,np.max(self.dimObsChannel)))))
        self.WReg    = torch.nn.Parameter(torch.Tensor(np.ones(self.DimState,)))
        self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.DimObs,))))
        self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))
        
        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi
        
    def forward(self, dx, dy):

        loss = self.alphaReg**2 * self.normPrior(dx,self.WReg**2,self.epsReg)
                
        if self.DimObs == 1 :
            loss +=  self.alphaObs[0]**2 * self.normObs(dy,self.WObs[0,:]**2,self.epsObs[0])
        else:
            for kk in range(0,self.DimObs):
                loss +=  self.alphaObs[kk]**2 * self.normObs(dy[kk],self.WObs[kk,0:dy[kk].size(1)]**2,self.epsObs[kk])

        return loss, self.alphaReg, self.alphaObs

    
# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters 
# (default norm (None) refers to the L2 norm)
# updated inner models to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    def __init__(self ,phi_r,mod_H, m_Grad, m_NormObs, m_NormPhi, ShapeData,n_iter_grad,eps=0.):
        super(Solver_Grad_4DVarNN, self).__init__()
        self.phi_r         = phi_r
        
        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        if m_NormPhi == None:    
            m_NormPhi = Model_WeightedL2Norm()
            
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, m_NormPhi, ShapeData,mod_H.DimObs,mod_H.dimObsChannel)
        
        self.eps = eps
        
        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
        
    def forward(self, x, yobs, mask, hidden = None , cell = None, normgrad = 0.):
        return self.solve(
            x_0=x,
            obs=yobs,
            mask = mask,
            hidden = hidden , 
            cell = cell, 
            normgrad = normgrad)

    def solve(self, x_0, obs, mask, hidden = None , cell = None, normgrad = 0.):
        x_k = torch.mul(x_0,1.) 
        hidden_ = hidden
        cell_ = cell 
        normgrad_ = normgrad
        
        for _ in range(self.n_grad):
            x_k_plus_1, hidden_, cell_, normgrad_, var_cost, alpha_reg, alpha_obs = self.solver_step(x_k, obs, mask,hidden_, cell_, normgrad_)

            x_k = torch.mul(x_k_plus_1,1.)

        return x_k_plus_1, hidden_, cell_, normgrad_, var_cost, alpha_reg, alpha_obs

    def solver_step(self, x_k, obs, mask, hidden, cell,normgrad = 0.):
        var_cost, var_cost_grad, alpha_reg, alpha_obs = self.var_cost(x_k, obs, mask)
        if normgrad == 0. :
            normgrad_= torch.sqrt( torch.mean( var_cost_grad**2 + self.eps ) )
        else:
            normgrad_= normgrad

        grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        grad *= 1./ self.n_grad
        x_k_plus_1 = x_k - grad
        return x_k_plus_1, hidden, cell, normgrad_ , var_cost, alpha_reg, alpha_obs

    def var_cost(self , x, yobs, mask):
        dy = self.model_H(x,yobs,mask)
        dx = x - self.phi_r.forward(x)
        
        loss, alpha_reg, alpha_obs = self.model_VarCost( dx , dy )
        
        var_cost_grad = torch.autograd.grad(loss, x, create_graph=True)[0]
        return loss, var_cost_grad, alpha_reg, alpha_obs
    
EPS_NORM_GRAD = 0. * 1.e-20  
class LitModel(pl.LightningModule):
    def __init__(self,conf=HParam(),*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # hyperparameters
        self.hparams.iter_update     = [0, 20, 50, 70, 100, 150, 800]  # [0,2,4,6,9,15]
        self.hparams.nb_grad_update  = [5, 5, 10, 10, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
        self.hparams.lr_update       = [1e-3, 1e-4, 1e-4, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        
        self.hparams.n_grad          = self.hparams.nb_grad_update[0]
        self.hparams.k_n_grad        = 1
        self.hparams.dim_grad_solver = dimGradSolver
        self.hparams.dropout         = rateDropout
        
        self.hparams.k_batch         = 1
        
        self.hparams.alpha_prior    = 0.5
        self.hparams.alpha_mse = 1.e1        

        self.hparams.w_loss          = torch.nn.Parameter(torch.Tensor(w_loss), requires_grad=False)
        self.hparams.automatic_optimization = False#True#

        # main model
        self.model        = Solver_Grad_4DVarNN(Phi_r(), 
                                                            Model_H(), 
                                                            model_GradUpdateLSTM(shapeData, UsePeriodicBoundary, self.hparams.dim_grad_solver, self.hparams.dropout), 
                                                            None, None, shapeData, self.hparams.n_grad, EPS_NORM_GRAD)#, self.hparams.eps_norm_grad)
        self.w_loss       = self.hparams.w_loss # duplicate for automatic upload to gpu
        self.x_rec    = None # variable to store output of test method
        self.x_rec_obs = None
        self.curr = 0

        self.automatic_optimization = self.hparams.automatic_optimization

        #tensorboard
        #self.logger.experiment.log_metrics('{loss}')        

    def forward(self):
        return 1

    def configure_optimizers(self):
        optimizer   = optim.Adam([{'params': self.model.model_Grad.parameters(), 'lr': self.hparams.lr_update[0]},
                                      {'params': self.model.model_VarCost.parameters(), 'lr': self.hparams.lr_update[0]},
                                    {'params': self.model.phi_r.parameters(), 'lr': 0.5*self.hparams.lr_update[0]},
                                    ], lr=0.)
        return optimizer
    
    def on_epoch_start(self):
        # enforce and check some hyperparameters  
        self.model.n_grad   = self.hparams.n_grad 

    def on_train_epoch_start(self):
        self.model.n_grad   = self.hparams.n_grad 

        opt = self.optimizers()
        if (self.current_epoch in self.hparams.iter_update) & (self.current_epoch > 0):
            indx             = self.hparams.iter_update.index(self.current_epoch)
            print('... Update Iterations number/learning rate #%d: NGrad = %d -- lr = %f'%(self.current_epoch,self.hparams.nb_grad_update[indx],self.hparams.lr_update[indx]))
            
            self.hparams.n_grad = self.hparams.nb_grad_update[indx]
            self.model.n_grad   = self.hparams.n_grad 
            
            mm = 0
            lrCurrent = self.hparams.lr_update[indx]
            lr = np.array([lrCurrent,lrCurrent,0.5*lrCurrent,0.])            
            for pg in opt.param_groups:
                pg['lr'] = lr[mm]# * self.hparams.learning_rate
                mm += 1
            #self.log("lr",lr, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        #if self.current_epoch == 0 :     
        #    self.save_hyperparameters()
        
    def training_step(self, train_batch, batch_idx, optimizer_idx=0):
        opt = self.optimizers()
                    
        # compute loss and metrics
        loss, out, metrics = self.compute_loss(train_batch, phase='train')
        
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(train_batch, phase='train',batch_init=out[0],hidden=out[1],cell=out[2],normgrad=out[3])
            loss = loss + loss1
        
        # log step metric        
        #self.log('train_mse', mse)
        #self.log("dev_loss", mse / var_Tr , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("loss", loss , on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_mse_loss", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("tr_prior_loss", stdTr**2 * metrics['prior'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

           # initial grad value
        if self.hparams.automatic_optimization == False :
            # backward
            self.manual_backward(loss)
        
            if (batch_idx + 1) % self.hparams.k_batch == 0:
                # optimisation step
                opt.step()
                
                # grad initialization to zero
                opt.zero_grad()
         
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        loss, out, metrics = self.compute_loss(val_batch, phase='val')
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(val_batch, phase='val',batch_init=out[0],hidden=out[1],cell=out[2],normgrad=out[3])
            loss = loss1

        self.log('val_loss', loss)
        self.log('val_loss', stdTr**2 * metrics['mse'] )
        self.log("val_loss_mse", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss_prior", stdTr**2 * metrics['prior'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("alphaObs",metrics['alpha_obs'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("alphaReg",metrics['alpha_reg'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("var_cost",metrics['var_cost'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('var_loss_classic', metrics['var_loss_classic'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('nmse', metrics['nmse'], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss, out, metrics = self.compute_loss(test_batch, phase='test')
        
        for kk in range(0,self.hparams.k_n_grad-1):
            loss1, out, metrics = self.compute_loss(test_batch, phase='test',batch_init=out[0].detach(),hidden=out[1],cell=out[2],normgrad=out[3])


        self.log('test_loss', loss1)
        self.log("test_loss_mse", stdTr**2 * metrics['mse'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_loss_prior", stdTr**2 * metrics['prior'] , on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        #return {'preds': out_ssh.detach().cpu(),'obs_ssh': out_ssh_obs.detach().cpu()}
        return {'preds': out[0].detach().cpu()}

    def test_epoch_end(self, outputs):
        x_test_rec = torch.cat([chunk['preds'] for chunk in outputs]).numpy()
        x_test_rec = stdTr * x_test_rec + meanTr        
        self.x_rec = x_test_rec.squeeze()

        return [{'mse':0.,'preds': 0.}]

    def compute_loss(self, batch, phase, batch_init = None , hidden = None , cell = None , normgrad = 0.0):
        
        inputs_init_,inputs_obs,masks,targets_GT = batch        

        #inputs_init = inputs_init_
        if batch_init is None :
            inputs_init = inputs_init_
        else:
            inputs_init = batch_init

        if phase == 'train' :                
            inputs_init = inputs_init.detach()
            
        with torch.set_grad_enabled(True):
            # with torch.set_grad_enabled(phase == 'train'):
            inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

            outputs, hidden_new, cell_new, normgrad_, var_cost, alpha_reg, alpha_obs = self.model(inputs_init, inputs_obs, masks, hidden = hidden , cell = cell , normgrad = normgrad )

            loss_mse = torch.mean((outputs[100:2400] - targets_GT[100:2400]) ** 2) #[100:2400]
            loss_nmse = loss_mse/torch.var(targets_GT)
            loss_prior = torch.mean((self.model.phi_r(outputs)[100:2400] - outputs[100:2400]) ** 2)
            loss_prior_gt = torch.mean((self.model.phi_r(targets_GT)[100:2400] - targets_GT[100:2400]) ** 2)

            #loss temporal derivative
            deriv = ((targets_GT[:,:,1:]-targets_GT[:,:,:-1])-(outputs[:,:,1:]-outputs[:,:,:-1]))**2
            loss_derivative=torch.mean(deriv)
            
            loss = self.hparams.alpha_mse * loss_mse
            loss += 0.5 * self.hparams.alpha_prior * (loss_prior + loss_prior_gt)
            loss += 0.5 * loss_derivative

            #4DVar classic loss
            var_loss = (1/(3*2500))* 10000000. * torch.sum( (self.model.phi_r(outputs[:] - outputs[:] )**2 ))
            R_values = np.array([1/3.155388832715881e-06,1/6.513491970576553e-05,1/5.488307566070105e-05])
            R = np.zeros((inputs_obs.shape[0],3,250))
            H = np.zeros((inputs_obs.shape[0],3,250))
            for i in range(inputs_obs.shape[0]):
                for t in range(0,250,5):
                    H[i, :, t] = [1, 1, 1]
                    R[i,:,t] = R_values
            H = np.resize(H,(inputs_obs.shape[0],3,250,1))
            R = np.resize(R, (inputs_obs.shape[0],3,250,1))
            R_covdiag_inv = torch.from_numpy(R).to(device)
            H=torch.from_numpy(H).to(device)
            
            var_loss += (1/(3*2500))* 1. * torch.sum( R_covdiag_inv[:] * (inputs_obs[:] - outputs[:])**2 * H[:])
            var_loss += (1/3)* 100. * torch.sum( H[:,:,0,:] * (targets_GT[:,:,0] - outputs[:,:,0])**2 )

            #var_loss=0.
            # metrics
            prior = loss_prior.detach()
            mse       = loss_mse.detach()
            nmse = loss_nmse.detach()
            metrics   = dict([('var_loss_classic', var_loss),('mse',mse),('nmse',nmse),('prior',prior), ('var_cost', var_cost), ('alpha_obs',alpha_obs),('alpha_reg',alpha_reg)])
            #print(mse.cpu().detach().numpy())
            if (phase == 'val') or (phase == 'test'):                
                outputs = outputs.detach()
        
        out = [outputs,hidden_new, cell_new, normgrad_]
        
        return loss,out, metrics

"""
MODELE TRAINING
"""
#pytorch dataloaders
UsePeriodicBoundary = False # use a periodic boundary for all conv operators in the gradient model (see torch_4DVarNN_dinAE)
w_loss = np.ones(dT) / float(dT)
batch_size = 128
idx_val = x_train.shape[0]-500

training_dataset     = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init[:idx_val,:,::step_calcul,:]),torch.Tensor(x_train_obs[:idx_val,:,::step_calcul,:]),torch.Tensor(mask_train[:idx_val,:,::step_calcul,:]),torch.Tensor(x_train[:idx_val,:,::step_calcul,:])) # create your datset
val_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_train_Init[idx_val:,:,::step_calcul,:]),torch.Tensor(x_train_obs[idx_val:,:,::step_calcul,:]),torch.Tensor(mask_train[idx_val:,:,::step_calcul,:]),torch.Tensor(x_train[idx_val:,:,::step_calcul,:])) # create your datset
test_dataset         = torch.utils.data.TensorDataset(torch.Tensor(x_test_Init[:,:,::step_calcul,:]),torch.Tensor(x_test_obs[:,:,::step_calcul,:]),torch.Tensor(mask_test[:,:,::step_calcul,:]),torch.Tensor(x_test[:,:,::step_calcul,:])) # create your datset

dataloaders = {
    'train': torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    'test': torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
}            
dataset_sizes = {'train': len(training_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
print(x_train_Init[idx_val:,:,::step_calcul,:].shape)

print(device)

# Training from scratch
dimGradSolver = 25
rateDropout = 0.2
mod = LitModel()

mod.hparams.n_grad          = 5
mod.hparams.k_n_grad        = 2
mod.hparams.iter_update     = [0, 100, 200, 300, 500, 700, 800]  # [0,2,4,6,9,15]
mod.hparams.nb_grad_update  = [5, 5, 5, 5, 15, 15, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
mod.hparams.lr_update       = [1e-3, 1e-4, 1e-5, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]
        
mod.hparams.alpha_prior = 0.1
mod.hparams.alpha_mse = 1.

profiler_kwargs = {'max_epochs': epoch }

suffix_exp = 'exp%02d'%flagTypeMissData
filename_chkpt = directory + '/checkpoint_varcost_ode/model-l63-'

filename_chkpt = filename_chkpt+flagAEType+'-'  
    
filename_chkpt = filename_chkpt +suffix_exp+'-sampling'+str(num_experiment)
filename_chkpt = filename_chkpt+'-igrad%02d_%02d'%(mod.hparams.n_grad,mod.hparams.k_n_grad)+'-dgrad%d'%dimGradSolver
filename_chkpt = filename_chkpt+'-drop%02d'%(100*rateDropout)

checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath= './resL63/'+suffix_exp,
                                      filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                      save_top_k=5,
                                      mode='min')

logger = TensorBoardLogger(name=directory+"/training_process_varcost_ode/", save_dir=directory, log_graph=True)
trainer = pl.Trainer(gpus=1,  **profiler_kwargs,callbacks=[checkpoint_callback], logger=logger)
trainer.fit(mod, dataloaders['train'], dataloaders['val'])


# Fine-tuning from pre-trained model (checkpoint)

pathCheckPOint = directory+'/checkpoint_varcost_ode/'+ os.listdir(directory+'/checkpoint_varcost_ode/')[-1]
print('.... load pre-trained model :'+pathCheckPOint)
mod = LitModel.load_from_checkpoint(pathCheckPOint)

mod.hparams.n_grad          = 5
mod.hparams.k_n_grad        = 2
mod.hparams.iter_update     = [0, 100, 200, 300, 500, 700, 800]  # [0,2,4,6,9,a15]
mod.hparams.nb_grad_update  = [5, 5, 5, 5, 5, 5, 20, 20, 20]  # [0,0,1,2,3,3]#[0,2,2,4,5,5]#
mod.hparams.lr_update       = [1e-4, 1e-5, 1e-6, 1e-5, 1e-4, 1e-5, 1e-5, 1e-6, 1e-7]

mod.hparams.alpha_prior = 0.1
mod.hparams.alpha_mse = 1.

profiler_kwargs = {'max_epochs': 150 }

suffix_exp = 'exp%02d'%flagTypeMissData
filename_chkpt = 'model-l63-'

filename_chkpt = filename_chkpt + suffix_exp
filename_chkpt = filename_chkpt+'-igrad%02d_%02d'%(mod.hparams.n_grad,mod.hparams.k_n_grad)+'-dgrad%d'%dimGradSolver
filename_chkpt = filename_chkpt+'-drop_%02d'%(100*rateDropout)

checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                      dirpath= './resL63/'+suffix_exp,
                                      filename= filename_chkpt + '-{epoch:02d}-{val_loss:.2f}',
                                      save_top_k=3,
                                      mode='min')
trainer = pl.Trainer(gpus=1,  **profiler_kwargs,callbacks=[checkpoint_callback],logger=logger)
trainer.fit(mod, dataloaders['train'], dataloaders['val'])

"""
MODEL TESTING

"""
dimGradSolver = 25
rateDropout = 0.2

pathCheckPOint = directory+'/checkpoint_varcost_ode/'+ os.listdir(directory+'/checkpoint_varcost_ode/')[-1]
print('.... load pre-trained model :'+pathCheckPOint)

mod = LitModel.load_from_checkpoint(pathCheckPOint)            

#print(mod.hparams)
mod.hparams.n_grad = 5
mod.hparams.k_n_grad = 2

print(' Ngrad = %d / %d'%(mod.hparams.n_grad,mod.model.n_grad))
#trainer = pl.Trainer(gpus=1, accelerator = "ddp", **profiler_kwargs)

profiler_kwargs = {'max_epochs': 1}
trainer = pl.Trainer(gpus=1,  **profiler_kwargs)

############################################################
# metrics for validation dataset
trainer.test(mod, dataloaders=dataloaders['val'])

X_val = X_train[idx_val:,:,::step_calcul]
mask_val = mask_train[idx_val:,:,::step_calcul,:].squeeze()
var_val  = np.mean( (X_val - np.mean(X_val,axis=0))**2 )
mse = np.mean( (mod.x_rec-X_val) **2 ) 
mse_i   = np.mean( (1.-mask_val.squeeze()) * (mod.x_rec-X_val) **2 ) / np.mean( (1.-mask_val) )
mse_r   = np.mean( mask_val.squeeze() * (mod.x_rec-X_val) **2 ) / np.mean( mask_val )

nmse = mse / var_val
nmse_i = mse_i / var_val
nmse_r = mse_r / var_val

print("..... Assimilation performance (validation data)")
print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))

trainer.test(mod, dataloaders=dataloaders['test'])

############################################################
# metrics for test dataset
var_test  = np.mean( (X_test[:,:,::step_calcul] - np.mean(X_test[:,:,::step_calcul] ,axis=0))**2 )
mask_test = mask_test[:,:,::step_calcul,:].squeeze()
mse = np.mean( (mod.x_rec-X_test[:,:,::step_calcul] ) **2 ) 
mse_i   = np.mean( (1.-mask_test.squeeze()) * (mod.x_rec-X_test[:,:,::step_calcul]) **2 ) / np.mean( (1.-mask_test) )
mse_r   = np.mean( mask_test.squeeze() * (mod.x_rec-X_test[:,:,::step_calcul]) **2 ) / np.mean( mask_test )

nmse = mse / var_test
nmse_i = mse_i / var_test
nmse_r = mse_r / var_test

print("..... Assimilation performance (test data)")
print(".. MSE ALL.   : %.3f / %.3f"%(mse,nmse))
print(".. MSE ObsData: %.3f / %.3f"%(mse_r,nmse_r))
print(".. MSE Interp : %.3f / %.3f"%(mse_i,nmse_i))

"""
RESULT SAVING
""" 

mod.x_rec = mod.x_rec.squeeze()
x_gt  = X_test[:,:,:]
x_rec = mod.x_rec[:,:]
y_obs = X_test_obs[:,:,:]

np.save(directory + results_name, x_rec)
