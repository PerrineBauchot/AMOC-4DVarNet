#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:24:09 2022

@author: bauchope
"""
import numpy as np
import matplotlib.pyplot as plt 
import torch
from scipy.integrate import odeint

"""
Parameters
"""

directory='/home5/bauchope/Documents/DOCTORAT/'
dT_obs = 50 #sampling period in yrs
flagTypeMissData = 1 #sampling strategy (1: every variable is observed; 2: only x1 is observed; 3: only (x2, x3) are observed )
step_calcul = 10 #undersampling factor to speed up the computation

#device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constants
ssf = 35/1000 #physical constant
l = 0.01 #friction coefficient
K = 10**(-4) #diffusion coefficient
omega0 = -2.5*10**(-2) #thermally and wind-driven circulation component
epsilon = 0.35 #buoyancy coefficient
beta = 7*10**(-4) #haline contraction coefficient
dim = beta*epsilon/l #scaling coefficient
dT=2500 #duration of a simulated trajectory 

"""
Phi_r operator: physical model used for data assimilation
"""

class Phi_r(torch.nn.Module):
  def __init__(self):
    super(Phi_r, self).__init__()
    self.l = torch.nn.Parameter(torch.Tensor([l]))
    self.epsilon    = torch.nn.Parameter(torch.Tensor([epsilon]))
    self.beta   = torch.nn.Parameter(torch.Tensor([beta]))
    self.omega0   = torch.nn.Parameter(torch.Tensor([omega0]))
    self.K   = torch.nn.Parameter(torch.Tensor([K]))
    self.ssf   = torch.nn.Parameter(torch.Tensor([ssf]))

    self.dt        = 1.
    self.IntScheme = 'rk4'
    self.dim = dim      
           
  def _ode(self, xin):
    dx_1 = -self.l*xin[:,:,0]- self.epsilon * self.beta * xin[:,:,2]
    dx_2 = ( self.omega0 + xin[:,:,0] ) * xin[:,:,2] - self.K * xin[:,:,1]+self.ssf
    dx_3 = -( self.omega0 + xin[:,:,0] ) * xin[:,:,1] - self.K * xin[:,:,2]

    return torch.stack((dx_1, dx_2, dx_3),2)

  def _RK4Solver(self, X_sol):

    k1 = self._ode(X_sol[:,:,:])
    x2_1 = X_sol[:,:,0] + 0.5*self.dt*k1[:,:,0]
    x2_2 = X_sol[:,:,1] + 0.5*self.dt*k1[:,:,1]
    x2_3 = X_sol[:,:,2] + 0.5*self.dt*k1[:,:,2]
     
    x2 = torch.stack((x2_1,x2_2,x2_3),2)
    k2 = self._ode(x2)
      
    x3_1 = X_sol[:,:,0] + 0.5*self.dt*k2[:,:,0]
    x3_2 = X_sol[:,:,1] + 0.5*self.dt*k2[:,:,1]
    x3_3 = X_sol[:,:,2] + 0.5*self.dt*k2[:,:,2]

    x3 = torch.stack((x3_1,x3_2,x3_3),2)
      
    k3 = self._ode(x3)

    x4_1 = X_sol[:,:,0] + self.dt*k3[:,:,0]
    x4_2 = X_sol[:,:,1] + self.dt*k3[:,:,1]
    x4_3 = X_sol[:,:,2] + self.dt*k3[:,:,2]
    
    x4 = torch.stack((x4_1,x4_2,x4_3),2)
      
    k4 = self._ode(x4)
    
    dx = (k1 + 2.*k2 + 2.*k3 + k4)

    xnew = X_sol + (1/6.)*self.dt*dx

    return xnew

  def _scaling(self,x):
      x_1 = x[:,:,0]
      x_2 = x[:,:,1] * (1./self.dim)
      x_3 = torch.mul((1./self.dim),x[:,:,2])
      X = torch.stack((x_1,x_2,x_3),2)
      return X

  def _rescaling(self,x):
      x_1 = x[:,:,0]
      x_2 = torch.mul(self.dim,x[:,:,1])
      x_3 = torch.mul(self.dim,x[:,:,2])

      X = torch.stack((x_1,x_2,x_3),2)
      return X
      
  def forward(self, x):
      
      x_init =  self._scaling(x)

      xpred = self._RK4Solver(x_init[:,:-1,:])

      xpred = torch.cat((x_init[:,0,:].view(100,1,3),xpred),1)

      xnew = self._rescaling(xpred)
      xnew = xnew.view(100, x.size(1),x.size(2))

      return xnew

"""
INITIALISATION and DATA GENERATION
"""

#Physical model to generate data by an odeint integration
def model(V, t, ssf, l, K, omega0, epsilon, beta):
  """
  V : vecteur of 3 variables to model
  l, K, omega0, epsilon, beta : parameters of the model
  ssf : surface salt flux (ssf=F0*S0/h with F0 the freshwater flux intensity; S0 the reference salinity; h the loop thickness)
  """
  x_1 = -l*V[0] - epsilon*beta*V[2]
  x_2 = (omega0+V[0])*V[2] - K*V[1] + ssf
  x_3 = -(omega0+V[0])*V[1] - K*V[2]
  dV = np.array([x_1, x_2, x_3])
  return dV

y0 = np.array([0.,0.,0.]) #initial conditions
tt = np.arange(0.,100000.,1) #temporal series
S_init = odeint(model,y0,tt,args=(ssf,l,K,omega0,epsilon,beta)) #initial integration
S_test=np.zeros((100,100000,3)) #tensor on which data assimilation will be performed
list_CI_test = np.linspace(65000,95000,100) #initial conditions for S_test extracted from the initial integration S_init to ensure the dataset variety

plt.figure(dpi=300)
plt.subplot(311)
plt.plot(tt, S_init[:,0], c='black', linewidth=0.5)
plt.ylabel('$x_1 (yr^{-1})$')
plt.subplot(312)
plt.plot(tt, S_init[:,1], c='black', linewidth=0.5)
plt.ylabel('$x_2 (yr^{-1})$')
plt.subplot(313)
plt.plot(tt, S_init[:,2], c='black', linewidth=0.5)
plt.ylabel('$x_3 (yr^{-1})$')
plt.xlabel('Time (yrs)')
plt.tight_layout()

#%%

#FILLING S_test as a basis to build our databases of observations and ground truth 
for i in range(100):
    k_test = int(list_CI_test[i])
    y0_test=S_init[k_test,:]
    S_temp_test = odeint(model,y0_test,tt,args=(ssf,l,K,omega0,epsilon,beta))
    S_temp_test[:,1]=dim*S_temp_test[:,1]
    S_temp_test[:,2]=dim*S_temp_test[:,2]
    S_test[i,:,:]=S_temp_test[:,:]

#Out of the complete time series, we select the trajectories of the test time period to align the performances computation with the 4DVarNet results 
S = np.zeros((100,dT,3))
t_depart=65000
for i in range(100):
    t_fin=t_depart+dT
    S[i,:,:] = S_test[i,t_depart:t_fin:1,:]
    t_depart=t_depart+100

#%%% Checking of the data simulation
plt.figure(1)
plt.plot(S[0,:,0], label='model')
plt.title('x1')

plt.figure(2)
plt.plot(S[0,:,1], label='model')
plt.legend()
plt.title('x2')

plt.figure(3)
plt.plot(S[0,:,2], label='model')
plt.legend()
plt.title('x3')    

#%%%    
"""
OBSERVATIONS
"""

#Observation noise as a 10% ratio of the variance of the small-scale oscillations of each variable

var_bruit_1 = 0.1*np.var(S[0,900:2000,0])
var_bruit_2 = 0.1*np.var(S[0,900:2000,1])
var_bruit_3 = 0.1*np.var(S[0,900:2000,2])

bruit_1 = np.random.normal(0,np.sqrt(var_bruit_1),dT) 
bruit_2 = np.random.normal(0,np.sqrt(var_bruit_2),dT)
bruit_3 = np.random.normal(0,np.sqrt(var_bruit_3),dT)

#Retrieving of observation points depending on the chosen sampling strategy and on the sampling period
#REGULAR
obs = np.zeros(S.shape)
if flagTypeMissData == 1: #3 observed variables with noise
  obs[:,::dT_obs,0]=S[:,::dT_obs,0] + bruit_1[::dT_obs]
  obs[:,::dT_obs,1]=S[:,::dT_obs,1] + bruit_2[::dT_obs]
  obs[:,::dT_obs,2]=S[:,::dT_obs,2] + bruit_3[::dT_obs]
elif flagTypeMissData == 2: #only x1 is observed with noise
  obs[:,::dT_obs,0]=S[:,::dT_obs,0] + bruit_1[::dT_obs]
elif flagTypeMissData == 3: #only x2 et x3 are observed with noise
  obs[:,::dT_obs,1]=S[:,::dT_obs,1] + bruit_2[::dT_obs]
  obs[:,::dT_obs,2]=S[:,::dT_obs, 2] + bruit_3[::dT_obs]
  
"""
#REGULAR CLUSTER
obs = np.zeros(S.shape)
if flagTypeMissData == 1: #3 observed variables with noise
  obs[:,:-20:dT_obs,0]=S[:,:-20:dT_obs,0] + bruit_1[:-20:dT_obs]
  obs[:,:-20:dT_obs,1]=S[:,:-20:dT_obs,1] + bruit_2[:-20:dT_obs]
  obs[:,:-20:dT_obs,2]=S[:,:-20:dT_obs,2] + bruit_3[:-20:dT_obs]
  obs[:,10:-10:dT_obs,0]=S[:,10:-10:dT_obs,0] + bruit_1[10:-10:dT_obs]
  obs[:,10:-10:dT_obs,1]=S[:,10:-10:dT_obs,1] + bruit_2[10:-10:dT_obs]
  obs[:,10:-10:dT_obs,2]=S[:,10:-10:dT_obs,2] + bruit_3[10:-10:dT_obs]
  obs[:,20::dT_obs,0]=S[:,20::dT_obs,0] + bruit_1[20::dT_obs]
  obs[:,20::dT_obs,1]=S[:,20::dT_obs,1] + bruit_2[20::dT_obs]
  obs[:,20::dT_obs,2]=S[:,20::dT_obs,2] + bruit_3[20::dT_obs]
  
#RANDOM
obs = np.zeros(S.shape)
if flagTypeMissData == 1: #3 observed variables with noise
    for j in range(100):
        points_obs_train = random.sample(range(dT)[::step_calcul], int(dT//time_step_obs))
        for k in points_obs_train:
            obs[j,k,0] = S[j,k,0]+ bruit_1[k]
            obs[j,k,1] = S[j,k,1]+ bruit_2[k]
            obs[j,k,2] = S[j,k,2]+ bruit_3[k]
  
#RANDOM CLUSTER
obs = np.zeros(S.shape)
if flagTypeMissData == 1: #3 observed variables with noise
    for j in range(100):
        points_obs_test = random.sample(range(10,dT-10)[::step_calcul], int(dT//time_step_obs))
        for k in points_obs_test:
            obs[j,k-10,0] = S[j,k-10,0]+bruit_1[k-10]
            obs[j,k-10,1] = S[j,k-10,1]+bruit_2[k-10]
            obs[j,k-10,2] = S[j,k-10,2]+bruit_3[k-10]
            obs[j,k,0] = S[j,k,0]+bruit_1[k]
            obs[j,k,1] = S[j,k,1]+bruit_2[k]
            obs[j,k,2] = S[j,k,2]+bruit_3[k]
            obs[j,k+10,0] = S[j,k+10,0]+bruit_1[k+10]
            obs[j,k+10,1] = S[j,k+10,1]+bruit_2[k+10]
            obs[j,k+10,2] = S[j,k+10,2]+bruit_3[k+10]
"""

obs_points = np.copy(obs)
time=np.arange(0,dT,1)

"""
Interpolation of observation points to get an initial condition based on observations for data assimilation for the analysed state X
"""

for i in range(100):
  state = obs
  if flagTypeMissData==1:
    for k in range(3):
      data_pts = np.where(obs[i,:,k]!=0)[0]
      y1 = obs[i,data_pts,k]
      x_1 = np.interp(time[:],time[data_pts],y1)
      state[i,:,k] = x_1[:]

  elif flagTypeMissData==2:
    data_pts = np.where(obs[i,:,0]!=0)[0]
    y1 = obs[i,data_pts,0]
    x_1 = np.interp(time[:],time[data_pts],y1)
    state[i,:,0] = x_1[:]
    state[i,:,1] = S[i,:,1]+10*bruit_2 #noisy initialisation of the unobserved variables
    state[i,:,2] = S[i,:,2]+10*bruit_3

  else:
    for k in range(3):
      if k==0:
        state[i,:,0] = S[i,:,0]+10*bruit_1 #noisy initialisation of the unobserved varible
      else:
        data_pts = np.where(obs[i,:,k]!=0)[0]
        y1 = obs[i,data_pts,k]
        x_1 = np.interp(time[:],time[data_pts],y1)
        state[i,:,k] = x_1[:]


"""
Tensors formatting
"""

XGT = np.zeros((100,dT,3)) #ground truth
Yobs = np.zeros((100,dT,3)) #observations
X = np.zeros((100,dT,3)) #analysed state
H = np.zeros((100,dT,3)) #observation operator (mask)

for i in range(100):
  XGT[i,:,:] = S[i,:,:]
  Yobs[i,:dT,:] = obs_points[i,:dT,:]
  X[i,:,:] = state[i,:,:]
  if flagTypeMissData==1:
    H[i,:dT:dT_obs,:] = [1,1,1]
  elif flagTypeMissData==2:
    H[i,:dT:dT_obs,:] = [1,0,0]
  else:
    H[i,:dT:dT_obs,:] = [0,1,1]

#conversion to pytorch framework
XGT_torch = torch.from_numpy(XGT)
YObs_torch = torch.from_numpy(Yobs)
X_torch = torch.from_numpy(X)
H=torch.from_numpy(H)
Time = torch.from_numpy(time)

#%%% Checking of the formating of our data
plt.figure()
plt.subplot(311)
plt.plot(X_torch[0,:,0], label='initial condition')
plt.plot(YObs_torch[0,:,0],'.', label='obs')
plt.plot(XGT_torch[0,:,0], label='ground truth')
#plt.legend()
plt.title('x1')

plt.subplot(312)
plt.plot(X_torch[0,:,1], label='initial condition')
plt.plot(YObs_torch[0,:,1],'.', label='obs')
plt.plot(XGT_torch[0,:,1], label='ground truth')
plt.legend()
plt.title('x2')

plt.subplot(313)
plt.plot(X_torch[0,:,2], label='initial condition')
plt.plot(YObs_torch[0,:,2],'.', label='obs')
plt.plot(XGT_torch[0,:,2], label='ground truth')
#plt.legend()
plt.title('x3')
plt.tight_layout()

#%%%
"""
4D-variational Data Assimilation
"""

step_calcul=1
mod = Phi_r()
X_torch_result = np.zeros((100,int(dT/step_calcul),3)) #tensor in which to store the result
t0    = 0 #initial time
nmse=[] #errors of reconstruction
list_loss=[] #loss evolution 
list_varnet_loss=[]
list_varnet_ode_loss=[]
#4DVAR parameters to vary and optimise
NIter = 100 #number of iterations necessary to reach the convergence of the variational cost
delta = 0.0001 #updating facto
t_window = int(dT/step_calcul) #time window to consider to make the calculations
Nwindow = int((dT/step_calcul)-t_window)+1 

alpha_obs = 1. #weight on the observation cost
alpha_reg = 10000000. #weight on the dynamical cost
alpha_b = 100. #weight on the background cost

#Covariance matrices
R_covdiag_inv = np.array([1/var_bruit_1,1/var_bruit_2,1/var_bruit_3]) #Observation covariance matrix
R_covdiag_inv = torch.Tensor(R_covdiag_inv )
R_covdiag_inv = R_covdiag_inv.view(1,-1).repeat(t_window,1)

P_covdiag_inv = np.array([1.,1.,1.]) #Dynamical covariance matrix
P_covdiag_inv = torch.Tensor(P_covdiag_inv )
P_covdiag_inv = P_covdiag_inv.view(1,-1).repeat(t_window,1)


B_cov_inv = np.eye(3) #Background covariance matrix
B_cov_inv = torch.Tensor(np.diag(B_cov_inv))

#undersampling to speed up computation (if necessary)
X_torch = torch.autograd.Variable(X_torch[:,::step_calcul,:], requires_grad=True)#variable pour autograd via pytorch
XGT_torch = XGT_torch[:,::step_calcul,:]
X_b_torch = torch.autograd.Variable(XGT_torch[:,0,:], requires_grad=True)
YObs_torch = torch.autograd.Variable(YObs_torch[:,::step_calcul,:],requires_grad=True)#variable pour autograd via pytorch
H_torch = torch.autograd.Variable(H[:,::step_calcul,:],requires_grad=True)#variable pour autograd via pytorch
X_torch_window_init = X_torch[:,:,:]
X_torch_window = X_torch_window_init.clone()

#time window of computation
for t in range(0,Nwindow,1): 
  XGT_window = XGT_torch[:,t:t+t_window,:]
  X_torch_window = X_torch[:,t:t+t_window,:]
  YObs_window = YObs_torch[:,t:t+t_window,:]
  H_torch_window = H_torch[:,t:t+t_window,:]
  X_torch_window = torch.autograd.Variable(X_torch_window,requires_grad=True)
  
  # assimilation loop
  for iter in range(0,NIter):
    # compute losses
    with torch.set_grad_enabled(True): 
      # dynamical loss
      X_pred  = mod( X_torch_window )
      loss_dyn = (1/(3*2500))*alpha_reg * torch.sum( P_covdiag_inv * (X_pred - X_torch_window )**2 ) 

      # observation loss
      loss_obs = (1/(3*2500))*alpha_obs * torch.sum( R_covdiag_inv * (YObs_window - X_torch_window)**2 * H_torch_window ) 
      
      # background loss
      loss_background = (1/3)*alpha_b * torch.sum( B_cov_inv * (X_b_torch[:,:] - X_torch_window[:,0,:])**2 )

      # overall loss
      loss = (loss_obs + loss_dyn + loss_background)

      loss_varnet = 0.5359**2 * torch.sum((X_torch_window - X_pred)**2) + 0.4283**2 * torch.sum((YObs_window - X_torch_window)**2 * H_torch_window )
      loss_varnet_ode = 0.6417**2 * torch.sum((X_torch_window - X_pred)**2) + (-0.3387)**2 * torch.sum((YObs_window - X_torch_window)**2 * H_torch_window )

      if( np.mod(iter,10) == 0 ):
        print(".... iter %d: loss %.3f dyn_loss %.3f obs_loss %.3f backround_loss %.3f"%(iter,loss,loss_dyn,loss_obs,loss_background))  
        list_loss.append(loss.cpu().detach().numpy())
        list_varnet_loss.append(loss_varnet.cpu().detach().numpy())
        list_varnet_ode_loss.append(loss_varnet_ode.cpu().detach().numpy())
        nmse.append((torch.mean((XGT_torch[:,100:2400,:] - X_torch_window[:,100:2400,:])**2)/torch.var(XGT_torch)).detach().numpy())  
      # compute gradient w.r.t. X and update X 
      grad_X  = torch.autograd.grad(loss,X_torch_window,create_graph=True)
      grad_update = grad_X[0]
      X_torch_window = X_torch_window - delta * grad_update[:]
      X_torch_window = torch.autograd.Variable(X_torch_window, requires_grad=True)
      
  X_torch = torch.cat((X_torch[:,:t,:], X_torch_window, X_torch[:,t+t_window:]),dim=1)

X_torch_result[:,:,:] = X_torch.cpu().detach().numpy()


#%%%
#Checking the results of assimilation
plt.figure(1, figsize=(10,10))
for jj in range(0,3):
  plt.subplot(311+jj)
  plt.plot(time[::step_calcul],XGT_torch.detach().cpu().numpy()[0,:,jj], label='Modèle')
  #plt.plot(time[::],X_torch_window_init.detach().cpu().numpy()[0,:,jj], label='Initial conditions')
  plt.plot(time[::step_calcul],YObs_torch.detach().cpu().numpy()[0,:,jj],'.', label='obs')
  #plt.plot(time[::step_calcul],X_b[0,::,jj], label='Background')
  plt.plot(time[::step_calcul],X_torch_result[0,:,jj], label='4D Var', linewidth=2)
  if jj==0:
    plt.ylabel('x1 ($yr^{-1}$)')
    plt.ylim((-0.1,0.1))
  elif jj==1:
    plt.ylabel('x2 ($yr^{-1}$)')
    plt.ylim((-0.2,0.2))
  else:
    plt.ylabel('x3 ($yr^{-1}$)')
    plt.xlabel('t (yr)')
    plt.ylim((-0.2,0.2))
  plt.tight_layout()
  
plt.suptitle('fe =' +str(1/dT_obs) + '$yr^{-1}$')
plt.legend()
plt.tight_layout()
plt.suptitle('Assimilation results')
plt.tight_layout()

plt.figure()
plt.plot(list_loss[:], label='loss', color='black')
plt.ylabel("loss")
plt.xlabel('iter')
plt.legend()

print('erreur = ', np.mean((XGT_torch.detach().cpu().numpy()-X_torch_result)**2)/np.var(XGT_torch.detach().cpu().numpy()))

fig = plt.figure(figsize=(6,4), dpi=300)
plt.plot(nmse, list_loss, color='orangered', marker='.', linewidth=2, label='4DVar-classic')
plt.grid()
plt.legend()
plt.xlabel('NMSE')
plt.ylabel('Variational Cost')
plt.xlim([0.15, 0.0])#, 10**(-3), 2*10**3])
plt.show()

fig, ax = plt.subplots(1,1)
plt.plot(nmse, list_varnet_loss, color='green', label='4DVarNet-unet')
plt.plot(nmse, list_varnet_ode_loss, color='darkviolet', label='4DVarNet-ode')
plt.legend()
plt.xlabel('Average NMSE')
plt.ylabel('Variational Cost computed as in 4DVarNet')
plt.xlim([max(nmse), min(nmse)])#, 0, 0.6])
plt.show()


#%%
#SAVE THE RESULTS

np.save(directory+'var_classic_50',X_torch_result) 

