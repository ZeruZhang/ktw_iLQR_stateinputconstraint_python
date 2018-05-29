
# coding: utf-8

# In[ ]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


# In[ ]:  
class quadrotor:
    def __init__(self,name):
        self.name = name;
        self.ix = 12
        self.iu = 4
        self.delT = 0.01
        
        # physical properties
        self.m = 1
        self.Ixx = 1
        self.Iyy = 1
        self.Izz = 1
        self.g = 9.81
        self.l = 0.2
        self.C = 0.1
        
    def forwardDyn(self,x,u,idx):
     
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
            
        # state & input
        x1 = x[:,0]
        y = x[:,1]
        z = x[:,2]
        phi = x[:,3]
        theta = x[:,4]
        psi = x[:,5]
        x_dot = x[:,6]
        y_dot = x[:,7]
        z_dot = x[:,8]
        p = x[:,9]
        q = x[:,10]
        r = x[:,11]
        
        u1 = u[:,0]
        u2 = u[:,1]
        u3 = u[:,2]
        u4 = u[:,3]
        
        # u1 = (F1 + F2 + F3 + F4) / self.m
        # u2 = (F4 - F2) / self.Iyy
        # u3 = (F3 - F1) / self.Ixx
        # u4 = self.C * (F1 - F2 + F3 - F4) / self.Izz
        
        # small angle assumption
        f = np.zeros_like(x)
        f[:,0] = x_dot
        f[:,1] = y_dot
        f[:,2] = z_dot
        f[:,3] = p
        f[:,4] = q
        f[:,5] = r
        f[:,6] = u1 * ( np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi) )
        f[:,7] = u1 * ( np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi) )
        f[:,8] = u1 * ( np.cos(phi) * np.cos(theta) ) - self.g
        f[:,9] = u2 * self.l
        f[:,10] = u3 * self.l
        f[:,11] = u4
        
        return np.squeeze(x + f * self.delT)
    
    def diffDyn(self,x,u):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            # x = np.expand_dims(x,axis=0)
            # u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_nominal = self.forwardDyn(x,u,0) 
        f_change = self.forwardDyn(x_aug,u_aug,0)
        f_change = np.reshape(f_change,(N,ix+iu,ix))
        f_diff = ( f_change - np.reshape(f_nominal,(N,1,ix)) ) / h
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        return np.squeeze(fx), np.squeeze(fu)




class unicycle:
    def __init__(self,name):
        self.name = name;
        self.ix = 3
        self.iu = 2
        self.delT = 0.1
        
    def forwardDyn(self,x,u,idx):
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
     
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(x3)
        f[:,1] = v * np.sin(x3)
        f[:,2] = w
        
        return np.squeeze(x + f * self.delT)
    
    def diffDyn(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]    
        
        fx = np.zeros((N,self.ix,self.ix))
        fx[:,0,0] = 1.0
        fx[:,0,1] = 0.0
        fx[:,0,2] = - self.delT * v * np.sin(x3)
        fx[:,1,0] = 0.0
        fx[:,1,1] = 1.0
        fx[:,1,2] = self.delT * v * np.cos(x3)
        fx[:,2,0] = 0.0
        fx[:,2,1] = 0.0
        fx[:,2,2] = 1.0
        
        fu = np.zeros((N,self.ix,self.iu))
        fu[:,0,0] = self.delT * np.cos(x3)
        fu[:,0,1] = 0.0
        fu[:,1,0] = self.delT * np.sin(x3)
        fu[:,1,1] = 0.0
        fu[:,2,0] = 0.0
        fu[:,2,1] = self.delT
        
        return np.squeeze(fx) , np.squeeze(fu)
    
    def diffDynNu(self,x,u) :
      
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            # x = np.expand_dims(x,axis=0)
            # u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_nominal = self.forwardDyn(x,u,0) 
        f_change = self.forwardDyn(x_aug,u_aug,0)
        # print_np(f_change)
        f_change = np.reshape(f_change,(N,ix+iu,ix))
        # print_np(f_nominal)
        # print_np(f_change)
        f_diff = ( f_change - np.reshape(f_nominal,(N,1,ix)) ) / h
        # print_np(f_diff)
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        return np.squeeze(fx), np.squeeze(fu)
    
class fitLinearModel:
    def __init__(self,name,ix,iu,N):
        
        self.name = name;
        self.ix = ix
        self.iu = iu
        self.N = N
        self.A = np.zeros((N,ix,ix))
        self.B = np.zeros((N,ix,iu))
        self.C = np.zeros((N,ix))
        
    def update(self,x,u) :
        
        ix = self.ix
        iu = self.iu
        
        num_fit = np.size(u,axis=2)
        
        for i in range(self.N) :
            x_next = x[i+1,:,:]
            x_now = x[i,:,:]
            u_now = u[i,:,:]
            colMat = np.vstack((x_now,u_now,np.ones((1,num_fit))))
            rowMat = np.dot(x_next, np.linalg.pinv(colMat))
            self.A[i,:,:] = rowMat[:,0:ix]
            self.B[i,:,:] = rowMat[:,ix:ix+iu]
            self.C[i,:] = rowMat[:,ix+iu]
        
    def forwardDyn(self,x,u,idx):
        
        return np.squeeze( np.dot(self.A[idx,:,:],x) + np.dot(self.B[idx,:,:],u) + self.C[idx,:] )
    
    def diffDyn(self,x,u):

        # actually, this function does not require x or u as input..
        
        return self.A , self.B


    
    
    
    