
# coding: utf-8

# In[4]:
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
    # print ("Values are: \n%s" % (x))


# In[5]:
import cost
import model


# In[7]:

class iLQR:
    def __init__(self,name,horizon,maxIterIn,maxIterOut,Model,Cost):
        self.name = name
        self.model = Model
        self.cost = Cost
        self.N = horizon
        
        # cost optimization
        self.verbosity = True
        self.dlamda = 1.0
        self.lamda = 1.0
        self.lamdaFactor = 1.6
        self.lamdaMax = 1e8
        self.lamdaMin = 1e-6
        self.tolFun = 1e-7
        self.tolGrad = 1e-9
        self.maxIterIn = maxIterIn
        self.maxIterOut = maxIterOut
        self.zMin = 0
        self.last_head = True
        
        # constraints
        self.phi = np.ones(self.N+1) * 2
        self.tolConst = 1e-3
        
        # state & input & cost
        self.x0 = np.zeros(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.u = np.ones((self.N,self.model.iu))
        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N,self.model.iu))
        self.c = np.zeros(self.N+1)
        self.cnew = np.zeros(self.N+1)
        
        # variables for constraints # self.cost.ic,
        mu_ini = 0.1
        self.Mu = np.tile(np.identity(self.cost.ic),(self.N+1,1,1)) * mu_ini
        self.Munew = np.tile(np.identity(self.cost.ic),(self.N+1,1,1)) * mu_ini
        self.Mu_e = np.tile(np.identity(self.cost.ic),(self.N+1,1,1)) * mu_ini
        self.lam = np.tile(np.identity(self.cost.ic),(self.N+1,1,1)) * 0.01
        
        self.initialize()
        
    def initialize(self) :
        
        self.dV = np.zeros((1,2))

        self.Alpha = np.zeros(11)
        self.Alpha = np.power(10,np.linspace(0,-3,11))
        self.l = np.zeros((self.N,self.model.iu))
        self.L = np.zeros((self.N,self.model.iu,self.model.ix))
        self.fx = np.zeros((self.N,self.model.ix,self.model.ix))
        self.fu = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.cx = np.zeros((self.N+1,self.model.ix))
        self.cu = np.zeros((self.N,self.model.iu))
        self.cxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
        self.cxu = np.zeros((self.N,self.model.ix,self.model.iu))
        self.cuu = np.zeros((self.N,self.model.iu,self.model.iu))
        self.Vx = np.zeros((self.N+1,self.model.ix))
        self.Vxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
    
    def forward(self,x0,u,K,x,k,alpha,Mu,lam):
        # horizon
        N = self.N
        
        # x-difference
        dx = np.zeros(self.model.ix)
        
        # variable setting
        xnew = np.zeros((N+1,self.model.ix))
        unew = np.zeros((N,self.model.iu))
        cnew = np.zeros(N+1)
        # Munew = np.zeros_like(self.Mu)
        
        # initial state
        xnew[0,:] = x0
        
        # roll-out
        for i in range(N):
            dx = xnew[i,:] - x[i,:]
            unew[i,:] = u[i,:] + k[i,:] * alpha + np.dot(K[i,:,:],dx)
            xnew[i+1,:] = self.model.forwardDyn(xnew[i,:],unew[i,:],i)
            cnew[i] = self.cost.estimateCost(xnew[i,:],unew[i,:],Mu[i,:,:],lam[i,:,:])
            
        cnew[N] = self.cost.estimateCost(xnew[N,:],np.zeros(self.model.iu),Mu[N,:,:],lam[N,:,:])
        return xnew,unew,cnew
        
    def backward(self):
        diverge = False
        
        # state & input size
        ix = self.model.ix
        iu = self.model.iu
        
        # V final value
        self.Vx[self.N,:] = self.cx[self.N,:]
        self.Vxx[self.N,:,:] = self.cxx[self.N,:,:]
        
        # Q function
        Qu = np.zeros(iu)
        Qx = np.zeros(ix)
        Qux = np.zeros([iu,ix])
        Quu = np.zeros([iu,iu])
        Quu_save = np.zeros([self.N,iu,iu]) # for saving
        Quu_inv_save = np.zeros([self.N,iu,iu])
        Qxx = np.zeros([ix,ix])
        
        Vxx_reg = np.zeros([ix,ix])
        Qux_reg = np.zeros([ix,iu])
        QuuF = np.zeros([iu,iu])
        
        # open-loop gain, feedback gain
        k_i = np.zeros(iu)
        K_i = np.zeros([iu,ix])
        
        self.dV[0,0] = 0.0
        self.dV[0,1] = 0.0
        
        diverge_test = 0
        for i in range(self.N-1,-1,-1):
            # print(i)
            Qu = self.cu[i,:] + np.dot(self.fu[i,:].T, self.Vx[i+1,:])
            Qx = self.cx[i,:] + np.dot(self.fx[i,:].T, self.Vx[i+1,:])
 
            Qux = self.cxu[i,:,:].T + np.dot( np.dot(self.fu[i,:,:].T, self.Vxx[i+1,:,:]),self.fx[i,:,:])
            Quu = self.cuu[i,:,:] + np.dot( np.dot(self.fu[i,:,:].T, self.Vxx[i+1,:,:]),self.fu[i,:,:])
            Qxx = self.cxx[i,:,:] + np.dot( np.dot(self.fx[i,:,:].T, self.Vxx[i+1,:,:]),self.fx[i,:,:])
            
            Vxx_reg = self.Vxx[i+1,:,:] + self.lamda * np.identity(ix)
            Qux_reg = self.cxu[i,:,:].T + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fx[i,:,:])
            QuuF = self.cuu[i,:,:] + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fu[i,:,:]) + 0*self.lamda * np.identity(iu)
            Quu_save[i,:,:] = QuuF
            # add input constraints here!!
        
            
            # control gain      
            try:
                R = sp.linalg.cholesky(QuuF,lower=False)
            except sp.linalg.LinAlgError as err:
                diverge_test = i+1
                return diverge_test, Quu_save, Quu_inv_save
                        
            R_inv = sp.linalg.inv(R)
            QuuF_inv = np.dot(R_inv,np.transpose(R_inv))
            # Quu_inv_save[i,:,:] = np.linalg.inv(Quu)
            Quu_inv_save[i,:,:] = QuuF_inv
            k_i = - np.dot(QuuF_inv, Qu)
            K_i = - np.dot(QuuF_inv, Qux_reg)
            # print k_i, K_i
            
            # update cost-to-go approximation
            self.dV[0,0] = np.dot(k_i.T, Qu) + self.dV[0,0]
            self.dV[0,1] = 0.5*np.dot( np.dot(k_i.T, Quu), k_i) + self.dV[0,1]
            self.Vx[i,:] = Qx + np.dot(np.dot(K_i.T,Quu),k_i) + np.dot(K_i.T,Qu) + np.dot(Qux.T,k_i)
            self.Vxx[i,:,:] = Qxx + np.dot(np.dot(K_i.T,Quu),K_i) + np.dot(K_i.T,Qux) + np.dot(Qux.T,K_i)
            self.Vxx[i,:,:] = 0.5 * ( self.Vxx[i,:,:].T + self.Vxx[i,:,:] )
                                                                                               
            # save the control gains
            self.l[i,:] = k_i
            self.L[i,:,:] = K_i
            
        return diverge_test, Quu_save, Quu_inv_save
                   
        
    def update(self,x0,u0):
        # current position
        self.x0 = x0
        
        # initial input
        self.u = u0
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # timer setting
        # trace for iteration
        # timer, counters, constraints
        # timer begin!!
        
        # generate initial trajectory
        diverge = False
        self.x[0,:] = self.x0
        for j in range(np.size(self.Alpha,axis=0)):   
            for i in range(self.N):
                self.x[i+1,:] = self.model.forwardDyn(self.x[i,:],self.Alpha[j]*self.u[i,:],i)       
                self.c[i] = self.cost.estimateCost(self.x[i,:],self.Alpha[j]*self.u[i,:],self.Mu_e[i,:,:],self.lam[i,:,:])
                
                if  np.max( self.x[i+1,:] ) > 1e8 :                
                    diverge = True
                    print "initial trajectory is already diverge"
                    pass

            self.c[self.N] = self.cost.estimateCost(self.x[self.N,:],np.zeros(self.model.iu), 
                                                    self.Mu_e[self.N,:,:],self.lam[self.N,:,:])
            if diverge == False:
                break
                pass
            pass
        # iterations starts!!
        iteration = 0
        # flgChange = True
        for iterationOut in range(self.maxIterOut) :
            print( "Outloop iteration == %d\n" % (iterationOut) )

            # initialzie parameters
            self.lamda = 1.0
            self.dlambda = 1.0
            self.lamdaFactor = 1.6
            
            # boolian parameter setting
            diverge = False
            stop = False
            lamda_max = False
            flgChange = True
            
            # intitialize some matrix
            self.initialize()
            for iteration in range(self.maxIterIn) :
                
                # set the cost function
                
                # differentiate dynamics and cost
                if flgChange == True:
                    start = time.clock()
                    self.fx, self.fu = self.model.diffDyn(self.x[0:N,:],self.u)
                    c_x_u = self.cost.diffCost(self.x[0:N,:],self.u,self.Mu_e[0:N,:,:],self.lam[0:N,:,:])
                    c_xx_uu = self.cost.hessCost(self.x[0:N,:],self.u,self.Mu_e[0:N,:,:],self.lam[0:N,:,:])
                    c_xx_uu = 0.5 * ( np.transpose(c_xx_uu,(0,2,1)) + c_xx_uu )
                    self.cx[0:N,:] = c_x_u[:,0:self.model.ix]
                    self.cu[0:N,:] = c_x_u[:,self.model.ix:self.model.ix+self.model.iu]
                    self.cxx[0:N,:,:] = c_xx_uu[:,0:ix,0:ix]
                    self.cxu[0:N,:,:] = c_xx_uu[:,0:ix,ix:(ix+iu)]
                    self.cuu[0:N,:,:] = c_xx_uu[:,ix:(ix+iu),ix:(ix+iu)]
                    c_x_u = self.cost.diffCost(self.x[N,:],np.zeros(iu),self.Mu_e[N,:,:],self.lam[N,:,:])
                    c_xx_uu = self.cost.hessCost(self.x[N,:],np.zeros(iu),self.Mu_e[N,:,:],self.lam[N,:,:])
                    c_xx_uu = 0.5 * ( c_xx_uu + c_xx_uu.T)
                    self.cx[N,:] = c_x_u[0:self.model.ix]
                    self.cxx[N,:,:] = c_xx_uu[0:ix,0:ix]
                    flgChange = False
                    pass

                time_derivs = (time.clock() - start)

                # backward pass
                backPassDone = False
                while backPassDone == False:
                    start =time.clock()
                    diverge,Quu_save,Quu_inv_save = self.backward()
                    time_backward = time.clock() - start
                    if diverge != 0 :
                        if self.verbosity == True:
                            print("Cholesky failed at %s" % (diverge))
                            pass
                        self.dlamda = np.maximum(self.dlamda * self.lamdaFactor, self.lamdaFactor)
                        self.lamda = np.maximum(self.lamda * self.dlamda,self.lamdaMin)
                        if self.lamda > self.lamdaMax :
                            break
                            pass
                        continue
                        pass
                    backPassDone = True
                # check for termination due to small gradient
                g_norm = np.mean( np.max( np.abs(self.l) / (np.abs(self.u) + 1), axis=1 ) )
                if g_norm < self.tolGrad and self.lamda < 1e-5 :
                    self.dlamda = np.minimum(self.dlamda / self.lamdaFactor, 1/self.lamdaFactor)
                    if self.lamda > self.lamdaMin :
                        temp_c = 1
                        pass
                    else :
                        temp_c = 0
                        pass       
                    self.lamda = self.lamda * self.dlamda * temp_c 
                    if self.verbosity == True:
                        print("SUCCEESS : gradient norm < tolGrad")
                        pass
                    break
                    pass
                # print np.sum(self.L),np.sum(self.l)
                # step3. line-search to find new control sequence, trajectory, cost
                fwdPassDone = False
                if backPassDone == True :
                    start = time.clock()
                    for i in self.Alpha :
                        self.xnew,self.unew,self.cnew= self.forward(self.x0,self.u,self.L,self.x,self.l,i,self.Mu_e,self.lam)
                        # print np.sum(self.unew-self.u)
                        dcost = np.sum( self.c ) - np.sum( self.cnew )
                        expected = -i * (self.dV[0,0] + i * self.dV[0,1])
                        if expected > 0. :
                            z = dcost / expected
                        else :
                            z = np.sign(dcost)
                            print("non-positive expected reduction: should not occur")
                            pass
                        # print(i)
                        if z > self.zMin :
                            fwdPassDone = True
                            break          
                    if fwdPassDone == False :
                        alpha_temp = 1e8 # % signals failure of forward pass
                        pass
                    time_forward = time.clock() - start
                    
                # step4. accept step, draw graphics, print status 
                if self.verbosity == True and self.last_head == True:
                    self.last_head = False
                    print("iteration   cost        reduction   expected    gradient    log10(lambda)")
                    pass

                if fwdPassDone == True:
                    if self.verbosity == True:
                        print("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f" % ( iteration,np.sum(self.c),dcost,expected,g_norm,np.log10(self.lamda)) )     
                        pass

                    # decrese lamda
                    self.dlamda = np.minimum(self.dlamda / self.lamdaFactor, 1/self.lamdaFactor)
                    if self.lamda > self.lamdaMin :
                        temp_c = 1
                        pass
                    else :
                        temp_c = 0
                        pass
                    self.lamda = self.lamda * self.dlamda * temp_c 

                    # accept changes
                    self.u = self.unew
                    self.x = self.xnew
                    self.c = self.cnew
                    # self.Mu = self.Munew
                    flgChange = True
                    # print(time_derivs)
                    # print(time_backward)
                    # print(time_forward)
                    # abc
                    # terminate?
                    if dcost < self.tolFun :
                        if self.verbosity == True:
                            print("SUCCEESS: cost change < tolFun",dcost)
                            pass
                        break
                        pass

                else : # no cost improvement
                    # increase lamda
                    # ssprint(iteration)
                    self.dlamda = np.maximum(self.dlamda * self.lamdaFactor, self.lamdaFactor)
                    self.lamda = np.maximum(self.lamda * self.dlamda,self.lamdaMin)

                    # print status
                    if self.verbosity == True :
                        print("%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f" %
                         ( iteration,'NO STEP', dcost, expected, g_norm, np.log10(self.lamda) ));
                        pass

                    if self.lamda > self.lamdaMax :
                        if self.verbosity == True:
                            print("EXIT : lamda > lamdaMax")
                            pass
                        break
                        pass
                    pass
                pass
            
            # outer loop updates 
            # terminal condition
            c_const = self.cost.ineqConst(self.x, np.vstack((self.u,np.zeros(self.model.iu)) )) # N * ic
            if np.max(c_const) < self.tolConst:
                print("EXIT : max(c) < tolConst")
                print np.max(c_const)
                break
                
            print "update lagrangian variables\n"
            for i in range(self.N+1) :
                    
                for j in range(self.cost.ic) :
                    # Mu uddate
                    ################
                    if c_const[i,j] > 0 or self.lam[i,j,j] > 0 :
                        self.Mu[i,j,j] = self.Mu_e[i,j,j]
                    else :
                        self.Mu[i,j,j] = 0  
                    ################
              
                    if c_const[i,j] < self.phi[i] :
                        # print "Hi",c_const[i,j],i
                        self.lam[i,j,j] = np.max(( 0, self.lam[i,j,j] + self.Mu_e[i,j,j] * c_const[i,j] ))
                        # print self.lam[i,j,j]
                        self.phi[i] = self.phi[i] * 1 / 1.5
                    else :
                        if self.Mu_e[i,j,j] < 1e30 :
                            self.Mu_e[i,j,j] = self.Mu_e[i,j,j] * 1.5
                            # print "Hi", self.Mu_e[i,j,j]
                        else :
                            print "Mu reaches the limit"
                            pass

                if i != N :
                    self.c[i] = self.cost.estimateCost(self.x[i,:],self.u[i,:],self.Mu_e[i,:,:],self.lam[i,:,:])
                else :
                    self.c[i] = self.cost.estimateCost(self.x[i,:],np.zeros(self.model.iu),self.Mu_e[i,:,:],self.lam[i,:,:])

                

                    
            fS = 18
            plt.plot(self.x[:,0], self.x[:,1], linewidth=2.0)
            plt.plot(np.linspace(0,4,10),np.linspace(0,4,10))
            plt.plot(self.cost.x_t[0],self.cost.x_t[1],"o")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis([0, 4.0, 0, 4.0])
            plt.xlabel('X (m)', fontsize = fS)
            plt.ylabel('Y (m)', fontsize = fS)
            plt.show()
            plt.subplot(121)
            plt.plot(np.array(range(N))*0.1, self.u[:,0], linewidth=2.0)
            plt.xlabel('time (s)', fontsize = fS)
            plt.ylabel('v (m/s)', fontsize = fS)
            plt.subplot(122)
            plt.plot(np.array(range(N))*0.1, self.u[:,1], linewidth=2.0)
            plt.xlabel('time (s)', fontsize = fS)
            plt.ylabel('w (rad/s)', fontsize = fS)
            plt.show()

        # for i in range(self.N+1) :
                    
        #         for j in range(self.cost.ic) :
        #             # Mu uddate
        #             ################
        #             if c_const[i,j] > 0 or self.lam[i,j,j] > 0 :
        #                 self.Mu[i,j,j] = self.Mu_e[i,j,j]
        #             else :
        #                 self.Mu[i,j,j] = 0  
        #             ################
        # flag_const = np.logical_or( c_const > 0, np.diagonal(self.lam,0,2,1) > 0 )
        # temp = np.expand_dims(flag_const,2) * self.Mu_e
        # print "Hello", np.sum(self.Mu - temp)        

        return self.x, self.u, Quu_save, Quu_inv_save, self.L, self.l, self.lam, self.Mu
        


        
        
        
        
        
        
        
        
        
        
        
        


