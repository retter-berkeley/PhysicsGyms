#dynamics.py
import numpy as np

def VanDerPol(t, a, K):
    #a is 2x1 array a1, a2
    #have to put inside step for now, which is very inefficient but not sure how
    #else to get it into the gym
    #not part of typical gym this holds the dynamics for solv_ivp to march through
    a1,a2=a
        
    #catch unstable forcing or system to prevent float overflow; set to max of state space defined in init  
    if a1>20:  a1=20.
    elif a1<-20:  a1=-20.
    if a2>20:  a2=20.
    elif a2<-20:  a2=-20.

    #put dynamics here, in ODE format
    #for VDP, dx/dt=y; dy/dt=mu*(1-x**2)y-x (+k for control input)
    mu=4
    da1=a2
    da2=mu*(1-a1**2)*a2-a1+K
    return np.array([da1, da2])
    
def Lorenz(t, X, K):
    #the Lorenz system is a classical example of chaotic system, where any initial conditions,
    #no matter how close together, will show solution divergence and unpredictable but deterministic behavior
    u, v, w = X
    sigma = 10.
    rho = 28.
    beta = 2.8
    up = -sigma*(u - v)
    vp = rho*u - v - u*w + K
    wp = -beta*w + u*v
    return np.array([up, vp, wp])
    
def MeanField(t, A, K):
    #non-linear generalized Mean Field system as described in
    #Chapter 5 of Duriez/Brunton/Noack Machine Learning Control-Taming Nonlinear Dynamics and Turbulence
    #this is a 4D system of osciallators with parameters of growth rate and growth rate change
    #This instantiation uses constant paramteres for most, with unstable nonlinear growth of first two a1, a2
    #As described in Duriez, this system is useful for describing several fluid phenomenon; derviation is
    #based on Navier Stokes equations
    a1, a2, a3, a4 = A
    if a1>20:  a1=20.
    elif a1<-20:  a1=-20.
    if a2>20:  a2=20.
    elif a2<-20:  a2=-20.
    if a3>20:  a3=20.
    elif a3<-20:  a3=-20.
    if a4>20:  a4=20.
    elif a4<-20:  a4=-20.
    sigma=0.1-a1**2-a2**2-a3**2-a4**2
    da1=sigma*a1-a2
    da2=sigma*a2-a1
    da3=-0.1*a3-10.*a4
    da4=-0.1*a4+10*a3+K
    return np.array([da1, da2, da3, da4])
    
def MGM(t, A, K):
    #non-linear approximation of surge/stall dynamics of a gas turbine engine per Moore-Greitzer model from
    #"Output-Feedbak Cotnrol on Nonlinear systems using Control Contraction Metrics and Convex Optimization"
    #by Machester and Slotine
    #2D system, x1 is mass flow, x2 is pressure increase
    x1, x2 = A
    if x1>20:  x1=20.
    elif x1<-20:  x1=-20.
    if x2>20:  x2=20.
    elif x2<-20:  x2=-20.
    dx1= -x2-1.5*x1**2-0.5*x1**3
    dx2=x1+K
    return np.array([dx1, dx2])