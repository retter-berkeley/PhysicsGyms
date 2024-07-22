#taken from https://www.gymlibrary.dev/content/environment_creation/
#create gym for van der pol oscillator
#action space: continuous  +/- 1.0 float , maybe make scale to mu 
#observation space:  -10,10 x2 float for x and y
#reward:  -1*(x^2+y^2)^1/2 (try to drive to 0)

#van der pol equations:
#xdot = y; ydot = mu(1-x^2)y-x
#for this gymnasium, assume mu constant hard coded, and ydot=ydot+b, where b is action

from os import path
from typing import Optional

import numpy as np
import math

import scipy
from scipy.integrate import solve_ivp

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
import dynamics
from dynamics import VanDerPol


class VDPEnv(gym.Env):
    #no render modes
    def __init__(self, render_mode=None, size=20):
        metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 30,
        }
        
        self.size=size
        self.window_size=1024
        
        self.observation_space =spaces.Box(low=-size+1, high=size-1, shape=(2,), dtype=float)
     
        self.action_space = spaces.Box(-10, 10, shape=(1,), dtype=float) 
        #need to update action to normal distribution
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode=render_mode
        
        self.window= None
        self.clock = None
        
    def _get_obs(self):
        return self.state

    # def _get_info(self):
        # #return np.linalg.norm(self._agent_location - self._target_location, ord=1)
        # #obsolete/unused for gymslotin6.3
        
        # return np.linalg.norm(self._agent_location - self._target_location, ord=1)

    def reset(self, seed: Optional[int] = None, options=None):
        #need below to seed self.np_random
        super().reset(seed=seed)
        #can i implement seed here?  does my current implementation work or lead to x=y?
        #start random x1, x2 origin
        np.random.seed(seed)
        x=np.random.uniform(-8.,8.)
        while (x>-0.5 and x<0.5):
            np.random.seed()
            x=np.random.uniform(-8.,8.)
        np.random.seed(seed)
        y=np.random.uniform(-8.,8.)
        while (y>-0.5 and y<0.5):
            np.random.seed()
            y=np.random.uniform(-8.,8.)
        self.state = np.array([x,y])#  self.np_random.uniform(0, 1, size=2, dtype=float)
        observation = self._get_obs()

        return observation, {}
    
    def step(self,action):
    
        # def dynamics(t, a, K):
            # #a is 2x1 array a1, a2
            # #have to put inside step for now, which is very inefficient but not sure how
            # #else to get it into the gym
            # #not part of typical gym this holds the dynamics for solv_ivp to march through
            # a1,a2=a
                
            # #catch unstable forcing or system to prevent float overflow; set to max of state space defined in init  
            # if a1>20:  a1=20.
            # elif a1<-20:  a1=-20.
            # if a2>20:  a2=20.
            # elif a2<-20:  a2=-20.
        
            # #put dynamics here, in ODE format
            # #for VDP, dx/dt=y; dy/dt=mu*(1-x**2)y-x (+k for control input)
            # mu=4
            # da1=a2+K
            # da2=mu*(1-a1**2)*a2-a1
            # return np.array([da1, da2])
    
        hit=0
        bang=0
        mu=4.
        u=action.item()
        #x,y=self.state
        #x1=0.005*(y+action)+x#dt=0.001 #old way
        #x2=0.005*((1-x**2)*y*mu+x)+y
        #new way, using solve_ivp for larger timesteps to improve convergence
        result=solve_ivp(VanDerPol, (0, 0.2), self.state, args=[u])
        #ivp returns a class with state in result.y
        #result.y is a nxm matrix, with n the dimenision of the state space
        #m the number of time steps solve_ivp use to cover the time domain
        x1=result.y[0,-1]
        x2=result.y[1,-1]
        self.state=np.array([x1.item(),x2.item()])
        done=False
        observation=self._get_obs()
        info=x1
        done=(math.sqrt(x1.item()**2+x2.item()**2)<0.5)# and action<0.5) 
        reward = -math.sqrt(x1.item()**2)#+x2.item()**2)
        #reward = -math.sqrt(x1.item()**2)
        #if done:
        #    reward = 100.0
        #    hit=1
        #else:
        #    reward = -math.sqrt(x1.item()**2+x2.item()**2)#-action
        truncated = False #placeholder for future expnasion/limits if solution diverges
        info = x1

        return observation, reward, done, truncated, {}
    
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
