#taken from https://www.gymlibrary.dev/content/environment_creation/
#create gym for Mean Field System
#action space: continuous  +/- 10.0 float , maybe make scale to mu 
#observation space:  -30,30 x2 float for x,y,zand
#reward:  -1*L2 norm of 4d vector (try to drive to 0)

#For details on equations see dynamcis.py

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
from dynamics import MeanField


class MeanFieldEnv(gym.Env):
    #no render modes
    def __init__(self, render_mode=None, size=30):
        
        self.size=size
        self.window_size=1024
        
        self.observation_space =spaces.Box(low=-size+1, high=size-1, shape=(4,), dtype=float)
     
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=float) 
        #need to update action to normal distribution
               
    def _get_obs(self):
        return self.state

    def reset(self, seed: Optional[int] = None, options=None):
        #need below to seed self.np_random
        super().reset(seed=seed)
        #can i implement seed here?  does my current implementation work or lead to x=y?
        #start random x1, x2 origin
        np.random.seed(seed)
        w=np.random.uniform(-8.,8.)
        while (w>-0.5 and w<0.5):
            np.random.seed()
            w=np.random.uniform(-8.,8.)
        x=np.random.uniform(-8.,8.)
        while (x>-0.5 and x<0.5):
            np.random.seed()
            x=np.random.uniform(-8.,8.)
        np.random.seed(seed)
        y=np.random.uniform(-8.,8.)
        while (y>-0.5 and y<0.5):
            np.random.seed()
            y=np.random.uniform(-8.,8.)
        np.random.seed(seed)
        z=np.random.uniform(-8.,8.)
        while (z>-0.5 and z<0.5):
            np.random.seed()
            z=np.random.uniform(-8.,8.)
        self.state = np.array([0.01,0.,0.,0.])#  use initial point from Druiez
        observation = self._get_obs()

        return observation, {}
    
    def step(self,action):
    
        u=action.item()

        result=solve_ivp(MeanField, (0, 1.), self.state, args=[u])

        x1=result.y[0,-1]
        x2=result.y[1,-1]
        x3=result.y[2,-1]
        x4=result.y[3,-1]
        self.state=np.array([x1.item(),x2.item(), x3.item(), x4.item()])
        done=False
        observation=self._get_obs()
        reward = -math.sqrt(x1.item()**2+x2.item()**2+x3.item()**2+x4.item()**2)
        truncated = False #placeholder for future expnasion/limits if solution diverges

        return observation, reward, done, truncated, {}
    
    
    #def close(self):
        
