#taken from https://www.gymlibrary.dev/content/environment_creation/
#create gym for Moore-Greitzer Model
#action space: continuous  +/- 10.0 float , maybe make scale to mu 
#observation space:  -30,30 x2 float for x,y,zand
#reward:  -1*(x^2+y^2+z^2)^1/2 (try to drive to 0)

#Moore-Grietzer model:


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
from dynamics import MGM


class MGMEnv(gym.Env):
    #no render modes
    def __init__(self, render_mode=None, size=30):
                
        self.observation_space =spaces.Box(low=-size+1, high=size-1, shape=(2,), dtype=float)
     
        self.action_space = spaces.Box(-10, 10, shape=(1,), dtype=float) 
        #need to update action to normal distribution
               
    def _get_obs(self):
        return self.state

    def reset(self, seed: Optional[int] = None, options=None):
        #need below to seed self.np_random
        super().reset(seed=seed)
        #can i implement seed here?  does my current implementation work or lead to x=y?
        #start random x1, x2 origin
        np.random.seed(seed)
        x=np.random.uniform(-8.,8.)
        while (x>-2.5 and x<2.5):
            np.random.seed()
            x=np.random.uniform(-8.,8.)
        np.random.seed(seed)
        y=np.random.uniform(-8.,8.)
        while (y>-2.5 and y<2.5):
            np.random.seed()
            y=np.random.uniform(-8.,8.)
        self.state = np.array([x,y])#  self.np_random.uniform(0, 1, size=2, dtype=float)
        observation = self._get_obs()

        return observation, {}
    
    def step(self,action):
    
        u=action.item()

        result=solve_ivp(MGM, (0, 0.05), self.state, args=[u])

        x1=result.y[0,-1]
        x2=result.y[1,-1]
        self.state=np.array([x1.item(),x2.item()])
        done=False
        observation=self._get_obs()
        info=x1
        #done=(math.sqrt(x1.item()**2+x2.item()**2<0.5)# and action<0.5) 
        reward = -math.sqrt(x1.item()**2)#+x2.item()**2)
        #reward = -math.sqrt(x1.item()**2)
        #if done:
        #    reward = 10.0
        #    hit=1
        #else:
        #    reward = -math.sqrt(x1.item()**2+x2.item()**2)#-action
        truncated = False #placeholder for future expnasion/limits if solution diverges
        info = x1

        return observation, reward, done, truncated, {}
    
    
    #def close(self):
        
