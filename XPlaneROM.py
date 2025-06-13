#Gymansium for Mooney Dynmaics ROM
#taken from https://www.gymlibrary.dev/content/environment_creation/
#create gymansium to pre-train an RL agent that will try and fly a Mooney M20J in XPlane
#pre train by using SINDY to make ROM
#will need same states and inputs as XPlane Gym
#state is:  throttle position, pitch ratio, roll ratio, altitude, heading, airspeed
#altitude, heading airspeed normalized
#SpeedNorm= (Speed-120.)/(170-70) #if speed outside these limits, then fail
#AltNorm= (Alt-4000)/(5500-2500) #if alt outside limits, fail
#HdgNorm = (Hdg-180)/(270-90) # hdg limis 90-270

from os import path
from typing import Optional

import numpy as np
import math
import pde
import random
#from XPPython3 import xp

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
##At the top of the code
import logging
logger = logging.getLogger('requests_throttler')
logger.addHandler(logging.NullHandler())
logger.propagate = False

class XPlaneROMEnv(gym.Env):
    #observation space is state and control spaces, normalized
    #space is 7-vector

    def __init__(self, render_mode=None, size: int =7):
        #gymansium init:  make spaces
        obs_size=7
        self.observation_space =spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=float)
        action_size=3
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_size,), dtype=float) 
        #need to update action to normal distribution

        self.grid=[]
        self.stepper=[]
        
    def _get_obs(self):
        #tbd
        
        #convert to degrees and normalize to +/- 1 with guessed limits 
        #for the ROM gym, all will be in normalized
        # SpeedNorm= (Speed-120.)/(170-70) #if speed outside these limits, then fail
        # AltNorm= (Alt-4000)/(5500-2500) #if alt outside limits, fail
        # HdgNorm = (Hdg-180)/(270-90) # hdg limis 90-270
        #all other already normalized
        
        #self.state.data=[Pitch, Roll, Throttle, AltNorm, HdgNorm, SpeedNorm]
        return self.state

    def reset(self, seed: Optional[int] = None, options=None):
        #reset to altitude 4000, airspeed 120, heading 180 or, as normalized, 0,0,0
        #pitch, roll to 0.  throttle to ??
        self.state= [0., 0., 0.5, 0., 0., 0.]

        observation = self._get_obs()

        #commands randmoly chosen from a matrix of possible commands 
        #+/- 45 deg heading, with fixed initial heading this is 225 or 135
        #+/- 500 ft elevation, with fixed initial altitude this is 4500 or 3500
        #combination of the two
        #commands is 2x8 array with heading change, elevation change as each row
        #commands = np.array([[225., 4000.], [135., 4000.], [180., 4500.], [180., 3500.], [225., 4500.], [135., 4500.], [225., 3500.], [135., 3500.]])
        #issue normalized commands
        commands = np.array([[45./180., 0.], [-45./180., 0.], [0., 500./3000], [0., -500./3000], [45./180, 500./3000], 
                             [-45./180, 500./3000], [45./180, -500./3000], [-45./180, -500./3000]])
        #choose a random command to send
        n=random.randint(0,7)
        self.command = commands[n,]

        return observation, self.state
        
    def step(self,action):
        #Use discrete equation determined by SINDY to advance state one time step
        
        #self.state.data[0,:]=action[0]
        state=self.state
        
        #[equation here]     
        self.state[3]=self.state[3]+action[0]
        self.state[5]=self.state[4]+action[1]
        self.state[5]=self.state[4]+action[2]
        #wait X seconds here, then get observation and reward
        
        observation=self._get_obs()
        reward=np.sqrt((self.state.data[3]-self.command[1])**2+(self.state.data[4]-self.command[0])**2+(self.state.data[5]-120.)**2)
        done=False
        truncated=False
        if (self.state.data[3]>1 or self.state.data[3]<-1 or self.state.data[4]>1 or self.state.data[4]<-1 or 
            self.state.data[5]>1 or self.state.data[5]<-1 ):
            truncated= True
            reward=-5000
        state=self.state
 
        return self.state, reward, done, truncated, {}
    
    
#    def close(self):
        