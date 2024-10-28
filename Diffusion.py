#taken from https://www.gymlibrary.dev/content/environment_creation/
#create gym for 2-D Diffusion equation using py-pde
#action space: continuous  +/- 10.0 float.  Action will be on boundary, but need to minimize total effort
#observation space:  -32,32 x2 float for x,y.  Observe grid in center and try to drive to a value
#reward:

#Diffusion model:


from os import path
from typing import Optional

import numpy as np
import math
import pde
from pde import DiffusionPDE, ScalarField, UnitGrid

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
##At the top of the code


#diffusion equation is included in py-pde, so don't need seperate dynamics
#import dynamics
#from dynamics import MGM

class DiffusionEnv(gym.Env):
    #no render modes
    def __init__(self, render_mode=None, size=32):
                
        self.observation_space =spaces.Box(low=0, high=size, shape=(2,), dtype=float)
     
        self.action_space = spaces.Box(-10, 10, shape=(1,), dtype=float) 
        #need to update action to normal distribution
               
    def _get_obs(self):
        
        return self.state

    def reset(self, seed: Optional[int] = None, options=None):
        #need below to seed self.np_random
        super().reset(seed=seed)
        grid = pde.CartesianGrid([[0, 1], [0, 1]], [32, 32], periodic=[False, False]) # generate grid
        state = ScalarField.random_uniform(grid, 0.0, 0.5)
        self.state = state.data
        observation = self._get_obs()

        return observation, grid, state
        
    def step(self,action):
        #need to keep state and observation distinct; how to use observation?
        #action is an object that holds the grid, the state, and the boundary conditions/controls
        #control row order is left, right, bottom, top
        #u=action.item()
        grid=action.grid
        
        ###uncomment if using spcific control points along boundary or all boundaries as control###
        bc_x_left = {"value": action.control[0,:]}
        bc_x_right = {"value": action.control[1,:]}
        bc_x = [bc_x_left, bc_x_right]
        bc_y_bottom = {"value": action.control[2,:]}
        bc_y_top = {"value": action.control[3,:]}
        bc_y = [bc_y_bottom, bc_y_top]
        
        ###uncomment if using bottom boundary as control###
        #bc_y_top={"value": 0}
        #bc_y_bottom={"value": action.control}
        #bc_y=[bc_y_bottom, bc_y_top]
        #bc_x="periodic"
          
        
        eq = DiffusionPDE(diffusivity=10, bc=[bc_x, bc_y])
        result = eq.solve(action.state, t_range=0.2, adaptive = True, tracker=None)
        self.state = result.data       
        done=False
        observation=self._get_obs()
        #reward will be based on difference across all grid cells of sensor area between
        #desired sensor readings and actual.  This is  hard coded in the gym step function
        #in future could make it an input to init
        target=np.array([[1,2,1,1],[2,3,2,2],[1,2,1,1],[0.5,0.5,0.5,0.5]])
        #use eucliead norm over n gird points
        #first reduce state to sensor points
        n_sense=int(np.sqrt(action.num_sens))
        meas=np.empty((n_sense,n_sense))
        dimx=result.data.shape[0]
        dimy=result.data.shape[1]
        startx=round((dimx-n_sense)/2)
        starty=round((dimy-n_sense)/2)
        for i in range(n_sense):
            for j in range(n_sense):
                meas[i,j]=float(result.data[i+startx,j+starty])
        reward=(meas-target)**2
        reward=1*math.sqrt(np.sum(np.sum(reward)))
        
        truncated = False #placeholder for future expnasion/limits if solution diverges

        return observation, reward, done, truncated, {}
    
    
    #def close(self):
        
