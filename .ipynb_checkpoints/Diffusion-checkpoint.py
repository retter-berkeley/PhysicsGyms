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
import logging
logger = logging.getLogger('requests_throttler')
logger.addHandler(logging.NullHandler())
logger.propagate = False


#diffusion equation is included in py-pde, so don't need seperate dynamics
#import dynamics
#from dynamics import MGM

class DiffusionEnv(gym.Env):
    #no render modes
    def __init__(self, render_mode=None, size: int =20):
                
        self.observation_space =spaces.Box(low=-10, high=10, shape=(size,), dtype=float)
     
        self.action_space = spaces.Box(-10, 10, shape=(20,), dtype=float) 
        #need to update action to normal distribution

        self.grid=[]
        self.stepper=[]
               
    def _get_obs(self):
        
        return self.state.data

    def reset(self, seed: Optional[int] = None, options=None):
        #need below to seed self.np_random
        super().reset(seed=seed)
        #print("check1")
        bc_x="periodic"
        bc_y="periodic"
        grid = pde.CartesianGrid([[0, 1], [0, 1]], [20, 20], periodic=[False, True]) # generate grid
        state = ScalarField.random_uniform(grid, 0.0, 0.2)
        bc_x=[{"value": 0.}, {"value": 0.}]
        eq = DiffusionPDE(diffusivity=.1, bc=[bc_x, bc_y])
        #print("check2")
        solver=pde.ExplicitSolver(eq,scheme="euler", adaptive=True)
        stepper=solver.make_stepper(state, dt=1e-3)
        #print("check3")
        self.state = state
        self.stepper = stepper
        self.grid = grid
        observation = self.state#self._get_obs()

        return observation, grid, state, stepper
        
    def step(self,action):
        #need to keep state and observation distinct; how to use observation?
        #action is an object that holds the grid, the state, and the boundary conditions/controls
        #control row order is left, right, bottom, top
        #u=action.item()
        #grid = pde.CartesianGrid([[0, 4], [0, 4]], [20, 20], periodic=[True, True])
        # state = ScalarField.random_uniform(grid, 0.0, 0.2)
        # state=self.state
        stepper=self.stepper
        ###uncomment if using spcific control points along boundary or all boundaries as control###
        #bc_x_left = {"value": action.control[0,:]}
        #bc_x_right = {"value": action.control[1,:]}
        #bc_x = [bc_x_left, bc_x_right]
        #bc_y_bottom = {"value": action.control[2,:]}
        #bc_y_top = {"value": action.control[3,:]}
        #bc_y = [bc_y_bottom, bc_y_top]
        
        ###uncomment if using bottom boundary as control###
        #bc_y_top={"value": 0}
        #bc_y_bottom={"value": action.control}
        #bc_y=[bc_y_bottom, bc_y_top]
        #bc_x="periodic"
          
        self.state.data[1,:]=action
        #grid=self.grid
        state=self.state
        
        t_current=stepper(self.state, 0., 0.+0.01)
        #result = eq.solve(state, t_range=0.2, adaptive = True, tracker=None)       
        done=False
        truncated=False
        observation=self._get_obs()
        state=self.state
        #reward will be based on difference across all grid cells of sensor area between
        #desired sensor readings and actual.  This is  hard coded in the gym step function
        #in future could make it an input to init

        #square target
        #target=np.array([[1,2,1,],[2,3,2,],[1,2,1,]])
        #use eucliead norm over n gird points
        #first reduce state to sensor points
        # n_sense=3 #hard coded
        # meas=np.empty((n_sense,n_sense))
        # dimx=state.data.shape[0]
        # dimy=state.data.shape[1]
        # startx=round((dimx-n_sense)/2)
        # starty=round((dimy-n_sense)/2)
        # for i in range(n_sense):
        #     for j in range(n_sense):
        #         meas[i,j]=float(state.data[i+startx,j+starty])
        # reward=(meas-target)**2
        # reward=-1*math.sqrt(np.sum(np.sum(reward)))

        #linear target
        target=np.array([0,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3,2,1,2,3])
        target=target/3.
        reward=(self.state.data[19,:]-target)**2
        reward=-1*math.sqrt(np.sum(reward))
        
        if reward>-1.0:
            reward=50
            done=True
        np.clip(self.state.data,-100,100)
        # if np.any(self.state.data>500.) or np.any(self.state.data<-500):
        #     reward=-100
        #     truncated = True #placeholder for future expnasion/limits if solution diverges

        return self.state, reward, done, False, {}
    
    
    #def close(self):
        
