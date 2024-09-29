Contains files needed for ODE-basede games.
VDP contains gymansium based providing Vanderpol Osciallator
LRZ is Lorenz Attractor
Mean Field is Coupled wave system as described in Duriez/Brunton/Noack "Machine Learning Control-Taming Nonlinear Dymamics and Turbulence"
MGP is the Moore-Greitzer Model (MGM) describes gas turbine engine surge-stall dynamics and described:  https://dspace.mit.edu/handle/1721.1/109342

Eache gymanisum file creates observation space consisting of the state space for the correct number dimensionality for each system.  
The action space is one-dimensional.  The step uses solve IVP with a small time step over which the control is applied as a constant. Solve IVP calls dynamics package, which holds exact equations, with fixed parameters.

I implement gymnasium files by placing in gymanisum classic control, updating necessary __init__.py files in that gymnaiusm (not incluced) and manually/locally importing library in python script.  I also manually/locally import the dynamics library in the python scrip
