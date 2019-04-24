import numpy as np
from physiology.muscle_params_BIC_TRI import *

# params dictionary

params = {
    "Horizon" : 300,
    "SimulationDuration" : 10,
    "dt" : 0.01,
    "U_o" : None,
    "p_target" : np.matrix([[np.pi/4,0]]).T,
    "LearningRate" : 0.2,
    "Q_f" : np.matrix(
            [
                [500,0],
                [0,50]
            ]
        ),
    "R" : np.matrix(
            [
                [2e-3,-1e-3],
                [-1e-3,2e-3]
            ]
        ),
    "InputBounds" : [
            [0,F_MAX1],
            [0,F_MAX2]
        ],
    "PlotResults" : True,
    "AnimateResults" : True,
    "ReturnAllResults" : True,
    "NumberOfIterations" : 100
}

# h is the step used to determine the derivative

h = 0.000001

# Cart Pole Parameters

M = 5 # kg

#gravity
gr = 9.81 # m/s²

# length parameter
L1 = 0.5 # m

# damping parameters
b1 = 1 # Nm
