import numpy as np

# params dictionary

params = {
    "Horizon" : 300,
    "SimulationDuration" : 10,
    "dt" : 0.01,
    "U_o" : None,
    "p_target" : np.matrix([[0,0,0,0,0,0]]).T,
    "LearningRate" : 0.2,
    "Q_f" : np.matrix(
            [
                [5,0,0,0,0,0],
                [0,1000,0,0,0,0],
                [0,0,1000,0,0,0],
                [0,0,0,5,0,0],
                [0,0,0,0,50,0],
                [0,0,0,0,0,50]
            ]
        ),
    "R" : 1e-3,
    "PlotResults" : True,
    "AnimateResults" : True,
    "ReturnAllResults" : True,
    "NumberOfIterations" : 100
}

# h is the step used to determine the derivative

h = 0.000001

# Cart Pole Parameters

m1 = 1 # kg
m2 = 1 # kg
M = 10 # kg

#gravity
gr = 9.81 # m/s²

# length parameter
L1 = 1.5 # m
L2 = 1 # m

# damping parameters
b1 = 0 # Nm
b2 = 0 # Nm
b3 = 0 # Nm
