import numpy as np

# h is the step used to determine the derivative

h = 0.000001

# Cart Pole Parameters

m2 = 1 # kg
m1 = 10 # kg

#gravity
gr = 9.81 # m/s²

# length parameter
L = 1.5 # m

# damping parameters
b1 = 0 # Nm
b2 = 0 # Nm

# Weight in Final State:
Q_f = np.matrix(np.zeros((4,4)))
Q_f[0,0] = 5
Q_f[1,1] = 1000
Q_f[2,2] = 5
Q_f[3,3] = 50

# Weight in the Control:
# Modified from original because our control is only one dimensional.
R = 1e-3
