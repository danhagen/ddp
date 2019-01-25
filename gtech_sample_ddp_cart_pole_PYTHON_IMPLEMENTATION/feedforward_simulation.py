import numpy as np
from dynamics import *

def return_state_trajectories_given_U_new(X_o,U_new,dt,**kwargs):
    """
    Simulates the system by plugging in the initial conditions (X_o), the new input (U_new), the Horizon (as a means of creating the length of the iteration), and the timestep (dt). Returns a (4,Horizon) numpy.ndarray of the states variables over time.

    #######################
    ###### **kwargs #######
    #######################

    1) Horizon -
        Determines how far into the future to calculate the corresponding states for a given U_new. If you care about the state at the Horizon, then you will need to consider the terminal cost AT the Horizon. Default is set to the length of the input U_new.

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] -

    """
    Horizon = kwargs.get("Horizon",np.shape(U_new)[0])
    X = np.zeros((4,Horizon))
    X[:,0] = X_o
    Fx = np.zeros((4,))

    for k in range(Horizon-1):
        Fx[0] = F1(X[:,k],U_new[k])
        Fx[1] = F2(X[:,k],U_new[k])
        Fx[2] = F3(X[:,k],U_new[k])
        Fx[3] = F4(X[:,k],U_new[k])

        X[:,k+1] = X[:,k] + Fx * dt
    return(X)
