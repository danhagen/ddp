import numpy as np
from params import *

def return_quadratic_cost_function_expansion_variables(X,U,R,dt):
    """
    Takes in the input U and the the corresponding output X, as well as dt and returns lists that contain the coefficient matrices for the quadratic expansion of the cost function (l(x,u)) for each timestep for range(len(Time)-1).
    """
    # returns a list of length len(Time)-1, each element with shape (1,1), where n is the number of states.

    l = list(
            map(
                lambda X,U: U.T * R * U * dt,
                X[:,1:].T,
                U.T
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,1), where n is the number of states.
    lx = list(
            map(
                lambda X,U: np.matrix(np.zeros((6,1)))*dt,
                X[:,1:].T,
                U.T
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,1), where n is the number of states.
    lu = list(
            map(
                lambda X,U: R * U * dt,
                X[:,1:].T,
                U.T
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,n), where m is the number of inputs and n is the number of states.
    lux = list(
            map(
                lambda X,U: np.matrix(np.zeros((1,6)))*dt,
                X[:,1:].T,
                U.T
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.
    lxu = list(
            map(
                lambda X,U: np.matrix(np.zeros((6,1)))*dt,
                X[:,1:].T,
                U.T
            )
        )

    # returns a list of length len(Time)-1, each element with shape (m,m), where m is the number of inputs.
    luu = list(
            map(
                lambda X,U: R*dt,
                X[:,1:].T,
                U.T
            )
        )

    # returns a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.
    lxx = list(
            map(
                lambda X,U: np.matrix(np.zeros((6,6)))*dt,
                X[:,1:].T,
                U.T
            )
        )

    return(l,lx,lu,lux,lxu,luu,lxx)
