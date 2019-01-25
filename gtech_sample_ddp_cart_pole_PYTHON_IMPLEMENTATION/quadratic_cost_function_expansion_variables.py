import numpy as np

def return_quadratic_cost_function_expansion_variables(X,U,R,**kwargs):
    """
    Takes in the state vector (X) and the input scalar (U) and the running cost scalar R (as well as some unused vars - k and dt) and outputs the derivatives of the running cost needed for the quadratic approximation of the cost.

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create tests that ensure that X, U, and R are the correct dimensions.
    [ ] - Create tests to make sure that the output are of the correct sizes.

    #######################
    ####### **kwargs ######
    #######################

    1) i -
        Discrete time step with default set as None.

    2) dt -
        Time step with default set as None.

    """

    l = U.T * R * U
    lx = np.matrix(np.zeros((4,1)))
    lxx = np.matrix(np.zeros((4,4)))
    lu = R * U
    luu = R
    lux = np.matrix(np.zeros((1,4)))

    return(l,lx,lxx,lu,luu,lux)
