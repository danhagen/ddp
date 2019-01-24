import numpy as np

def fnCost(x,u,k,R,dt):
    """
    Takes in the state vector (x) and the input scalar (u) and the running cost scalar R (as well as some unused vars - k and dt) and outputs the derivatives of the running cost.

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create tests that ensure that x, u, and R are the correct dimensions.
    [ ] - Create tests to make sure that the output are of the correct sizes.
    """
    l0 = u.T * R * u
    l_x = np.zeros((4,1))
    l_xx = np.zeros((4,4))
    l_u = R * u
    l_uu = R
    l_ux = np.zeros((1,4))

    return(l0,l_x,l_xx,l_u,l_uu,l_ux)
