import numpy as np
from Function3 import *
from Function4 import *

def fnState_And_Control_Transition_Matrices(x,u,du,dt):
    """
    Takes in the state vector (x), the input vector (u), and the input derivative (du) (as well as some unused vars - du and dt) and outputs the linearized state and control matrices (A and B, respectively).

    NOTE: h is double defined here (see TO DO list) and is used to determine the linearized matrices, *without* calculating the explicit formula for the state equation derivatives. Therefore, there is no need for the excessive calculation of these formulae explicitly.

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - See if h is needed in any other function and get rid of it if it is doubly defined.
    [ ] - Create tests that ensure that x and u are the correct dimensions.
    [ ] - Create tests to make sure that the outputs are of the correct sizes.
    """

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]

    # Removed the u split into two scalars because u is already a scalar.
    h = .00000001
    A = np.zeros((4,4))

    # Build the A matrix

    # A[0,0] = 0 # dF1/x1⋅dx1 = x3
    # A[0,1] = 0 # dF1/x2⋅dx1 = x3
    A[0,2] = 1 # dF1/x3⋅dx1 = x3
    # A[0,3] = 0 # dF1/x4⋅dx1 = x3

    # A[1,0] = 0 # dF2/x1⋅dx2 = x4
    # A[1,1] = 0 # dF2/x2⋅dx2 = x4
    # A[1,2] = 0 # dF2/x3⋅dx2 = x4
    A[1,3] = 1 # dF2/x4⋅dx2 = x4

    # F3 is the acceleration of the cart.
    A[2,0] = (Function3(x1,x2,x3,x4,u)-Function3(x1-h,x2,x3,x4,u))/h
    A[2,1] = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2-h,x3,x4,u))/h
    A[2,2] = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2,x3-h,x4,u))/h
    A[2,3] = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2,x3,x4-h,u))/h

    # F4 is the angular acceleration of the pendulum.
    A[3,0] = (Function4(x1,x2,x3,x4,u)-Function4(x1-h,x2,x3,x4,u))/h
    A[3,1] = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2-h,x3,x4,u))/h
    A[3,2] = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2,x3-h,x4,u))/h
    A[3,3] = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2,x3,x4-h,u))/h

    #Build the B matrix
    B = np.zeros((4,1))
    B[0,0] = 0
    B[1,0] = 0
    B[2,0] = (Function3(x1,x2,x3,x4,u)-Function3(x1,x2,x3,x4,u-h))/h
    B[3,0] = (Function4(x1,x2,x3,x4,u)-Function4(x1,x2,x3,x4,u-h))/h

    return(A,B)
