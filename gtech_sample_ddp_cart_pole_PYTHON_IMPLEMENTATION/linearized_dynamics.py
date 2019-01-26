import numpy as np
from params import *
from dynamics import *

def return_Phi(X,U,dt):
    """
    Takes in the state vector (X), the input vector (U) and returns the discretized and linearized state matrix, Phi.

    NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create tests that ensure that X and U are the correct dimensions.
    [ ] - Create tests to make sure that the outputs are of the correct sizes.
    """
    assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
            and np.shape(X)==(4,)), "Error with the type and shape of X ["+ fnState_And_Control_Transition_Matrices.__name__+"()]."
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."

    # Removed the U split into two scalars because U is already a scalar.

    h1 = np.array([h,0,0,0])
    h2 = np.array([0,h,0,0])
    h3 = np.array([0,0,h,0])
    h4 = np.array([0,0,0,h])

    # Build the dFx matrix

    dFx = np.zeros((4,4))

    # dFx[0,0] = 0 # dF1/dx1⋅dx1 = (F1(X,U)-F1(X-h1,U))/h = 0
    # dFx[0,1] = 0 # dF1/dx2⋅dx2 = (F1(X,U)-F1(X-h2,U))/h = 0
    dFx[0,2] = 1 # dF1/dx3⋅dx3 = (F1(X,U)-F1(X-h3,U))/h = 1
    # dFx[0,3] = 0 # dF1/dx4⋅dx4 = (F1(X,U)-F1(X-h4,U))/h = 0

    # dFx[1,0] = 0 # dF2/dx1⋅dx2 = (F2(X,U)-F2(X-h1,U))/h = 0
    # dFx[1,1] = 0 # dF2/dx2⋅dx2 = (F2(X,U)-F2(X-h2,U))/h = 0
    # dFx[1,2] = 0 # dF2/dx3⋅dx2 = (F2(X,U)-F2(X-h3,U))/h = 1
    dFx[1,3] = 1 # dF2/dx4⋅dx2 = (F2(X,U)-F2(X-h4,U))/h = 0

    # F3 is the acceleration of the cart.
    dFx[2,0] = (F3(X,U)-F3(X-h1,U))/h
    dFx[2,1] = (F3(X,U)-F3(X-h2,U))/h
    dFx[2,2] = (F3(X,U)-F3(X-h3,U))/h
    dFx[2,3] = (F3(X,U)-F3(X-h4,U))/h

    # F4 is the angular acceleration of the pendulum.
    dFx[3,0] = (F4(X,U)-F4(X-h1,U))/h
    dFx[3,1] = (F4(X,U)-F4(X-h2,U))/h
    dFx[3,2] = (F4(X,U)-F4(X-h3,U))/h
    dFx[3,3] = (F4(X,U)-F4(X-h4,U))/h

    Phi = np.matrix(np.eye(4) + dFx*dt)
    assert np.shape(Phi)==(4,4) \
        and str(type(Phi))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
    "Phi must be a (4,4) numpy matrix. Not " + str(type(Phi)) + " of shape " + str(np.shape(Phi)) + "."
    return(Phi)
def return_B(X,U,dt):
    """
    Takes in the state vector (X), the input vector (U) and returns the discretized and linearized input matrix, B.

    NOTE: Although you can spend the time to calculate the explicit definitions of the derivatives of the state equations. This is unnecessary for real time control (especially when the state equations may not be perfect to begin with!). Instead, we can approximate the derivative by the difference quotient. For explicit functions, please see functions below (unused).

    #######################
    ##### NEED TO DO: #####
    #######################

    [ ] - Create tests that ensure that X and U are the correct dimensions.
    [ ] - Create tests to make sure that the outputs are of the correct sizes.
    """
    assert (str(type(X)) in ["<class 'numpy.ndarray'>"]
            and np.shape(X)==(4,)), "Error with the type and shape of X ["+ fnState_And_Control_Transition_Matrices.__name__+"()]."
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."

    # Removed the U split into two scalars because U is already a scalar.

    h1 = np.array([h,0,0,0])
    h2 = np.array([0,h,0,0])
    h3 = np.array([0,0,h,0])
    h4 = np.array([0,0,0,h])

    #Build the dFu matrix

    dFu = np.zeros((4,1))

    dFu[0,0] = 0
    dFu[1,0] = 0
    dFu[2,0] = (F3(X,U)-F3(X,U-h))/h
    dFu[3,0] = (F4(X,U)-F4(X,U-h))/h

    B = np.matrix(dFu*dt)
    assert np.shape(B)==(4,1) \
            and str(type(B))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
        "B must be a (4,1) numpy matrix. Not " + str(type(B)) + " of shape " + str(np.shape(B)) + "."

    return(B)
def return_linearized_dynamics_matrices(X,U,dt):
    """
    Takes in the input U and the the corresponding output X, as well as dt and returns two lists that contain the linearized dynamic matrices for each timestep for range(len(Time)-1).

    Note that if np.shape(X)[1] = N and len(U) = M, then N = M + 1 (i.e., there is one more timestep for output than input since the initial conditions are assigned to the first state space timestep). Therefore, we only concern ourselves with the linearized dynamics of the (N-1) steps where U drives X to the next timestep (i.e., X will only go up to the N-1 step or index X[:,:-1].)

    Phi is a list of length len(Time)-1, each element with shape (n,n), where n is the number of states.

    B is a list of length len(Time)-1, each element with shape (n,m), where n is the number of states and m is the number of inputs.

    ### NEEDS TO BE TESTED ###

    np.shape(X)[1] == len(U)+1

    len(Phi) == len(U)
    type(Phi) == list
    len(B) == len(U)
    type(B) == list

    ##########################
    """
    Phi = list(
            map(
                lambda X,U: return_Phi(X,U,dt),
                X[:,:-1].T,
                U
            )
        )

    B = list(
            map(
                lambda X,U: return_B(X,U,dt),
                X[:,:-1].T,
                U
            )
        )
    return(Phi,B)
