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
            and np.shape(X)==(6,)), "Error with the type and shape of X ["+ return_Phi.__name__+"()]."
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."

    # Removed the U split into two scalars because U is already a scalar.

    h1 = np.array([h,0,0,0,0,0])
    h2 = np.array([0,h,0,0,0,0])
    h3 = np.array([0,0,h,0,0,0])
    h4 = np.array([0,0,0,h,0,0])
    h5 = np.array([0,0,0,0,h,0])
    h6 = np.array([0,0,0,0,0,h])

    # Build the dFx matrix

    dFx = np.zeros((6,6))

    # dFx[0,0] = 0 # dF1/dx1⋅dx1 = (F1(X,U)-F1(X-h1,U))/h = 0
    # dFx[0,1] = 0 # dF1/dx2⋅dx2 = (F1(X,U)-F1(X-h2,U))/h = 0
    # dFx[0,2] = 0 # dF1/dx3⋅dx3 = (F1(X,U)-F1(X-h3,U))/h = 0
    dFx[0,3] = 1 # dF1/dx4⋅dx4 = (F1(X,U)-F1(X-h4,U))/h = 1
    # dFx[0,4] = 0 # dF1/dx5⋅dx5 = (F1(X,U)-F1(X-h5,U))/h = 0
    # dFx[0,5] = 0 # dF1/dx6⋅dx6 = (F1(X,U)-F1(X-h6,U))/h = 0

    # dFx[1,0] = 0 # dF2/dx1⋅dx1 = (F2(X,U)-F2(X-h1,U))/h = 0
    # dFx[1,1] = 0 # dF2/dx2⋅dx2 = (F2(X,U)-F2(X-h2,U))/h = 0
    # dFx[1,2] = 0 # dF2/dx3⋅dx3 = (F2(X,U)-F2(X-h3,U))/h = 0
    # dFx[1,3] = 0 # dF2/dx4⋅dx4 = (F2(X,U)-F2(X-h4,U))/h = 0
    dFx[1,4] = 1 # dF2/dx5⋅dx5 = (F2(X,U)-F2(X-h5,U))/h = 1
    # dFx[1,5] = 0 # dF2/dx6⋅dx6 = (F2(X,U)-F2(X-h6,U))/h = 0

    # dFx[2,0] = 0 # dF3/dx1⋅dx1 = (F3(X,U)-F3(X-h1,U))/h = 0
    # dFx[2,1] = 0 # dF3/dx2⋅dx2 = (F3(X,U)-F3(X-h2,U))/h = 0
    # dFx[2,2] = 0 # dF3/dx3⋅dx3 = (F3(X,U)-F3(X-h3,U))/h = 0
    # dFx[2,3] = 0 # dF3/dx4⋅dx4 = (F3(X,U)-F3(X-h4,U))/h = 0
    # dFx[2,4] = 0 # dF3/dx5⋅dx5 = (F3(X,U)-F3(X-h5,U))/h = 0
    dFx[2,5] = 1 # dF3/dx6⋅dx6 = (F3(X,U)-F3(X-h6,U))/h = 1

    # F4 is the acceleration of the cart.
    dFx[3,0] = (F4(X,U)-F4(X-h1,U))/h
    dFx[3,1] = (F4(X,U)-F4(X-h2,U))/h
    dFx[3,2] = (F4(X,U)-F4(X-h3,U))/h
    dFx[3,3] = (F4(X,U)-F4(X-h4,U))/h
    dFx[3,4] = (F4(X,U)-F4(X-h5,U))/h
    dFx[3,5] = (F4(X,U)-F4(X-h6,U))/h

    # F5 is the proximal angular acceleration of the pendulum.
    dFx[4,0] = (F5(X,U)-F5(X-h1,U))/h
    dFx[4,1] = (F5(X,U)-F5(X-h2,U))/h
    dFx[4,2] = (F5(X,U)-F5(X-h3,U))/h
    dFx[4,3] = (F5(X,U)-F5(X-h4,U))/h
    dFx[4,4] = (F5(X,U)-F5(X-h5,U))/h
    dFx[4,5] = (F5(X,U)-F5(X-h6,U))/h

    # F6 is the distal angular acceleration of the pendulum.
    dFx[5,0] = (F6(X,U)-F6(X-h1,U))/h
    dFx[5,1] = (F6(X,U)-F6(X-h2,U))/h
    dFx[5,2] = (F6(X,U)-F6(X-h3,U))/h
    dFx[5,3] = (F6(X,U)-F6(X-h4,U))/h
    dFx[5,4] = (F6(X,U)-F6(X-h5,U))/h
    dFx[5,5] = (F6(X,U)-F6(X-h6,U))/h

    Phi = np.matrix(np.eye(6) + dFx*dt)
    assert np.shape(Phi)==(6,6) \
        and str(type(Phi))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
    "Phi must be a (6,6) numpy matrix. Not " + str(type(Phi)) + " of shape " + str(np.shape(Phi)) + "."
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
            and np.shape(X)==(6,)), "Error with the type and shape of X ["+ return_B.__name__+"()]."
    assert str(type(U)) in ["<class 'int'>",
            "<class 'float'>",
            "<class 'numpy.float'>",
            "<class 'numpy.float64'>",
            "<class 'numpy.int32'>",
            "<class 'numpy.int64'>"],\
        "U must be a number. Not " + str(type(U)) + "."

    # Removed the U split into two scalars because U is already a scalar.

    #Build the dFu matrix

    dFu = np.zeros((6,1))

    dFu[0,0] = 0
    dFu[1,0] = 0
    dFu[2,0] = 0
    dFu[3,0] = (F4(X,U)-F4(X,U-h))/h
    dFu[4,0] = (F5(X,U)-F5(X,U-h))/h
    dFu[5,0] = (F6(X,U)-F6(X,U-h))/h

    B = np.matrix(dFu*dt)
    assert np.shape(B)==(6,1) \
            and str(type(B))=="<class 'numpy.matrixlib.defmatrix.matrix'>", \
        "B must be a (6,1) numpy matrix. Not " + str(type(B)) + " of shape " + str(np.shape(B)) + "."

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
                U.T
            )
        )

    B = list(
            map(
                lambda X,U: return_B(X,U,dt),
                X[:,:-1].T,
                U.T
            )
        )
    return(Phi,B)
